from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_ball_tracking import _extract_ball_candidates, _get_ball_tracking_profile, _pick_best_candidate
from backend.rally_timeline_contract import Correction, RallyTimelinePoint, load_rally_timeline, save_rally_timeline


def _frozen_boundary_signature(points: Iterable[RallyTimelinePoint]) -> list[tuple[str, float, float]]:
    return [(str(point.id), float(point.t_start), float(point.t_end)) for point in points]


def _winner_window(
    point: RallyTimelinePoint,
    *,
    ratio: float,
    full_rally_threshold_sec: float,
    min_window_sec: float,
    max_window_sec: float,
) -> tuple[float, float]:
    # Winner inference now always sees the full frozen rally clip.
    # Keep the legacy parameters in the function signature so older CLI calls
    # remain compatible, but ignore them here on purpose.
    _ = (ratio, full_rally_threshold_sec, min_window_sec, max_window_sec)
    rally_start = float(point.t_start)
    rally_end = float(max(point.t_end, point.t_start + 0.01))
    return float(rally_start), float(rally_end)


def _selected_point_ids(raw_values: Iterable[str]) -> set[str]:
    selected: set[str] = set()
    for raw in raw_values:
        for item in str(raw).split(","):
            item = item.strip()
            if item:
                selected.add(item)
    return selected


def _extract_winner_label(text: str) -> str:
    raw = str(text or "").strip().lower()
    labeled = re.search(r'["\']?winner["\']?\s*[:=]\s*["\']?(player_a|player_b)["\']?\b', raw)
    if labeled:
        return labeled.group(1)
    match = re.search(r"\b(player_a|player_b)\b", raw)
    if match:
        return match.group(1)
    if "near" in raw and "far" not in raw:
        return "player_a"
    if "far" in raw and "near" not in raw:
        return "player_b"
    return "unknown"


def _extract_field_label(text: str, field_name: str) -> str:
    raw = str(text or "").strip().lower()
    field_pattern = re.escape(field_name.lower()).replace(r"\_", r"[_\s]*")
    match = re.search(rf'["\']?{field_pattern}["\']?\s*[:=]\s*["\']?(player_a|player_b)["\']?\b', raw)
    if match:
        return match.group(1)
    return "unknown"


def _extract_comparative_winner(text: str) -> str:
    winner = _extract_field_label(text, "winner")
    if winner in {"player_a", "player_b"}:
        return winner
    loser = _extract_field_label(text, "loser")
    if loser == "player_a":
        return "player_b"
    if loser == "player_b":
        return "player_a"
    return _extract_winner_label(text)


def _augmented_var_prompt(clip_scope: str) -> str:
    return (
        f"This {'full table-tennis rally' if clip_scope == 'full' else 'table-tennis rally segment'} may include visual overlays. "
        "If present: the green box marks the table and the red trail marks recent ball motion. "
        "Player A is the near-side player. Player B is the far-side player. "
        "Analyze the overlays first, then use player reaction only as secondary evidence. "
        "Return strict JSON with these fields only: "
        '{"last_hitter":"player_a|player_b|unknown","trail_end":"in|out|unclear","reaction_support":"player_a|player_b|neutral","winner":"player_a|player_b|unknown","confidence":0.0}.'
    )


def _overlay_prompt_prefix(branch_overlay: str, clip_scope: str) -> str:
    clip_desc = "full table-tennis rally" if clip_scope == "full" else "table-tennis rally segment"
    if branch_overlay == "augmented_v1":
        return (
            f"This {clip_desc} may include visual overlays. "
            "If present: the green box marks the table and the red trail marks recent ball motion. "
            "Use those overlays first, and use player posture only as secondary evidence. "
        )
    if branch_overlay == "augmented_v2":
        return (
            f"This {clip_desc} may include visual overlays. "
            "If present: the green box marks the table. "
            "The ball trail is shown with hollow markers that preserve the original ball pixels. "
            "Yellow markers are older, orange markers are middle, and red markers are newest. "
            "Trust the red newest markers most when judging whether the final touch landed legally or went out. "
            "Use those overlays first, and use player posture only as secondary evidence. "
        )
    return ""


def _extract_yes_no(text: str) -> str:
    raw = str(text or "").strip().lower()
    if re.search(r"\byes\b", raw):
        return "yes"
    if re.search(r"\bno\b", raw):
        return "no"
    return "unknown"


def _extract_outcome_label(text: str) -> str:
    raw = str(text or "").strip().lower()
    match = re.search(r"\boutcome\s*[:=]\s*(legal|out|unclear)\b", raw)
    if match:
        return match.group(1)
    if re.search(r"\b(go(?:es)? out|out of bounds|outside the table|miss(?:es|ed)? the table|went out)\b", raw):
        return "out"
    if re.search(r"\b(lands? legally|lands? on the table|legal return|hits? the table)\b", raw):
        return "legal"
    return "unclear"


_END_CATEGORIES = {
    "clean_winner_no_touch",
    "touched_but_out",
    "touched_but_no_net_cross",
    "attacker_direct_out",
    "attacker_into_net",
    "double_bounce_before_return",
    "ball_hits_player_or_body",
    "ball_hits_non_racket_object",
    "illegal_or_mishit_return",
    "blocked_by_visibility",
    "ambiguous_review",
}


def _map_vertical_position_to_player(label: str) -> str:
    if label == "top":
        return "player_b"
    if label == "bottom":
        return "player_a"
    return "unknown"


def _extract_labeled_choice(text: str, field_name: str, choices: set[str]) -> str:
    raw = str(text or "").strip().lower()
    field_pattern = re.escape(field_name.lower()).replace(r"\_", r"[_\s]*")
    match = re.search(rf'["\']?{field_pattern}["\']?\s*[:=]\s*["\']?([a-z0-9_]+)["\']?\b', raw)
    if match:
        value = str(match.group(1)).strip().lower()
        if value in choices:
            return value
    return "unknown"


def _extract_end_category(text: str) -> str:
    raw = str(text or "").strip().lower()
    direct = _extract_labeled_choice(raw, "category", _END_CATEGORIES)
    if direct in _END_CATEGORIES:
        return direct
    taxonomy = _extract_labeled_choice(raw, "taxonomy", _END_CATEGORIES)
    if taxonomy in _END_CATEGORIES:
        return taxonomy
    for category in _END_CATEGORIES:
        if category in raw:
            return category
    return "ambiguous_review"


def _extract_confidence_value(text: str) -> float | None:
    raw = str(text or "").strip().lower()
    match = re.search(r'["\']?confidence["\']?\s*[:=]\s*["\']?([0-9]+(?:\.[0-9]+)?)["\']?\b', raw)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return max(0.0, min(1.0, value))


def _opposite_player(label: str) -> str:
    if label == "player_a":
        return "player_b"
    if label == "player_b":
        return "player_a"
    return "unknown"


def _category_schema_prompt(clip_scope: str, branch_overlay: str) -> str:
    clip_desc = "full table-tennis rally" if clip_scope == "full" else "table-tennis rally segment"
    return (
        _overlay_prompt_prefix(branch_overlay, clip_scope)
        + f"This {clip_desc} ends with one final losing event. "
        + "Player A is the near-side player. Player B is the far-side player. "
        + "Choose exactly one category from: "
        + "clean_winner_no_touch, touched_but_out, touched_but_no_net_cross, "
        + "attacker_direct_out, attacker_into_net, double_bounce_before_return, "
        + "ball_hits_player_or_body, ball_hits_non_racket_object, "
        + "illegal_or_mishit_return, blocked_by_visibility, ambiguous_review. "
        + "Definitions: clean_winner_no_touch = the loser cannot touch a legal shot. "
        + "touched_but_out = the loser touches the ball but sends it out. "
        + "touched_but_no_net_cross = the loser touches the ball but the return does not cross the net. "
        + "attacker_direct_out = the attacking player hits directly out. "
        + "attacker_into_net = the attacking player hits into the net or fails to cross. "
        + "double_bounce_before_return = the loser allows a second bounce before making a legal return. "
        + "ball_hits_player_or_body = the ball hits the losing player's body. "
        + "ball_hits_non_racket_object = the ball hits a non-racket object on the losing side such as hand, clothing, or another object. "
        + "illegal_or_mishit_return = another clear illegal final return. "
        + "blocked_by_visibility = the point likely has a real category but the decisive contact is hidden or too unclear in the video. "
        + "ambiguous_review = the evidence is still unclear. "
        + "Answer with exactly these labels in one short block: "
        + "Loser=player_a or player_b or unknown; "
        + "LastHitter=player_a or player_b or unknown; "
        + "Category=<one category>; "
        + "Winner=player_a or player_b or unknown; "
        + "Confidence=0.00; "
        + "Reason=short phrase."
    )


def _category_schema_rules_v2_prompt(clip_scope: str, branch_overlay: str) -> str:
    clip_desc = "full table-tennis rally" if clip_scope == "full" else "table-tennis rally segment"
    return (
        _overlay_prompt_prefix(branch_overlay, clip_scope)
        + f"This {clip_desc} ends with one final losing event. "
        + "Player A is the near-side player. Player B is the far-side player. "
        + "Judge only the final losing event near the end of the rally. "
        + "Rules: "
        + "If the loser never touches the final legal shot, use clean_winner_no_touch. "
        + "If the loser touches the ball but sends it out, use touched_but_out. "
        + "If the loser touches the ball but the return does not cross the net, use touched_but_no_net_cross. "
        + "If the final attacker hits directly out, use attacker_direct_out. "
        + "If the final attacker hits into the net or fails to cross, use attacker_into_net. "
        + "If the loser lets the ball bounce twice before a legal return, use double_bounce_before_return. "
        + "If the ball hits the losing player's body, use ball_hits_player_or_body. "
        + "If the ball hits a non-racket object on the losing side, use ball_hits_non_racket_object. "
        + "If the decisive contact is hidden, use blocked_by_visibility. "
        + "If still unclear, use ambiguous_review. "
        + "Important consistency rules: "
        + "For clean_winner_no_touch, last_hitter is the winner. "
        + "For touched_but_out or touched_but_no_net_cross, the loser makes the final touch and the winner is the opponent. "
        + "For attacker_direct_out or attacker_into_net, the last_hitter is the loser and the winner is the opponent. "
        + "Answer with exactly these labels in one short block: "
        + "Loser=player_a or player_b or unknown; "
        + "LastHitter=player_a or player_b or unknown; "
        + "LoserTouchedFinalBall=yes or no or unknown; "
        + "FinalReturnCrossedNet=yes or no or unknown; "
        + "FinalReturnLandedIn=yes or no or unknown; "
        + "Category=<one category>; "
        + "Winner=player_a or player_b or unknown; "
        + "Confidence=0.00; "
        + "Reason=short phrase."
    )


def _category_schema_anchor4_prompt(clip_scope: str, branch_overlay: str) -> str:
    clip_desc = "full table-tennis rally" if clip_scope == "full" else "table-tennis rally segment"
    return (
        _overlay_prompt_prefix(branch_overlay, clip_scope)
        + f"This {clip_desc} ends with one final losing event. "
        + "Player A is the near-side player. Player B is the far-side player. "
        + "For this task, choose exactly one category from only these four options: "
        + "clean_winner_no_touch, touched_but_out, touched_but_no_net_cross, attacker_direct_out. "
        + "Definitions: "
        + "clean_winner_no_touch = the winner hits a legal shot and the loser never touches the ball. "
        + "touched_but_out = the loser touches the final ball but sends it out. "
        + "touched_but_no_net_cross = the loser touches the final ball but the return does not cross the net. "
        + "attacker_direct_out = the attacker makes the final shot and sends it directly out. "
        + "Consistency rules: "
        + "For clean_winner_no_touch, last_hitter is the winner. "
        + "For touched_but_out or touched_but_no_net_cross, the loser makes the final touch and the winner is the opponent. "
        + "For attacker_direct_out, the loser makes the final touch and the winner is the opponent. "
        + "Answer with exactly these labels in one short block: "
        + "Loser=player_a or player_b or unknown; "
        + "LastHitter=player_a or player_b or unknown; "
        + "LoserTouchedFinalBall=yes or no or unknown; "
        + "FinalReturnCrossedNet=yes or no or unknown; "
        + "FinalReturnLandedIn=yes or no or unknown; "
        + "Category=clean_winner_no_touch or touched_but_out or touched_but_no_net_cross or attacker_direct_out; "
        + "Winner=player_a or player_b or unknown; "
        + "Confidence=0.00; "
        + "Reason=short phrase."
    )


def _category_schema_taxonomy_first_anchor4_prompt(clip_scope: str, branch_overlay: str) -> str:
    clip_desc = "full table-tennis rally" if clip_scope == "full" else "table-tennis rally segment"
    return (
        _overlay_prompt_prefix(branch_overlay, clip_scope)
        + f"This {clip_desc} ends with one final losing event. "
        + "Player A is the near-side player. Player B is the far-side player. "
        + "Decide the taxonomy first, then loser, then winner. "
        + "Use only these four taxonomy labels unless the evidence is still unclear: "
        + "clean_winner_no_touch, touched_but_out, touched_but_no_net_cross, attacker_direct_out, ambiguous_review. "
        + "Definitions: "
        + "clean_winner_no_touch = the winner hits a legal shot and the loser never touches the final ball. "
        + "touched_but_out = the loser touches the final ball but sends the return out. "
        + "touched_but_no_net_cross = the loser touches the final ball but the return does not cross the net. "
        + "attacker_direct_out = the attacker makes the final shot and sends it directly out. "
        + "Examples: "
        + "If Player A attacks, Player B touches the ball, and Player B sends it out, then taxonomy=touched_but_out, loser=player_b, last_hitter=player_b, winner=player_a. "
        + "If Player B attacks directly out, then taxonomy=attacker_direct_out, loser=player_b, last_hitter=player_b, winner=player_a. "
        + "If Player A touches the final ball but it does not cross the net, then taxonomy=touched_but_no_net_cross, loser=player_a, last_hitter=player_a, winner=player_b. "
        + "If Player A hits a clean winner and Player B never touches the final ball, then taxonomy=clean_winner_no_touch, loser=player_b, last_hitter=player_a, winner=player_a. "
        + "The winner can be Player A or Player B. Do not default to either side. "
        + "Do not decide from body language alone. Focus on the final losing event. "
        + "Return strict JSON only with these keys: "
        + '{"taxonomy":"clean_winner_no_touch|touched_but_out|touched_but_no_net_cross|attacker_direct_out|ambiguous_review","loser":"player_a|player_b|unknown","winner":"player_a|player_b|unknown","last_hitter":"player_a|player_b|unknown","loser_touched_final_ball":"yes|no|unknown","final_return_crossed_net":"yes|no|unknown","final_return_landed_in":"yes|no|unknown","confidence":0.0,"reason":"short phrase"}.'
    )


def _load_taxonomy_fewshot_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = str(line).strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _build_taxonomy_fewshot_examples_text(
    rows: list[dict[str, object]],
    *,
    exclude_point_id: str,
    max_examples: int,
) -> str:
    confirmed_rows = [
        row
        for row in rows
        if str(row.get("id", "")).strip() != str(exclude_point_id).strip()
        and str(row.get("winner_label_status", "")).strip().lower() == "confirmed"
        and str(row.get("taxonomy_label_status", "")).strip().lower() == "confirmed"
        and str(row.get("taxonomy", "")).strip()
        and str(row.get("winner", "")).strip() in {"player_a", "player_b"}
        and str(row.get("loser", "")).strip() in {"player_a", "player_b"}
        and str(row.get("last_hitter", "")).strip() in {"player_a", "player_b"}
    ]
    if max_examples > 0:
        confirmed_rows = confirmed_rows[:max_examples]
    if not confirmed_rows:
        return ""
    parts: list[str] = []
    for idx, row in enumerate(confirmed_rows, start=1):
        note = str(row.get("note", "")).strip()
        note_text = f" note={note}." if note else ""
        parts.append(
            f"Example {idx}: id={row['id']}; taxonomy={row['taxonomy']}; "
            f"loser={row['loser']}; winner={row['winner']}; last_hitter={row['last_hitter']};"
            f"{note_text}"
        )
    return "Reviewed examples: " + " ".join(parts) + " "


def _category_schema_taxonomy_first_anchor4_fewshot_prompt(
    clip_scope: str,
    branch_overlay: str,
    *,
    fewshot_examples_text: str,
) -> str:
    clip_desc = "full table-tennis rally" if clip_scope == "full" else "table-tennis rally segment"
    return (
        _overlay_prompt_prefix(branch_overlay, clip_scope)
        + f"This {clip_desc} ends with one final losing event. "
        + "Player A is the near-side player. Player B is the far-side player. "
        + "Decide the taxonomy first, then loser, then winner. "
        + "Use only these four taxonomy labels unless the evidence is still unclear: "
        + "clean_winner_no_touch, touched_but_out, touched_but_no_net_cross, attacker_direct_out, ambiguous_review. "
        + "Definitions: "
        + "clean_winner_no_touch = the winner hits a legal shot and the loser never touches the final ball. "
        + "touched_but_out = the loser touches the final ball but sends the return out. "
        + "touched_but_no_net_cross = the loser touches the final ball but the return does not cross the net. "
        + "attacker_direct_out = the attacker makes the final shot and sends it directly out. "
        + "Treat the following reviewed examples as authoritative taxonomy references. "
        + str(fewshot_examples_text)
        + "The winner can be Player A or Player B. Do not default to either side. "
        + "Do not decide from body language alone. Focus on the final losing event. "
        + "Return strict JSON only with these keys: "
        + '{"taxonomy":"clean_winner_no_touch|touched_but_out|touched_but_no_net_cross|attacker_direct_out|ambiguous_review","loser":"player_a|player_b|unknown","winner":"player_a|player_b|unknown","last_hitter":"player_a|player_b|unknown","loser_touched_final_ball":"yes|no|unknown","final_return_crossed_net":"yes|no|unknown","final_return_landed_in":"yes|no|unknown","confidence":0.0,"reason":"short phrase"}.'
    )


def _category_schema_taxonomy_first_topbottom_anchor4_prompt(clip_scope: str, branch_overlay: str) -> str:
    clip_desc = "full table-tennis rally" if clip_scope == "full" else "table-tennis rally segment"
    return (
        _overlay_prompt_prefix(branch_overlay, clip_scope)
        + f"This {clip_desc} ends with one final losing event. "
        + "The TOP player is the far-side player. The BOTTOM player is the near-side player. "
        + "Decide the taxonomy first, then loser position, then winner position. "
        + "Use only these four taxonomy labels unless the evidence is still unclear: "
        + "clean_winner_no_touch, touched_but_out, touched_but_no_net_cross, attacker_direct_out, ambiguous_review. "
        + "Definitions: "
        + "clean_winner_no_touch = the winner hits a legal shot and the loser never touches the final ball. "
        + "touched_but_out = the loser touches the final ball but sends the return out. "
        + "touched_but_no_net_cross = the loser touches the final ball but the return does not cross the net. "
        + "attacker_direct_out = the attacker makes the final shot and sends it directly out. "
        + "Examples: "
        + "If the BOTTOM player attacks, the TOP player touches the ball, and the TOP player sends it out, then taxonomy=touched_but_out, loser_position=top, last_hitter_position=top, winner_position=bottom. "
        + "If the TOP player attacks directly out, then taxonomy=attacker_direct_out, loser_position=top, last_hitter_position=top, winner_position=bottom. "
        + "If the BOTTOM player hits a clean winner and the TOP player never touches the final ball, then taxonomy=clean_winner_no_touch, loser_position=top, last_hitter_position=bottom, winner_position=bottom. "
        + "If the TOP player touches the final ball but it does not cross the net, then taxonomy=touched_but_no_net_cross, loser_position=top, last_hitter_position=top, winner_position=bottom. "
        + "The winner can be TOP or BOTTOM. Do not default to either side. "
        + "Do not decide from body language alone. Focus on the final losing event. "
        + "Return strict JSON only with these keys: "
        + '{"taxonomy":"clean_winner_no_touch|touched_but_out|touched_but_no_net_cross|attacker_direct_out|ambiguous_review","loser_position":"top|bottom|unknown","winner_position":"top|bottom|unknown","last_hitter_position":"top|bottom|unknown","loser_touched_final_ball":"yes|no|unknown","final_return_crossed_net":"yes|no|unknown","final_return_landed_in":"yes|no|unknown","confidence":0.0,"reason":"short phrase"}.'
    )


def _category_schema_touchprobe_topbottom_anchor4_prompt(clip_scope: str, branch_overlay: str) -> str:
    clip_desc = "full table-tennis rally" if clip_scope == "full" else "table-tennis rally segment"
    return (
        _overlay_prompt_prefix(branch_overlay, clip_scope)
        + f"This {clip_desc} ends with one final losing event. "
        + "The TOP player is the far-side player. The BOTTOM player is the near-side player. "
        + "First decide who touched the final ball. Then choose the taxonomy. Then choose loser and winner. "
        + "Use only these taxonomy labels unless the evidence is still unclear: "
        + "clean_winner_no_touch, touched_but_out, touched_but_no_net_cross, attacker_direct_out, ambiguous_review. "
        + "Definitions: "
        + "clean_winner_no_touch = the winner hits a legal shot and the loser never touches the final ball. "
        + "touched_but_out = the loser touches the final ball but sends the return out. "
        + "touched_but_no_net_cross = the loser touches the final ball but the return does not cross the net. "
        + "attacker_direct_out = the attacker makes the final shot and sends it directly out without a losing touch by the opponent. "
        + "Important: if only one player touches the final ball, say so clearly. "
        + "Return strict JSON only with these keys: "
        + '{"top_touched_final_ball":"yes|no|unknown","bottom_touched_final_ball":"yes|no|unknown","final_touch_position":"top|bottom|unknown","taxonomy":"clean_winner_no_touch|touched_but_out|touched_but_no_net_cross|attacker_direct_out|ambiguous_review","loser_position":"top|bottom|unknown","winner_position":"top|bottom|unknown","confidence":0.0,"reason":"short phrase"}.'
    )


def _normalize_category_decision(
    *,
    winner: str,
    loser: str,
    last_hitter: str,
    category: str,
) -> tuple[str, str, str]:
    normalized_winner = winner if winner in {"player_a", "player_b"} else "unknown"
    normalized_loser = loser if loser in {"player_a", "player_b"} else "unknown"
    normalized_last_hitter = last_hitter if last_hitter in {"player_a", "player_b"} else "unknown"

    if category == "clean_winner_no_touch":
        if normalized_winner == "unknown" and normalized_last_hitter in {"player_a", "player_b"}:
            normalized_winner = normalized_last_hitter
        if normalized_loser == "unknown" and normalized_winner in {"player_a", "player_b"}:
            normalized_loser = _opposite_player(normalized_winner)
    elif category in {"touched_but_out", "touched_but_no_net_cross"}:
        if normalized_winner == "unknown" and normalized_loser in {"player_a", "player_b"}:
            normalized_winner = _opposite_player(normalized_loser)
        if normalized_loser == "unknown" and normalized_winner in {"player_a", "player_b"}:
            normalized_loser = _opposite_player(normalized_winner)
    elif category in {"attacker_direct_out", "attacker_into_net"}:
        if normalized_loser == "unknown" and normalized_last_hitter in {"player_a", "player_b"}:
            normalized_loser = normalized_last_hitter
        if normalized_winner == "unknown" and normalized_loser in {"player_a", "player_b"}:
            normalized_winner = _opposite_player(normalized_loser)
        if normalized_last_hitter == "unknown" and normalized_loser in {"player_a", "player_b"}:
            normalized_last_hitter = normalized_loser
    else:
        if normalized_winner == "unknown" and normalized_loser in {"player_a", "player_b"}:
            normalized_winner = _opposite_player(normalized_loser)
        if normalized_loser == "unknown" and normalized_winner in {"player_a", "player_b"}:
            normalized_loser = _opposite_player(normalized_winner)

    return normalized_winner, normalized_loser, normalized_last_hitter


def _build_category_prompt_for_family(
    *,
    prompt_family_name: str,
    clip_scope: str,
    branch_overlay: str,
    point_id: str,
    fewshot_rows: list[dict[str, object]],
    fewshot_max_examples: int,
) -> str:
    if prompt_family_name == "category_schema_touchprobe_topbottom_anchor4":
        return _category_schema_touchprobe_topbottom_anchor4_prompt(clip_scope, branch_overlay)
    if prompt_family_name == "category_schema_taxonomy_first_topbottom_anchor4":
        return _category_schema_taxonomy_first_topbottom_anchor4_prompt(clip_scope, branch_overlay)
    if prompt_family_name == "category_schema_taxonomy_first_anchor4_fewshot":
        fewshot_examples_text = _build_taxonomy_fewshot_examples_text(
            fewshot_rows,
            exclude_point_id=point_id,
            max_examples=fewshot_max_examples,
        )
        return _category_schema_taxonomy_first_anchor4_fewshot_prompt(
            clip_scope,
            branch_overlay,
            fewshot_examples_text=fewshot_examples_text,
        )
    if prompt_family_name == "category_schema_taxonomy_first_anchor4":
        return _category_schema_taxonomy_first_anchor4_prompt(clip_scope, branch_overlay)
    if prompt_family_name == "category_schema_anchor4":
        return _category_schema_anchor4_prompt(clip_scope, branch_overlay)
    if prompt_family_name == "category_schema_rules_v2":
        return _category_schema_rules_v2_prompt(clip_scope, branch_overlay)
    return _category_schema_prompt(clip_scope, branch_overlay)


def _category_consistency_flags(
    *,
    winner: str,
    loser: str,
    last_hitter: str,
    category: str,
    loser_touched_final_ball: str = "unknown",
    final_return_crossed_net: str = "unknown",
    final_return_landed_in: str = "unknown",
) -> list[str]:
    flags: list[str] = []
    if category not in _END_CATEGORIES:
        flags.append("winner_category_unknown_label")
        return flags

    if winner in {"player_a", "player_b"} and loser in {"player_a", "player_b"}:
        if winner == loser:
            flags.append("winner_category_inconsistent_same_winner_loser")
        elif _opposite_player(winner) != loser:
            flags.append("winner_category_inconsistent_winner_loser_pair")

    if category == "clean_winner_no_touch":
        if loser_touched_final_ball == "yes":
            flags.append("winner_category_inconsistent_clean_touch_flag")
        if loser in {"player_a", "player_b"} and last_hitter in {"player_a", "player_b"} and loser == last_hitter:
            flags.append("winner_category_inconsistent_clean_last_hitter")
    elif category == "touched_but_out":
        if loser_touched_final_ball == "no":
            flags.append("winner_category_inconsistent_touch_out_touch_flag")
        if final_return_landed_in == "yes":
            flags.append("winner_category_inconsistent_touch_out_landed_in")
        if loser in {"player_a", "player_b"} and last_hitter in {"player_a", "player_b"} and loser != last_hitter:
            flags.append("winner_category_inconsistent_touch_out_last_hitter")
    elif category == "touched_but_no_net_cross":
        if loser_touched_final_ball == "no":
            flags.append("winner_category_inconsistent_no_net_touch_flag")
        if final_return_crossed_net == "yes":
            flags.append("winner_category_inconsistent_no_net_cross_flag")
        if loser in {"player_a", "player_b"} and last_hitter in {"player_a", "player_b"} and loser != last_hitter:
            flags.append("winner_category_inconsistent_no_net_last_hitter")
    elif category == "attacker_direct_out":
        if final_return_landed_in == "yes":
            flags.append("winner_category_inconsistent_attack_out_landed_in")
        if loser in {"player_a", "player_b"} and last_hitter in {"player_a", "player_b"} and loser != last_hitter:
            flags.append("winner_category_inconsistent_attack_out_last_hitter")

    return flags


def _slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(model_name).strip().lower())
    slug = slug.strip("_")
    return slug or "native_video_model"


def _clip_window_video(
    *,
    source_video: str,
    clip_path: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        source_video,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        str(clip_path),
    ]
    subprocess.run(cmd, check=True)


def _video_dimensions(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    raw = result.stdout.strip()
    width_str, height_str = raw.split("x", 1)
    return int(width_str), int(height_str)


def _build_composite_clip(
    *,
    source_clip: Path,
    composite_clip: Path,
    roi: dict,
) -> None:
    video_w, video_h = _video_dimensions(source_clip)
    roi_x = int(roi.get("x", 0))
    roi_y = int(roi.get("y", 0))
    roi_w = int(roi.get("w", max(1, video_w // 4)))
    roi_h = int(roi.get("h", max(1, video_h // 8)))
    crop_w = min(video_w, max(1400, int(round(roi_w * 2.0))))
    crop_h = min(video_h, max(900, int(round(roi_h * 4.0))))
    center_x = roi_x + roi_w / 2.0
    center_y = roi_y + roi_h / 2.0
    crop_x = max(0, min(video_w - crop_w, int(round(center_x - crop_w / 2.0))))
    crop_y = max(0, min(video_h - crop_h, int(round(center_y - crop_h / 2.0))))

    composite_clip.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_clip),
        "-filter_complex",
        (
            f"[0:v]scale=960:540[left];"
            f"[0:v]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale=960:540[right];"
            f"[left][right]hstack=inputs=2[v]"
        ),
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        str(composite_clip),
    ]
    subprocess.run(cmd, check=True)


def _compute_roi_crop_bounds(
    *,
    video_w: int,
    video_h: int,
    roi: dict,
    margin_ratio: float,
    margin_y_ratio: float,
) -> tuple[int, int, int, int]:
    roi_x = int(roi.get("x", 0))
    roi_y = int(roi.get("y", 0))
    roi_w = int(roi.get("w", max(1, video_w // 4)))
    roi_h = int(roi.get("h", max(1, video_h // 8)))
    expand_w = int(round(roi_w * float(margin_ratio)))
    expand_h = int(round(roi_h * float(margin_y_ratio)))
    crop_x = max(0, roi_x - expand_w)
    crop_y = max(0, roi_y - expand_h)
    crop_w = min(video_w - crop_x, roi_w + (2 * expand_w))
    crop_h = min(video_h - crop_y, roi_h + (2 * expand_h))
    return crop_x, crop_y, crop_w, crop_h


def _build_roi_clip(
    *,
    source_clip: Path,
    roi_clip: Path,
    roi: dict,
    margin_ratio: float,
    margin_y_ratio: float,
) -> None:
    video_w, video_h = _video_dimensions(source_clip)
    crop_x, crop_y, crop_w, crop_h = _compute_roi_crop_bounds(
        video_w=video_w,
        video_h=video_h,
        roi=roi,
        margin_ratio=margin_ratio,
        margin_y_ratio=margin_y_ratio,
    )

    roi_clip.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_clip),
        "-filter:v",
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        str(roi_clip),
    ]
    subprocess.run(cmd, check=True)


def _compute_table_only_crop_bounds(
    *,
    video_w: int,
    video_h: int,
    roi: dict,
    x_margin_ratio: float,
    top_margin_ratio: float,
    bottom_margin_ratio: float,
) -> tuple[int, int, int, int]:
    roi_x = int(roi.get("x", 0))
    roi_y = int(roi.get("y", 0))
    roi_w = int(roi.get("w", max(1, video_w // 4)))
    roi_h = int(roi.get("h", max(1, video_h // 8)))
    expand_x = int(round(roi_w * float(x_margin_ratio)))
    expand_top = int(round(roi_h * float(top_margin_ratio)))
    expand_bottom = int(round(roi_h * float(bottom_margin_ratio)))
    crop_x = max(0, roi_x - expand_x)
    crop_y = max(0, roi_y - expand_top)
    crop_x2 = min(video_w, roi_x + roi_w + expand_x)
    crop_y2 = min(video_h, roi_y + roi_h + expand_bottom)
    crop_w = max(1, crop_x2 - crop_x)
    crop_h = max(1, crop_y2 - crop_y)
    return crop_x, crop_y, crop_w, crop_h


def _build_table_only_clip(
    *,
    source_clip: Path,
    table_only_clip: Path,
    roi: dict,
    x_margin_ratio: float,
    top_margin_ratio: float,
    bottom_margin_ratio: float,
) -> None:
    video_w, video_h = _video_dimensions(source_clip)
    crop_x, crop_y, crop_w, crop_h = _compute_table_only_crop_bounds(
        video_w=video_w,
        video_h=video_h,
        roi=roi,
        x_margin_ratio=x_margin_ratio,
        top_margin_ratio=top_margin_ratio,
        bottom_margin_ratio=bottom_margin_ratio,
    )

    table_only_clip.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_clip),
        "-filter:v",
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        str(table_only_clip),
    ]
    subprocess.run(cmd, check=True)


def _draw_table_overlay(frame: np.ndarray, table_xywh: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = table_xywh
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (60, 180, 60), thickness=-1)
    blended = cv2.addWeighted(overlay, 0.14, frame, 0.86, 0.0)
    cv2.rectangle(blended, (x, y), (x + w, y + h), (70, 255, 70), thickness=3)
    return blended


def _draw_ball_trail(
    frame: np.ndarray,
    trail_points: list[tuple[int, int] | None],
) -> np.ndarray:
    recent = [pt for pt in trail_points if pt is not None]
    if not recent:
        return frame
    for idx, pt in enumerate(recent):
        alpha = float(idx + 1) / float(max(1, len(recent)))
        color = (0, int(40 + (160 * alpha)), 255)
        radius = 3 if idx < len(recent) - 1 else 5
        cv2.circle(frame, pt, radius, color, thickness=-1)
        if idx > 0:
            prev_pt = recent[idx - 1]
            if float(np.linalg.norm(np.asarray(pt, dtype=np.float32) - np.asarray(prev_pt, dtype=np.float32))) <= 120.0:
                cv2.line(frame, prev_pt, pt, color, thickness=2)
    return frame


def _draw_ball_trail_v2(
    frame: np.ndarray,
    trail_points: list[tuple[tuple[int, int], float] | None],
) -> np.ndarray:
    recent = [item for item in trail_points if item is not None]
    if not recent:
        return frame

    overlay = frame.copy()
    total = max(1, len(recent))
    for idx, item in enumerate(recent):
        pt, confidence = item
        if idx < max(1, total // 3):
            color = (0, 255, 255)  # yellow = older
        elif idx < max(2, (2 * total) // 3):
            color = (0, 165, 255)  # orange = middle
        else:
            color = (0, 0, 255)  # red = newest

        alpha = max(0.12, min(0.92, float(confidence) ** 2))
        radius = 5 if idx < total - 1 else 7
        thickness = 2 if idx < total - 1 else 3

        cv2.circle(overlay, pt, radius, color, thickness=thickness, lineType=cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)
        overlay = frame.copy()
    return frame


def _in_tight_trail_zone(point_xy: tuple[int, int], table_xywh: tuple[int, int, int, int]) -> bool:
    x, y = point_xy
    tx, ty, tw, th = table_xywh
    zone_x1 = tx - int(round(tw * 0.08))
    zone_x2 = tx + tw + int(round(tw * 0.08))
    zone_y1 = ty - int(round(th * 0.10))
    zone_y2 = ty + th + int(round(th * 0.38))
    return zone_x1 <= x <= zone_x2 and zone_y1 <= y <= zone_y2


def _candidate_visual_ball_score(
    cropped_frame: np.ndarray,
    *,
    candidate_local: np.ndarray,
    search_x: int,
    search_y: int,
    table_xywh: tuple[int, int, int, int],
) -> tuple[tuple[int, int], float]:
    cx = int(round(search_x + float(candidate_local[0])))
    cy = int(round(search_y + float(candidate_local[1])))
    h, w = cropped_frame.shape[:2]
    px1 = max(0, cx - 4)
    py1 = max(0, cy - 4)
    px2 = min(w, cx + 5)
    py2 = min(h, cy + 5)
    patch = cropped_frame[py1:py2, px1:px2]
    if patch.size == 0:
        return (cx, cy), 0.0

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[..., 1].mean()) / 255.0
    mean_val = float(hsv[..., 2].mean()) / 255.0
    whiteness = max(0.0, 1.0 - (mean_sat * 1.55))
    brightness = mean_val

    tx, ty, tw, th = table_xywh
    zone_x1 = tx - int(round(tw * 0.20))
    zone_x2 = tx + tw + int(round(tw * 0.20))
    zone_y1 = ty - int(round(th * 0.28))
    zone_y2 = ty + th + int(round(th * 0.62))
    in_zone = zone_x1 <= cx <= zone_x2 and zone_y1 <= cy <= zone_y2
    proximity = 1.0 if in_zone else 0.05
    if cy < (ty - int(round(th * 0.45))):
        proximity *= 0.25
    if cy > (ty + th + int(round(th * 0.95))):
        proximity *= 0.55

    visual_score = (0.42 * whiteness) + (0.38 * brightness) + (0.20 * proximity)
    if mean_sat > 0.42 and mean_val < 0.86:
        visual_score *= 0.22
    return (cx, cy), float(visual_score)


def _build_augmented_v1_clip(
    *,
    source_clip: Path,
    augmented_clip: Path,
    roi: dict,
    margin_ratio: float,
    margin_y_ratio: float,
    trail_length: int,
    ball_profile: str,
) -> None:
    video_w, video_h = _video_dimensions(source_clip)
    crop_x, crop_y, crop_w, crop_h = _compute_roi_crop_bounds(
        video_w=video_w,
        video_h=video_h,
        roi=roi,
        margin_ratio=margin_ratio,
        margin_y_ratio=margin_y_ratio,
    )
    roi_x = int(roi.get("x", 0))
    roi_y = int(roi.get("y", 0))
    roi_w = int(roi.get("w", max(1, video_w // 4)))
    roi_h = int(roi.get("h", max(1, video_h // 8)))
    table_xywh = (roi_x - crop_x, roi_y - crop_y, roi_w, roi_h)

    cfg = _get_ball_tracking_profile(ball_profile)
    search_x = max(0, table_xywh[0] - int(round(table_xywh[2] * cfg.pad_x_ratio)))
    search_y = max(0, table_xywh[1] - int(round(table_xywh[3] * cfg.pad_top_ratio)))
    search_x2 = min(crop_w, table_xywh[0] + table_xywh[2] + int(round(table_xywh[2] * cfg.pad_x_ratio)))
    search_y2 = min(crop_h, table_xywh[1] + table_xywh[3] + int(round(table_xywh[3] * cfg.pad_bottom_ratio)))
    search_w = max(1, search_x2 - search_x)
    search_h = max(1, search_y2 - search_y)

    cap = cv2.VideoCapture(str(source_clip))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip for augmented overlay: {source_clip}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    augmented_clip.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(augmented_clip), fourcc, fps, (crop_w, crop_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer for augmented clip: {augmented_clip}")

    prev_gray: np.ndarray | None = None
    prev_center: np.ndarray | None = None
    prev_velocity = np.zeros(2, dtype=np.float32)
    missing_count = 0
    trail_history: deque[tuple[int, int] | None] = deque(maxlen=max(4, int(trail_length)))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cropped = frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w].copy()
            gray = cv2.cvtColor(cropped[search_y : search_y + search_h, search_x : search_x + search_w], cv2.COLOR_BGR2GRAY)
            chosen_xy: tuple[int, int] | None = None
            if prev_gray is not None:
                diff_gray = cv2.absdiff(gray, prev_gray)
                raw_candidates = _extract_ball_candidates(diff_gray)
                candidates: list[tuple[np.ndarray, float]] = []
                for cand_center, cand_score in raw_candidates:
                    _xy, visual_score = _candidate_visual_ball_score(
                        cropped,
                        candidate_local=cand_center,
                        search_x=search_x,
                        search_y=search_y,
                        table_xywh=table_xywh,
                    )
                    if visual_score < 0.34:
                        continue
                    combined_score = (0.60 * float(cand_score)) + (0.40 * visual_score)
                    candidates.append((cand_center, float(combined_score)))
                candidates.sort(key=lambda item: item[1], reverse=True)
                chosen_center: np.ndarray | None = None
                chosen_score = 0.0
                if candidates:
                    if prev_center is None:
                        chosen_center, chosen_score = _pick_best_candidate(
                            candidates,
                            min_score=cfg.min_start_score,
                        )
                    else:
                        predicted = prev_center + prev_velocity
                        max_jump = cfg.max_jump_px + (cfg.max_jump_missing_gain * float(missing_count))
                        chosen_center, chosen_score = _pick_best_candidate(
                            candidates,
                            min_score=cfg.min_continue_score,
                            predicted_center=predicted,
                            max_jump_px=max_jump,
                        )
                        if chosen_center is None and cfg.allow_top_fallback:
                            chosen_center, chosen_score = _pick_best_candidate(
                                candidates,
                                min_score=max(cfg.min_start_score, cfg.min_continue_score),
                            )
                if chosen_center is not None:
                    if prev_center is not None:
                        delta = chosen_center - prev_center
                        speed = float(np.linalg.norm(delta))
                        if speed >= cfg.min_continue_motion_px or chosen_score >= cfg.strong_score:
                            prev_velocity = (0.45 * prev_velocity) + (0.55 * delta)
                            prev_center = chosen_center
                            missing_count = 0
                            raw_xy = (
                                int(round(search_x + float(chosen_center[0]))),
                                int(round(search_y + float(chosen_center[1]))),
                            )
                            chosen_xy = raw_xy if _in_tight_trail_zone(raw_xy, table_xywh) else None
                        else:
                            missing_count += 1
                    else:
                        prev_center = chosen_center
                        prev_velocity = np.zeros(2, dtype=np.float32)
                        missing_count = 0
                        raw_xy = (
                            int(round(search_x + float(chosen_center[0]))),
                            int(round(search_y + float(chosen_center[1]))),
                        )
                        chosen_xy = raw_xy if _in_tight_trail_zone(raw_xy, table_xywh) else None
                else:
                    missing_count += 1
                    if missing_count > cfg.hold_misses:
                        prev_center = None
                        prev_velocity = np.zeros(2, dtype=np.float32)
            prev_gray = gray
            trail_history.append(chosen_xy)
            augmented = _draw_table_overlay(cropped, table_xywh)
            augmented = _draw_ball_trail(augmented, list(trail_history))
            writer.write(augmented)
    finally:
        writer.release()
        cap.release()


def _build_augmented_v2_clip(
    *,
    source_clip: Path,
    augmented_clip: Path,
    roi: dict,
    margin_ratio: float,
    margin_y_ratio: float,
    trail_length: int,
    ball_profile: str,
) -> None:
    video_w, video_h = _video_dimensions(source_clip)
    crop_x, crop_y, crop_w, crop_h = _compute_roi_crop_bounds(
        video_w=video_w,
        video_h=video_h,
        roi=roi,
        margin_ratio=margin_ratio,
        margin_y_ratio=margin_y_ratio,
    )
    roi_x = int(roi.get("x", 0))
    roi_y = int(roi.get("y", 0))
    roi_w = int(roi.get("w", max(1, video_w // 4)))
    roi_h = int(roi.get("h", max(1, video_h // 8)))
    table_xywh = (roi_x - crop_x, roi_y - crop_y, roi_w, roi_h)

    cfg = _get_ball_tracking_profile(ball_profile)
    search_x = max(0, table_xywh[0] - int(round(table_xywh[2] * cfg.pad_x_ratio)))
    search_y = max(0, table_xywh[1] - int(round(table_xywh[3] * cfg.pad_top_ratio)))
    search_x2 = min(crop_w, table_xywh[0] + table_xywh[2] + int(round(table_xywh[2] * cfg.pad_x_ratio)))
    search_y2 = min(crop_h, table_xywh[1] + table_xywh[3] + int(round(table_xywh[3] * cfg.pad_bottom_ratio)))
    search_w = max(1, search_x2 - search_x)
    search_h = max(1, search_y2 - search_y)

    cap = cv2.VideoCapture(str(source_clip))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip for augmented overlay: {source_clip}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    augmented_clip.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(augmented_clip), fourcc, fps, (crop_w, crop_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer for augmented clip: {augmented_clip}")

    prev_gray: np.ndarray | None = None
    prev_center: np.ndarray | None = None
    prev_velocity = np.zeros(2, dtype=np.float32)
    missing_count = 0
    trail_history: deque[tuple[tuple[int, int], float] | None] = deque(maxlen=max(6, int(trail_length)))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cropped = frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w].copy()
            gray = cv2.cvtColor(cropped[search_y : search_y + search_h, search_x : search_x + search_w], cv2.COLOR_BGR2GRAY)
            chosen_item: tuple[tuple[int, int], float] | None = None
            if prev_gray is not None:
                diff_gray = cv2.absdiff(gray, prev_gray)
                raw_candidates = _extract_ball_candidates(diff_gray)
                candidates: list[tuple[np.ndarray, float]] = []
                for cand_center, cand_score in raw_candidates:
                    _xy, visual_score = _candidate_visual_ball_score(
                        cropped,
                        candidate_local=cand_center,
                        search_x=search_x,
                        search_y=search_y,
                        table_xywh=table_xywh,
                    )
                    if visual_score < 0.22:
                        continue
                    combined_score = (0.58 * float(cand_score)) + (0.42 * visual_score)
                    candidates.append((cand_center, float(combined_score)))
                candidates.sort(key=lambda item: item[1], reverse=True)
                chosen_center: np.ndarray | None = None
                chosen_score = 0.0
                if candidates:
                    if prev_center is None:
                        chosen_center, chosen_score = _pick_best_candidate(
                            candidates,
                            min_score=max(0.0, cfg.min_start_score * 0.75),
                        )
                    else:
                        predicted = prev_center + prev_velocity
                        max_jump = cfg.max_jump_px + (cfg.max_jump_missing_gain * float(missing_count))
                        chosen_center, chosen_score = _pick_best_candidate(
                            candidates,
                            min_score=max(0.0, cfg.min_continue_score * 0.75),
                            predicted_center=predicted,
                            max_jump_px=max_jump,
                        )
                        if chosen_center is None and cfg.allow_top_fallback:
                            chosen_center, chosen_score = _pick_best_candidate(
                                candidates,
                                min_score=max(0.0, cfg.min_start_score * 0.75),
                            )
                if chosen_center is not None:
                    if prev_center is not None:
                        delta = chosen_center - prev_center
                        speed = float(np.linalg.norm(delta))
                        if speed >= cfg.min_continue_motion_px or chosen_score >= cfg.strong_score:
                            prev_velocity = (0.45 * prev_velocity) + (0.55 * delta)
                            prev_center = chosen_center
                            missing_count = 0
                            raw_xy = (
                                int(round(search_x + float(chosen_center[0]))),
                                int(round(search_y + float(chosen_center[1]))),
                            )
                            if _in_tight_trail_zone(raw_xy, table_xywh):
                                chosen_item = (raw_xy, float(max(0.0, min(1.0, chosen_score))))
                        else:
                            missing_count += 1
                    else:
                        prev_center = chosen_center
                        prev_velocity = np.zeros(2, dtype=np.float32)
                        missing_count = 0
                        raw_xy = (
                            int(round(search_x + float(chosen_center[0]))),
                            int(round(search_y + float(chosen_center[1]))),
                        )
                        if _in_tight_trail_zone(raw_xy, table_xywh):
                            chosen_item = (raw_xy, float(max(0.0, min(1.0, chosen_score))))
                else:
                    missing_count += 1
                    if missing_count > cfg.hold_misses:
                        prev_center = None
                        prev_velocity = np.zeros(2, dtype=np.float32)
            prev_gray = gray
            trail_history.append(chosen_item)
            augmented = _draw_table_overlay(cropped, table_xywh)
            augmented = _draw_ball_trail_v2(augmented, list(trail_history))
            writer.write(augmented)
    finally:
        writer.release()
        cap.release()


def _build_flipped_clip(
    *,
    source_clip: Path,
    flipped_clip: Path,
) -> None:
    flipped_clip.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_clip),
        "-filter:v",
        "hflip",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        str(flipped_clip),
    ]
    subprocess.run(cmd, check=True)


def _rename_clip_with_winner(
    clip_path: Path,
    *,
    point_id: str,
    winner: str,
    model_slug: str,
    clip_label: str | None = None,
) -> Path:
    side = "near" if winner == "player_a" else "far" if winner == "player_b" else "unknown"
    if clip_label:
        final_name = f"{point_id}__{clip_label}__pick_{side}__native_video_{model_slug}.mp4"
    else:
        final_name = f"{point_id}__pick_{side}__native_video_{model_slug}.mp4"
    final_path = clip_path.with_name(final_name)
    if final_path.exists():
        final_path.unlink()
    clip_path.rename(final_path)
    return final_path


def _update_point(
    point: RallyTimelinePoint,
    *,
    winner: str,
    raw_text: str,
    model_name: str,
    model_slug: str,
    clip_path: Path,
    score_a: float,
    score_b: float,
    decision: str | None = None,
    confidence: float | None = None,
    end_category: str | None = None,
    loser_candidate: str | None = None,
    last_hitter_candidate: str | None = None,
    preserve_model_labels: bool = False,
    prompt_family_flag: str = "winner_pairwise_yes_no",
    extra_flags: Iterable[str] = (),
    extra_change_fields: dict[str, object] | None = None,
) -> None:
    normalized_loser = loser_candidate or ("unknown" if preserve_model_labels else _opposite_player(winner))
    if normalized_loser not in {"player_a", "player_b"}:
        normalized_loser = "unknown"
    normalized_last_hitter = last_hitter_candidate if last_hitter_candidate in {"player_a", "player_b"} else "unknown"
    normalized_end_category = end_category or "ambiguous_review"
    point.winner_candidate = winner  # type: ignore[assignment]
    point.winner_confidence = float(
        confidence if confidence is not None else (max(score_a, score_b) if winner in {"player_a", "player_b"} else 0.0)
    )
    point.winner_decision = decision if decision is not None else ("review" if winner in {"player_a", "player_b"} else "blocked")
    point.winner_reason = raw_text[:160].strip() or None
    point.winner_model = model_name
    point.winner_score_a = float(score_a)
    point.winner_score_b = float(score_b)
    point.winner_end_category = normalized_end_category
    point.winner_loser_candidate = normalized_loser  # type: ignore[assignment]
    point.winner_last_hitter_candidate = normalized_last_hitter  # type: ignore[assignment]
    point.winner = "unknown"
    point.source = "ai"
    point.flags = sorted(
        set(
            point.flags
            + [
                "winner_native_video",
                f"winner_model_{model_slug}_transformers",
                prompt_family_flag,
                "winner_dense_video_config",
            ]
            + list(extra_flags)
        )
    )
    change_fields = {
        "winner_candidate": winner,
        "winner_confidence": point.winner_confidence,
        "winner_decision": point.winner_decision,
        "winner_reason": point.winner_reason,
        "winner_model": model_name,
        "winner_score_a": point.winner_score_a,
        "winner_score_b": point.winner_score_b,
        "winner_end_category": point.winner_end_category,
        "winner_loser_candidate": point.winner_loser_candidate,
        "winner_last_hitter_candidate": point.winner_last_hitter_candidate,
        "winner_clip": str(clip_path),
    }
    if extra_change_fields:
        change_fields.update(extra_change_fields)
    point.corrections.append(
        Correction(
            at="",
            by="local_vlm_native_video",
            changes={
                "winner_native_video": change_fields
            },
            note=f"native-video {model_name} winner inference",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Refine rally winners using native-video Transformers.")
    parser.add_argument("--timeline", required=True, help="Path to input rally timeline JSON")
    parser.add_argument("--out", required=True, help="Path to output JSON")
    parser.add_argument("--model-dir", default="models/Qwen3-VL-4B-Instruct", help="Local HF model directory")
    parser.add_argument("--clip-dir", required=True, help="Directory for exported review clips")
    parser.add_argument(
        "--window-ratio",
        type=float,
        default=1.0,
        help="Deprecated; winner inference now always uses the full frozen rally clip",
    )
    parser.add_argument(
        "--full-rally-threshold-sec",
        type=float,
        default=0.0,
        help="Deprecated; winner inference now always uses the full frozen rally clip",
    )
    parser.add_argument(
        "--min-window-sec",
        type=float,
        default=0.0,
        help="Deprecated; winner inference now always uses the full frozen rally clip",
    )
    parser.add_argument(
        "--max-window-sec",
        type=float,
        default=0.0,
        help="Deprecated; winner inference now always uses the full frozen rally clip",
    )
    parser.add_argument("--fps-sample", type=float, default=4.0, help="Native-video sampling fps")
    parser.add_argument("--min-frames", type=int, default=12, help="Minimum sampled frames")
    parser.add_argument("--max-frames", type=int, default=16, help="Maximum sampled frames")
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=(1280 * 720),
        help="Maximum visual pixel budget for the sampled video tokens; use 0 to disable",
    )
    parser.add_argument("--size-shortest-edge", type=int, default=576, help="Video processor shortest edge")
    parser.add_argument("--size-longest-edge", type=int, default=1048576, help="Video processor longest edge")
    parser.add_argument("--point-ids", nargs="*", default=[], help="Optional point ids to process")
    parser.add_argument("--max-points", type=int, default=None, help="Optional cap on processed rallies")
    parser.add_argument(
        "--main-pass-view",
        choices=["full", "roi", "table_only"],
        default="roi",
        help="Video view used for the main A/B prompt pass",
    )
    parser.add_argument(
        "--roi-margin-ratio",
        type=float,
        default=0.4,
        help="Extra margin around table ROI when main-pass-view=roi",
    )
    parser.add_argument(
        "--roi-margin-y-ratio",
        type=float,
        default=0.9,
        help="Optional extra vertical margin ratio around table ROI; negative means reuse roi-margin-ratio",
    )
    parser.add_argument(
        "--table-only-x-margin-ratio",
        type=float,
        default=0.2,
        help="Extra horizontal margin around the table when main-pass-view=table_only",
    )
    parser.add_argument(
        "--table-only-top-margin-ratio",
        type=float,
        default=0.2,
        help="Extra vertical space above the table when main-pass-view=table_only",
    )
    parser.add_argument(
        "--table-only-bottom-margin-ratio",
        type=float,
        default=0.0,
        help="Extra vertical space below the table when main-pass-view=table_only",
    )
    parser.add_argument(
        "--flip-main-pass",
        action="store_true",
        help="Horizontally flip the main-pass clip before asking A/B winner prompts",
    )
    parser.add_argument(
        "--main-pass-overlay",
        choices=["none", "augmented_v1", "augmented_v2"],
        default="none",
        help="Optional overlay mode applied to the main-pass clip before inference",
    )
    parser.add_argument(
        "--winner-mode",
        choices=["single", "dual4b_raw_augv1"],
        default="dual4b_raw_augv1",
        help="Winner orchestration mode; dual4b runs raw and augmented_v1 in parallel and records agreement/disagreement",
    )
    parser.add_argument(
        "--ensemble-primary-branch",
        choices=["raw", "augv1"],
        default="augv1",
        help="Primary branch used when the dual4b branches disagree or tie",
    )
    parser.add_argument(
        "--aug-ball-profile",
        choices=["support", "standalone"],
        default="support",
        help="Ball-tracking profile used for augmented_v1 ball trail overlay",
    )
    parser.add_argument(
        "--aug-ball-trail-length",
        type=int,
        default=18,
        help="Maximum recent trail points rendered for augmented_v1",
    )
    parser.add_argument(
        "--aug-prompt-mode",
        choices=["pairwise_overlay_yesno", "var_overlay_json"],
        default="pairwise_overlay_yesno",
        help="Prompt mode used for augmented_v1 branch; var_overlay_json is a focused test mode",
    )
    parser.add_argument(
        "--aug-legal-return-verifier",
        action="store_true",
        help="For augmented_v1, verify whether the predicted side's final touch was legal or out before finalizing winner",
    )
    parser.add_argument(
        "--winner-prompt-family",
        choices=[
            "pairwise_yesno",
            "category_schema_direct",
            "category_schema_rules_v2",
            "category_schema_anchor4",
            "category_schema_taxonomy_first_anchor4",
            "category_schema_taxonomy_first_anchor4_fewshot",
            "category_schema_taxonomy_first_topbottom_anchor4",
            "category_schema_touchprobe_topbottom_anchor4",
        ],
        default="pairwise_yesno",
        help="Prompt family for winner inference; category families add loser/last_hitter/end_category outputs",
    )
    parser.add_argument(
        "--winner-fewshot-path",
        default="dataset/reviewed_matches/match_vinh_001/set_04/fewshot_seed.jsonl",
        help="Optional JSONL file with reviewed taxonomy examples for few-shot prompt families",
    )
    parser.add_argument(
        "--winner-fewshot-max-examples",
        type=int,
        default=8,
        help="Maximum few-shot examples inserted into taxonomy few-shot prompts; 0 means use all",
    )
    args = parser.parse_args()

    timeline = load_rally_timeline(Path(args.timeline))
    boundary_signature_before = _frozen_boundary_signature(timeline.points)
    output_path = Path(args.out)
    clip_dir = Path(args.clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(args.model_dir).name
    model_slug = _slugify_model_name(model_name)

    print(f"--- Native-video Winner Refinement ({args.model_dir}) ---")
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model loaded on {model.device}")

    fewshot_rows: list[dict[str, object]] = []
    fewshot_path = Path(str(args.winner_fewshot_path))
    if str(args.winner_prompt_family) == "category_schema_taxonomy_first_anchor4_fewshot":
        if not fewshot_path.exists():
            raise FileNotFoundError(f"Few-shot seed file not found: {fewshot_path}")
        fewshot_rows = _load_taxonomy_fewshot_rows(fewshot_path)

    selected_ids = _selected_point_ids(args.point_ids)
    processed = 0
    csv_rows: list[dict[str, object]] = []
    prompt_family_flag_value = f"winner_{str(args.winner_prompt_family)}"
    preserve_model_taxonomy_labels = str(args.winner_prompt_family) in {
        "category_schema_taxonomy_first_anchor4",
        "category_schema_taxonomy_first_anchor4_fewshot",
    }

    def ask_text_for_video(video_path: Path, prompt_text: str, *, max_new_tokens: int = 32) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path.resolve())},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processor_kwargs = {
            "text": [text],
            "videos": [str(video_path.resolve())],
            "return_tensors": "pt",
            "fps": float(args.fps_sample),
            "min_frames": int(args.min_frames),
            "max_frames": int(args.max_frames),
            "size": {
                "shortest_edge": int(args.size_shortest_edge),
                "longest_edge": int(args.size_longest_edge),
            },
        }
        if int(args.max_pixels) > 0:
            processor_kwargs["max_pixels"] = int(args.max_pixels)
        inputs = processor(**processor_kwargs)
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        generated_ids = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return str(output_text[0]).strip() if output_text else ""

    def run_branch(
        *,
        point: RallyTimelinePoint,
        temp_clip: Path,
        clip_scope: str,
        branch_overlay: str,
        branch_label: str | None,
    ) -> dict[str, object]:
        main_pass_clip = temp_clip
        roi_clip: Path | None = None
        table_only_clip: Path | None = None
        flip_clip: Path | None = None
        augmented_clip: Path | None = None
        composite_clip: Path | None = None

        if str(args.main_pass_view) == "roi":
            roi_suffix = branch_label or "single"
            roi_clip = clip_dir / f"{point.id}__native_roi_{roi_suffix}.mp4"
            _build_roi_clip(
                source_clip=temp_clip,
                roi_clip=roi_clip,
                roi=timeline.roi,
                margin_ratio=float(args.roi_margin_ratio),
                margin_y_ratio=(float(args.roi_margin_ratio) if float(args.roi_margin_y_ratio) < 0 else float(args.roi_margin_y_ratio)),
            )
            main_pass_clip = roi_clip
        elif str(args.main_pass_view) == "table_only":
            table_suffix = branch_label or "single"
            table_only_clip = clip_dir / f"{point.id}__native_tableonly_{table_suffix}.mp4"
            _build_table_only_clip(
                source_clip=temp_clip,
                table_only_clip=table_only_clip,
                roi=timeline.roi,
                x_margin_ratio=float(args.table_only_x_margin_ratio),
                top_margin_ratio=float(args.table_only_top_margin_ratio),
                bottom_margin_ratio=float(args.table_only_bottom_margin_ratio),
            )
            main_pass_clip = table_only_clip

        if branch_overlay == "augmented_v1":
            aug_suffix = branch_label or "single"
            augmented_clip = clip_dir / f"{point.id}__native_augv1_{aug_suffix}.mp4"
            _build_augmented_v1_clip(
                source_clip=temp_clip,
                augmented_clip=augmented_clip,
                roi=timeline.roi,
                margin_ratio=float(args.roi_margin_ratio),
                margin_y_ratio=(float(args.roi_margin_ratio) if float(args.roi_margin_y_ratio) < 0 else float(args.roi_margin_y_ratio)),
                trail_length=int(args.aug_ball_trail_length),
                ball_profile=str(args.aug_ball_profile),
            )
            main_pass_clip = augmented_clip
        elif branch_overlay == "augmented_v2":
            aug_suffix = branch_label or "single"
            augmented_clip = clip_dir / f"{point.id}__native_augv2_{aug_suffix}.mp4"
            _build_augmented_v2_clip(
                source_clip=temp_clip,
                augmented_clip=augmented_clip,
                roi=timeline.roi,
                margin_ratio=float(args.roi_margin_ratio),
                margin_y_ratio=(float(args.roi_margin_ratio) if float(args.roi_margin_y_ratio) < 0 else float(args.roi_margin_y_ratio)),
                trail_length=int(args.aug_ball_trail_length),
                ball_profile=str(args.aug_ball_profile),
            )
            main_pass_clip = augmented_clip

        if bool(args.flip_main_pass):
            flip_suffix = branch_label or "single"
            flip_clip = clip_dir / f"{point.id}__native_mainpass_flip_{flip_suffix}.mp4"
            _build_flipped_clip(
                source_clip=main_pass_clip,
                flipped_clip=flip_clip,
            )
            main_pass_clip = flip_clip

        raw_text_tiebreak = ""
        raw_text_verifier = ""
        parsed_end_category = "ambiguous_review"
        parsed_loser = "unknown"
        parsed_last_hitter = "unknown"
        parsed_confidence: float | None = None
        loser_touched_final_ball = "unknown"
        final_return_crossed_net = "unknown"
        final_return_landed_in = "unknown"
        consistency_flags: list[str] = []
        category_prompt_families = {
            "category_schema_direct",
            "category_schema_rules_v2",
            "category_schema_anchor4",
            "category_schema_taxonomy_first_anchor4",
            "category_schema_taxonomy_first_anchor4_fewshot",
            "category_schema_taxonomy_first_topbottom_anchor4",
            "category_schema_touchprobe_topbottom_anchor4",
        }
        if str(args.winner_prompt_family) in category_prompt_families:
            raw_text_a = ask_text_for_video(
                main_pass_clip,
                _build_category_prompt_for_family(
                    prompt_family_name=str(args.winner_prompt_family),
                    clip_scope=clip_scope,
                    branch_overlay=branch_overlay,
                    point_id=str(point.id),
                    fewshot_rows=fewshot_rows,
                    fewshot_max_examples=int(args.winner_fewshot_max_examples),
                ),
                max_new_tokens=64,
            )
            raw_text_b = ""
            parsed_end_category = _extract_end_category(raw_text_a)
            parsed_confidence = _extract_confidence_value(raw_text_a)
            loser_touched_final_ball = _extract_labeled_choice(raw_text_a, "loser_touched_final_ball", {"yes", "no", "unknown"})
            final_return_crossed_net = _extract_labeled_choice(raw_text_a, "final_return_crossed_net", {"yes", "no", "unknown"})
            final_return_landed_in = _extract_labeled_choice(raw_text_a, "final_return_landed_in", {"yes", "no", "unknown"})
            prompt_family_name = str(args.winner_prompt_family)
            if prompt_family_name in {"category_schema_taxonomy_first_topbottom_anchor4", "category_schema_touchprobe_topbottom_anchor4"}:
                parsed_loser = _map_vertical_position_to_player(
                    _extract_labeled_choice(raw_text_a, "loser_position", {"top", "bottom", "unknown"})
                )
                parsed_last_hitter = _map_vertical_position_to_player(
                    _extract_labeled_choice(
                        raw_text_a,
                        "last_hitter_position",
                        {"top", "bottom", "unknown"},
                    )
                )
                if parsed_last_hitter == "unknown":
                    parsed_last_hitter = _map_vertical_position_to_player(
                        _extract_labeled_choice(raw_text_a, "final_touch_position", {"top", "bottom", "unknown"})
                    )
                winner = _map_vertical_position_to_player(
                    _extract_labeled_choice(raw_text_a, "winner_position", {"top", "bottom", "unknown"})
                )
            else:
                parsed_loser = _extract_labeled_choice(raw_text_a, "loser", {"player_a", "player_b", "unknown"})
                parsed_last_hitter = _extract_labeled_choice(raw_text_a, "last_hitter", {"player_a", "player_b", "unknown"})
                winner = _extract_winner_label(raw_text_a)
            if prompt_family_name not in {"category_schema_taxonomy_first_anchor4", "category_schema_taxonomy_first_anchor4_fewshot", "category_schema_taxonomy_first_topbottom_anchor4", "category_schema_touchprobe_topbottom_anchor4"} and winner not in {"player_a", "player_b"}:
                winner = _opposite_player(parsed_loser)
            if prompt_family_name not in {"category_schema_taxonomy_first_anchor4", "category_schema_taxonomy_first_anchor4_fewshot", "category_schema_taxonomy_first_topbottom_anchor4", "category_schema_touchprobe_topbottom_anchor4"} and parsed_loser == "unknown" and winner in {"player_a", "player_b"}:
                parsed_loser = _opposite_player(winner)
            if prompt_family_name not in {"category_schema_taxonomy_first_anchor4", "category_schema_taxonomy_first_anchor4_fewshot", "category_schema_taxonomy_first_topbottom_anchor4", "category_schema_touchprobe_topbottom_anchor4"}:
                winner, parsed_loser, parsed_last_hitter = _normalize_category_decision(
                    winner=winner,
                    loser=parsed_loser,
                    last_hitter=parsed_last_hitter,
                    category=parsed_end_category,
                )
            consistency_flags = _category_consistency_flags(
                winner=winner,
                loser=parsed_loser,
                last_hitter=parsed_last_hitter,
                category=parsed_end_category,
                loser_touched_final_ball=loser_touched_final_ball,
                final_return_crossed_net=final_return_crossed_net,
                final_return_landed_in=final_return_landed_in,
            )
            score_a = 1.0 if winner == "player_a" else 0.0
            score_b = 1.0 if winner == "player_b" else 0.0
        elif branch_overlay in {"augmented_v1", "augmented_v2"} and str(args.aug_prompt_mode) == "var_overlay_json":
            raw_text_a = ask_text_for_video(main_pass_clip, _augmented_var_prompt(clip_scope))
            raw_text_b = ""
            winner = _extract_comparative_winner(raw_text_a)
            if winner in {"player_a", "player_b"}:
                score_a = 1.0 if winner == "player_a" else 0.0
                score_b = 1.0 if winner == "player_b" else 0.0
            else:
                score_a = 0.0
                score_b = 0.0
                composite_suffix = branch_label or "single"
                composite_clip = clip_dir / f"{point.id}__native_composite_{composite_suffix}.mp4"
                _build_composite_clip(
                    source_clip=temp_clip,
                    composite_clip=composite_clip,
                    roi=timeline.roi,
                )
                raw_text_tiebreak = ask_text_for_video(
                    main_pass_clip,
                    (
                        "This rally video may include visual overlays. "
                        "If present: the green box marks the table and the red trail marks recent ball motion. "
                        "Player A is the near-side player. Player B is the far-side player. "
                        "Decide the winner primarily from the trail ending and table box, then use player reaction only as secondary evidence. "
                        "Do not prefer the near-side player by default. "
                        "Answer with one short sentence: Winner=player_a or Winner=player_b, then a brief reason."
                    ),
                )
                tiebreak_winner = _extract_winner_label(raw_text_tiebreak)
                if tiebreak_winner in {"player_a", "player_b"}:
                    winner = tiebreak_winner
                    score_a = 1.0 if winner == "player_a" else 0.0
                    score_b = 1.0 if winner == "player_b" else 0.0
                else:
                    winner = "unknown"
        else:
            raw_text_a = ask_text_for_video(
                main_pass_clip,
                (
                    (
                        _overlay_prompt_prefix(branch_overlay, clip_scope)
                        + "Did Player A (near side) win? "
                        + "Answer with Yes or No, then one short reason based on the last successful shot and failed return."
                    )
                    if branch_overlay in {"augmented_v1", "augmented_v2"}
                    else (
                        f"Did Player A (near side) win this {'full table-tennis rally' if clip_scope == 'full' else 'table-tennis rally segment'}? "
                        "Answer with Yes or No, then one short reason based on the last successful shot and failed return."
                    )
                ),
            )
            raw_text_b = ask_text_for_video(
                main_pass_clip,
                (
                    (
                        _overlay_prompt_prefix(branch_overlay, clip_scope)
                        + "Did Player B (far side) win? "
                        + "Answer with Yes or No, then one short reason based on the last successful shot and failed return."
                    )
                    if branch_overlay in {"augmented_v1", "augmented_v2"}
                    else (
                        f"Did Player B (far side) win this {'full table-tennis rally' if clip_scope == 'full' else 'table-tennis rally segment'}? "
                        "Answer with Yes or No, then one short reason based on the last successful shot and failed return."
                    )
                ),
            )

            answer_a = _extract_yes_no(raw_text_a)
            answer_b = _extract_yes_no(raw_text_b)
            score_a = 1.0 if answer_a == "yes" else 0.0
            score_b = 1.0 if answer_b == "yes" else 0.0
            if answer_a == "yes" and answer_b != "yes":
                winner = "player_a"
            elif answer_b == "yes" and answer_a != "yes":
                winner = "player_b"
            else:
                composite_suffix = branch_label or "single"
                composite_clip = clip_dir / f"{point.id}__native_composite_{composite_suffix}.mp4"
                _build_composite_clip(
                    source_clip=temp_clip,
                    composite_clip=composite_clip,
                    roi=timeline.roi,
                )
                raw_text_tiebreak = ask_text_for_video(
                    (main_pass_clip if branch_overlay in {"augmented_v1", "augmented_v2"} else composite_clip),
                    (
                        (
                            _overlay_prompt_prefix(branch_overlay, clip_scope)
                            + "Player A is the near-side player. Player B is the far-side player. "
                            + "Decide the winner primarily from the overlay evidence near the end of the rally, then use player reaction only as secondary evidence. "
                            + "Do not prefer the near-side player by default. "
                            + "Answer with one short sentence: Winner=player_a or Winner=player_b, then a brief reason."
                        )
                        if branch_overlay in {"augmented_v1", "augmented_v2"}
                        else (
                            f"This video shows one {'full table-tennis rally' if clip_scope == 'full' else 'table-tennis rally segment'}. "
                            "The LEFT half is the original full frame. The RIGHT half is a zoom around the table and players. "
                            "Player A is the near-side player. Player B is the far-side player. "
                            "Decide the winner from the final successful shot and the failed return. "
                            "Do not prefer the near-side player by default. "
                            "Answer with one short sentence: Winner=player_a or Winner=player_b, then a brief reason."
                        )
                    ),
                )
                tiebreak_winner = _extract_winner_label(raw_text_tiebreak)
                if tiebreak_winner in {"player_a", "player_b"}:
                    winner = tiebreak_winner
                    score_a = 1.0 if winner == "player_a" else 0.0
                    score_b = 1.0 if winner == "player_b" else 0.0
                else:
                    winner = "unknown"
                    score_a = 0.0
                    score_b = 0.0

        if branch_overlay in {"augmented_v1", "augmented_v2"} and bool(args.aug_legal_return_verifier) and winner in {"player_a", "player_b"}:
            predicted_side = "Player A (near side)" if winner == "player_a" else "Player B (far side)"
            opponent_side = "Player B (far side)" if winner == "player_a" else "Player A (near side)"
            raw_text_verifier = ask_text_for_video(
                main_pass_clip,
                (
                    "This rally video may include visual overlays. "
                    "If present: the green box marks the table and the red trail marks recent ball motion. "
                    f"Focus only on the final touch by {predicted_side} near the end of the rally. "
                    f"If that final touch lands legally on {opponent_side}'s side of the table, answer Outcome=legal. "
                    f"If {predicted_side}'s final touch sends the ball out, misses the table, or otherwise fails to land legally, answer Outcome=out. "
                    "If you cannot tell, answer Outcome=unclear. "
                    "Answer with one short sentence: Outcome=legal or Outcome=out or Outcome=unclear, then a brief reason."
                ),
            )
            verifier_outcome = _extract_outcome_label(raw_text_verifier)
            if verifier_outcome == "out":
                winner = "player_b" if winner == "player_a" else "player_a"
                score_a = 1.0 if winner == "player_a" else 0.0
                score_b = 1.0 if winner == "player_b" else 0.0

        raw_text = f"A? {raw_text_a} || B? {raw_text_b}"
        if raw_text_tiebreak:
            raw_text += f" || T? {raw_text_tiebreak}"
        if raw_text_verifier:
            raw_text += f" || V? {raw_text_verifier}"

        export_source = (
            main_pass_clip
            if branch_overlay in {"augmented_v1", "augmented_v2"} or str(args.main_pass_view) != "full" or bool(args.flip_main_pass)
            else temp_clip
        )
        export_copy = clip_dir / f"{point.id}__{(branch_label or 'single')}__export.mp4"
        shutil.copyfile(export_source, export_copy)
        final_clip = _rename_clip_with_winner(
            export_copy,
            point_id=point.id,
            winner=winner,
            model_slug=model_slug,
            clip_label=branch_label,
        )

        for clip in [roi_clip, table_only_clip, augmented_clip, flip_clip, composite_clip]:
            if clip is not None and clip.exists():
                clip.unlink()

        return {
            "branch_label": branch_label or "single",
            "overlay": branch_overlay,
            "winner": winner,
            "score_a": float(score_a),
            "score_b": float(score_b),
            "end_category": parsed_end_category,
            "loser": parsed_loser,
            "last_hitter": parsed_last_hitter,
            "confidence": parsed_confidence,
            "consistency_flags": consistency_flags,
            "loser_touched_final_ball": loser_touched_final_ball,
            "final_return_crossed_net": final_return_crossed_net,
            "final_return_landed_in": final_return_landed_in,
            "raw_text": raw_text,
            "raw_text_a": raw_text_a,
            "raw_text_b": raw_text_b,
            "raw_text_tiebreak": raw_text_tiebreak,
            "clip_path": final_clip,
        }

    for point in timeline.points:
        if selected_ids and point.id not in selected_ids:
            continue
        if args.max_points is not None and processed >= max(0, int(args.max_points)):
            break

        window_start, window_end = _winner_window(
            point,
            ratio=float(args.window_ratio),
            full_rally_threshold_sec=float(args.full_rally_threshold_sec),
            min_window_sec=float(args.min_window_sec),
            max_window_sec=float(args.max_window_sec),
        )
        temp_clip = clip_dir / f"{point.id}__native_window.mp4"
        _clip_window_video(
            source_video=timeline.video_path,
            clip_path=temp_clip,
            start_sec=window_start,
            end_sec=window_end,
        )
        clip_scope = "full" if window_start <= float(point.t_start) + 1e-6 else "partial"
        branch_rows: dict[str, dict[str, object]] = {}
        if str(args.winner_mode) == "dual4b_raw_augv1":
            branch_rows["raw"] = run_branch(
                point=point,
                temp_clip=temp_clip,
                clip_scope=clip_scope,
                branch_overlay="none",
                branch_label="raw",
            )
            branch_rows["augv1"] = run_branch(
                point=point,
                temp_clip=temp_clip,
                clip_scope=clip_scope,
                branch_overlay="augmented_v1",
                branch_label="augv1",
            )

            primary_branch = str(args.ensemble_primary_branch)
            raw_winner = str(branch_rows["raw"]["winner"])
            aug_winner = str(branch_rows["augv1"]["winner"])
            vote_a = 0.0
            vote_b = 0.0
            for branch in branch_rows.values():
                if branch["winner"] == "player_a":
                    vote_a += 0.5
                elif branch["winner"] == "player_b":
                    vote_b += 0.5

            if vote_a > vote_b:
                winner = "player_a"
            elif vote_b > vote_a:
                winner = "player_b"
            else:
                primary_candidate = str(branch_rows[primary_branch]["winner"])
                winner = primary_candidate if primary_candidate in {"player_a", "player_b"} else "unknown"

            if raw_winner == aug_winner and raw_winner in {"player_a", "player_b"}:
                ensemble_status = "agree"
            elif raw_winner in {"player_a", "player_b"} and aug_winner in {"player_a", "player_b"} and raw_winner != aug_winner:
                ensemble_status = "disagree"
            elif winner in {"player_a", "player_b"}:
                ensemble_status = "partial"
            else:
                ensemble_status = "unknown"

            chosen_branch_label = primary_branch
            if ensemble_status == "agree":
                chosen_branch_label = "raw" if raw_winner == winner else primary_branch
            chosen_branch = branch_rows[chosen_branch_label]
            raw_text = (
                f"dual4b_{ensemble_status} raw={raw_winner} augv1={aug_winner} "
                f"primary={primary_branch}"
            )
            ensemble_model_name = f"{model_name}+dual4b"
            _update_point(
                point,
                winner=winner,
                raw_text=raw_text,
                model_name=ensemble_model_name,
                model_slug=model_slug,
                clip_path=Path(str(chosen_branch["clip_path"])),
                score_a=vote_a,
                score_b=vote_b,
                confidence=max(vote_a, vote_b) if winner in {"player_a", "player_b"} else 0.0,
                end_category=str(chosen_branch["end_category"]),
                loser_candidate=str(chosen_branch["loser"]),
                last_hitter_candidate=str(chosen_branch["last_hitter"]),
                preserve_model_labels=preserve_model_taxonomy_labels,
                prompt_family_flag=prompt_family_flag_value,
                decision="review" if winner in {"player_a", "player_b"} else "blocked",
                extra_flags=[
                    "winner_dual4b",
                    f"winner_dual4b_{ensemble_status}",
                    f"winner_dual4b_primary_{primary_branch}",
                ] + list(chosen_branch["consistency_flags"]),
                extra_change_fields={
                    "winner_branch_raw": raw_winner,
                    "winner_branch_augv1": aug_winner,
                    "winner_ensemble_status": ensemble_status,
                    "winner_ensemble_primary_branch": primary_branch,
                    "winner_clip_raw": str(branch_rows["raw"]["clip_path"]),
                    "winner_clip_augv1": str(branch_rows["augv1"]["clip_path"]),
                    "winner_category_consistency_flags": list(chosen_branch["consistency_flags"]),
                    "winner_loser_touched_final_ball": str(chosen_branch["loser_touched_final_ball"]),
                    "winner_final_return_crossed_net": str(chosen_branch["final_return_crossed_net"]),
                    "winner_final_return_landed_in": str(chosen_branch["final_return_landed_in"]),
                },
            )
            csv_rows.append(
                {
                    "id": point.id,
                    "t_start": float(point.t_start),
                    "t_end": float(point.t_end),
                    "clip_start": window_start,
                    "clip_end": window_end,
                    "winner_candidate": winner,
                    "winner_decision": point.winner_decision,
                    "winner_confidence": point.winner_confidence,
                    "winner_score_a": point.winner_score_a,
                    "winner_score_b": point.winner_score_b,
                    "winner_end_category": point.winner_end_category,
                    "winner_loser_candidate": point.winner_loser_candidate,
                    "winner_last_hitter_candidate": point.winner_last_hitter_candidate,
                    "raw_output": raw_text,
                    "raw_output_a": "",
                    "raw_output_b": "",
                    "raw_output_tiebreak": "",
                    "file": Path(str(chosen_branch["clip_path"])).name,
                    "ensemble_status": ensemble_status,
                    "primary_branch": primary_branch,
                    "branch_raw_winner_candidate": raw_winner,
                    "branch_raw_output_a": str(branch_rows["raw"]["raw_text_a"]),
                    "branch_raw_output_b": str(branch_rows["raw"]["raw_text_b"]),
                    "branch_raw_output_tiebreak": str(branch_rows["raw"]["raw_text_tiebreak"]),
                    "branch_raw_file": Path(str(branch_rows["raw"]["clip_path"])).name,
                    "branch_augv1_winner_candidate": aug_winner,
                    "branch_augv1_output_a": str(branch_rows["augv1"]["raw_text_a"]),
                    "branch_augv1_output_b": str(branch_rows["augv1"]["raw_text_b"]),
                    "branch_augv1_output_tiebreak": str(branch_rows["augv1"]["raw_text_tiebreak"]),
                    "branch_augv1_file": Path(str(branch_rows["augv1"]["clip_path"])).name,
                }
            )
        else:
            branch_overlay = str(args.main_pass_overlay)
            single_branch = run_branch(
                point=point,
                temp_clip=temp_clip,
                clip_scope=clip_scope,
                branch_overlay=branch_overlay,
                branch_label=None,
            )
            winner = str(single_branch["winner"])
            raw_text = str(single_branch["raw_text"])
            _update_point(
                point,
                winner=winner,
                raw_text=raw_text,
                model_name=model_name,
                model_slug=model_slug,
                clip_path=Path(str(single_branch["clip_path"])),
                score_a=float(single_branch["score_a"]),
                score_b=float(single_branch["score_b"]),
                confidence=(None if single_branch["confidence"] is None else float(single_branch["confidence"])),
                end_category=str(single_branch["end_category"]),
                loser_candidate=str(single_branch["loser"]),
                last_hitter_candidate=str(single_branch["last_hitter"]),
                preserve_model_labels=preserve_model_taxonomy_labels,
                prompt_family_flag=prompt_family_flag_value,
                extra_flags=list(single_branch["consistency_flags"]),
                extra_change_fields={
                    "winner_category_consistency_flags": list(single_branch["consistency_flags"]),
                    "winner_loser_touched_final_ball": str(single_branch["loser_touched_final_ball"]),
                    "winner_final_return_crossed_net": str(single_branch["final_return_crossed_net"]),
                    "winner_final_return_landed_in": str(single_branch["final_return_landed_in"]),
                },
            )
            csv_rows.append(
                {
                    "id": point.id,
                    "t_start": float(point.t_start),
                    "t_end": float(point.t_end),
                    "clip_start": window_start,
                    "clip_end": window_end,
                    "winner_candidate": winner,
                    "winner_decision": point.winner_decision,
                    "winner_confidence": point.winner_confidence,
                    "winner_score_a": point.winner_score_a,
                    "winner_score_b": point.winner_score_b,
                    "winner_end_category": point.winner_end_category,
                    "winner_loser_candidate": point.winner_loser_candidate,
                    "winner_last_hitter_candidate": point.winner_last_hitter_candidate,
                    "raw_output": raw_text,
                    "raw_output_a": str(single_branch["raw_text_a"]),
                    "raw_output_b": str(single_branch["raw_text_b"]),
                    "raw_output_tiebreak": str(single_branch["raw_text_tiebreak"]),
                    "file": Path(str(single_branch["clip_path"])).name,
                    "ensemble_status": "single",
                    "primary_branch": "",
                    "branch_raw_winner_candidate": "",
                    "branch_raw_output_a": "",
                    "branch_raw_output_b": "",
                    "branch_raw_output_tiebreak": "",
                    "branch_raw_file": "",
                    "branch_augv1_winner_candidate": "",
                    "branch_augv1_output_a": "",
                    "branch_augv1_output_b": "",
                    "branch_augv1_output_tiebreak": "",
                    "branch_augv1_file": "",
                }
            )
        if temp_clip.exists():
            temp_clip.unlink()
        processed += 1
        print(
            f"   > {point.id}: {winner} | {point.winner_decision} | "
            f"window={window_start:.2f}->{window_end:.2f} | "
            f"raw={raw_text!r}"
        )

    inference_suffix = "dual4b_raw_augv1" if str(args.winner_mode) == "dual4b_raw_augv1" else "single"
    timeline.analysis_metadata["winner_inference_mode"] = f"transformers_native_video_{model_slug}_{inference_suffix}"
    timeline.analysis_metadata["winner_native_video_model_dir"] = str(Path(args.model_dir))
    timeline.analysis_metadata["winner_vlm_window_ratio"] = float(args.window_ratio)
    timeline.analysis_metadata["winner_full_rally_threshold_sec"] = float(args.full_rally_threshold_sec)
    timeline.analysis_metadata["winner_native_video_clip_scope"] = "full_frozen_rally"
    timeline.analysis_metadata["winner_native_video_winner_mode"] = str(args.winner_mode)
    timeline.analysis_metadata["winner_native_video_prompt_family"] = str(args.winner_prompt_family)
    timeline.analysis_metadata["winner_native_video_ensemble_primary_branch"] = str(args.ensemble_primary_branch)
    timeline.analysis_metadata["winner_native_video_fps_sample"] = float(args.fps_sample)
    timeline.analysis_metadata["winner_native_video_min_frames"] = int(args.min_frames)
    timeline.analysis_metadata["winner_native_video_max_frames"] = int(args.max_frames)
    timeline.analysis_metadata["winner_native_video_max_pixels"] = int(args.max_pixels)
    timeline.analysis_metadata["winner_native_video_size_shortest_edge"] = int(args.size_shortest_edge)
    timeline.analysis_metadata["winner_native_video_size_longest_edge"] = int(args.size_longest_edge)
    timeline.analysis_metadata["winner_native_video_main_pass_view"] = str(args.main_pass_view)
    timeline.analysis_metadata["winner_native_video_roi_margin_ratio"] = float(args.roi_margin_ratio)
    timeline.analysis_metadata["winner_native_video_roi_margin_y_ratio"] = (
        float(args.roi_margin_ratio) if float(args.roi_margin_y_ratio) < 0 else float(args.roi_margin_y_ratio)
    )
    timeline.analysis_metadata["winner_native_video_table_only_x_margin_ratio"] = float(args.table_only_x_margin_ratio)
    timeline.analysis_metadata["winner_native_video_table_only_top_margin_ratio"] = float(args.table_only_top_margin_ratio)
    timeline.analysis_metadata["winner_native_video_table_only_bottom_margin_ratio"] = float(args.table_only_bottom_margin_ratio)
    timeline.analysis_metadata["winner_native_video_flip_main_pass"] = bool(args.flip_main_pass)
    timeline.analysis_metadata["winner_native_video_clip_dir"] = str(clip_dir)
    timeline.analysis_metadata["winner_native_video_main_pass_overlay"] = str(args.main_pass_overlay)
    timeline.analysis_metadata["winner_native_video_aug_ball_profile"] = str(args.aug_ball_profile)
    timeline.analysis_metadata["winner_native_video_aug_ball_trail_length"] = int(args.aug_ball_trail_length)
    timeline.analysis_metadata["winner_native_video_fewshot_path"] = str(fewshot_path)
    timeline.analysis_metadata["winner_native_video_fewshot_max_examples"] = int(args.winner_fewshot_max_examples)
    boundary_signature_after = _frozen_boundary_signature(timeline.points)
    if boundary_signature_after != boundary_signature_before:
        raise RuntimeError("Winner phase must not modify frozen rally boundaries (id/t_start/t_end).")
    save_rally_timeline(output_path, timeline)

    csv_path = clip_dir / "rally_clips.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "t_start",
                "t_end",
                "clip_start",
                "clip_end",
                "winner_candidate",
                "winner_decision",
                "winner_confidence",
                "winner_score_a",
                "winner_score_b",
                "winner_end_category",
                "winner_loser_candidate",
                "winner_last_hitter_candidate",
                "raw_output",
                "raw_output_a",
                "raw_output_b",
                "raw_output_tiebreak",
                "file",
                "ensemble_status",
                "primary_branch",
                "branch_raw_winner_candidate",
                "branch_raw_output_a",
                "branch_raw_output_b",
                "branch_raw_output_tiebreak",
                "branch_raw_file",
                "branch_augv1_winner_candidate",
                "branch_augv1_output_a",
                "branch_augv1_output_b",
                "branch_augv1_output_tiebreak",
                "branch_augv1_file",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n--- DONE ---")
    print(f"Processed {processed} rallies. Output: {output_path}")
    print(f"Clips: {clip_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
