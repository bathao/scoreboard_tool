from __future__ import annotations

from dataclasses import dataclass, replace
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

from backend.config import PROJECT_ROOT
from backend.rally_timeline_contract import (
    RallyTimeline,
    counts_toward_score,
    load_rally_timeline,
    save_rally_timeline,
)
from backend.ai_multistream_rally import _infer_player_serve_mode_from_starter_roles


SCRIPTS_DIR = PROJECT_ROOT / "scripts"
STEP3_1_ALGORITHM = "total_rally_start_time_review_v2"


@dataclass(frozen=True)
class Step3PlayerContext:
    """Trusted Step 2 identity mapping for the initial set-1 sides."""

    player_a_name: str
    player_b_name: str
    player_a_starts_near: bool = True


@dataclass
class Step3RallyStartReviewResult:
    timeline: RallyTimeline
    events: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class Step3SideStateReviewResult:
    events: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass
class Step3LogicAuditResult:
    events: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class Step3SideIdentificationConfig:
    """Local side-evidence scan around one rally start."""

    window_before_sec: float = 1.00
    window_after_sec: float = 4.00
    break_gap_sec: float = 12.00
    next_event_guard_sec: float = 0.25
    sample_fps: float = 4.0
    match_threshold: float = 0.35
    match_margin: float = 0.04
    min_best_similarity: float = 0.45
    min_avg_similarity: float = 0.38
    min_face_score: float = 0.35
    min_accepted_samples: int = 4
    enable_jersey_fallback: bool = False
    jersey_anchor_window_sec: float = 45.0
    jersey_sample_fps: float = 2.0
    jersey_match_margin: float = 0.035
    jersey_max_distance: float = 0.55
    retry_unknown_enabled: bool = True
    retry_window_after_sec: float = 12.00
    retry_sample_fps: float = 8.0
    retry_min_accepted_samples: int = 3
    promote_strong_candidate_enabled: bool = True
    promote_min_samples: int = 3
    promote_min_best_similarity: float = 0.55
    promote_min_avg_similarity: float = 0.50
    promote_min_margin: float = 0.25
    continuity_fill_unknown_enabled: bool = True
    continuity_terminal_max_gap_sec: float = 12.00
    player_zone_expand_x: float = 0.25
    player_zone_expand_y: float = 1.10


@dataclass(frozen=True)
class Step3LogicAuditConfig:
    """Rule audit after Step 3.2 side-state detection.

    Step 3.3 validates the server timeline by player identity, not by the raw
    A/B tracker role. LET rows replay the current server and do not advance the
    scoring serve index.
    """

    max_repair_iterations: int = 2
    rescan_neighbor_radius: int = 1
    set_boundary_gap_sec: float = 12.0
    min_scoring_before_set_boundary: int = 11
    deuce_switch_after_scoring_count: int = 20
    max_issues_in_summary: int = 80


def _ensure_scripts_importable() -> None:
    scripts_path = str(SCRIPTS_DIR)
    if scripts_path not in sys.path:
        sys.path.append(scripts_path)


def _load_build_rally_timeline():
    _ensure_scripts_importable()
    from generate_rally_timeline import build_rally_timeline  # type: ignore

    return build_rally_timeline


def server_identity_for_starter_role(
    starter_role: str | None,
    player_context: Step3PlayerContext | None,
) -> dict[str, str]:
    """Map role-tracker starter role A/B to the trusted Step 2 player name.

    The current role tracker seeds role A on the initial near-side player and
    role B on the initial far-side player. Step 2 owns the actual names.
    """

    role = str(starter_role or "").strip()
    if role not in {"A", "B"}:
        return {
            "server_initial_side": "",
            "server_player_key": "",
            "server_player_name": "unknown",
            "server_identity_source": "unknown_starter_role",
        }
    if player_context is None:
        return {
            "server_initial_side": "near" if role == "A" else "far",
            "server_player_key": "",
            "server_player_name": "unknown",
            "server_identity_source": "starter_role_only",
        }

    near_key = "player_a" if bool(player_context.player_a_starts_near) else "player_b"
    far_key = "player_b" if bool(player_context.player_a_starts_near) else "player_a"
    key = near_key if role == "A" else far_key
    name = player_context.player_a_name if key == "player_a" else player_context.player_b_name
    return {
        "server_initial_side": "near" if role == "A" else "far",
        "server_player_key": key,
        "server_player_name": str(name).strip() or "unknown",
        "server_identity_source": "step2_initial_role_map",
    }


def starter_role_side_hint(starter_role: str | None) -> str:
    """Return the detector's raw side hint for a starter role.

    This is only a low-level tracker hint. It is not enough to identify the
    current player after a side swap; Step 3.2 must pair it with independent
    per-rally side evidence before exposing a server/current_side result.
    """

    role = str(starter_role or "").strip()
    if role == "A":
        return "NEAR"
    if role == "B":
        return "FAR"
    return "unknown"


def _opposite_side(side: str | None) -> str:
    side_norm = str(side or "").strip().upper()
    if side_norm == "NEAR":
        return "FAR"
    if side_norm == "FAR":
        return "NEAR"
    return "unknown"


def _player_name_for_key(player_key: str | None, player_context: Step3PlayerContext | None) -> str:
    if player_context is None:
        return "unknown"
    if player_key == "player_a":
        return str(player_context.player_a_name).strip() or "unknown"
    if player_key == "player_b":
        return str(player_context.player_b_name).strip() or "unknown"
    return "unknown"


def infer_current_sides_from_single_player(
    *,
    identified_player_key: str | None,
    identified_side: str | None,
) -> dict[str, str]:
    """Infer both trusted Step 2 players' current NEAR/FAR sides.

    Exactly two players are in the match. Once one trusted player is identified
    on one current side, the other player's current side follows by exclusion.
    """

    key = str(identified_player_key or "").strip()
    side = str(identified_side or "").strip().upper()
    if key not in {"player_a", "player_b"} or side not in {"NEAR", "FAR"}:
        return {
            "player_a_current_side": "unknown",
            "player_b_current_side": "unknown",
        }
    other_side = _opposite_side(side)
    if key == "player_a":
        return {
            "player_a_current_side": side,
            "player_b_current_side": other_side,
        }
    return {
        "player_a_current_side": other_side,
        "player_b_current_side": side,
    }


def server_identity_from_event_side_evidence(
    event: dict[str, Any],
    *,
    starter_role: str | None = None,
    player_context: Step3PlayerContext | None,
) -> dict[str, str]:
    """Map a starter role to a server only when current side evidence exists."""

    side = starter_role_side_hint(starter_role if starter_role is not None else event.get("starter_role"))
    if side not in {"NEAR", "FAR"} or player_context is None:
        return {
            "current_side": "unknown",
            "server_player_key": "",
            "server_player_name": "unknown",
            "server_identity_source": "unknown_current_side_evidence",
        }

    player_a_side = str(event.get("player_a_current_side", "") or "").strip().upper()
    player_b_side = str(event.get("player_b_current_side", "") or "").strip().upper()
    key = ""
    if player_a_side == side:
        key = "player_a"
    elif player_b_side == side:
        key = "player_b"
    if not key:
        return {
            "current_side": "unknown",
            "server_player_key": "",
            "server_player_name": "unknown",
            "server_identity_source": "unknown_current_side_evidence",
        }
    return {
        "current_side": side,
        "server_player_key": key,
        "server_player_name": _player_name_for_key(key, player_context),
        "server_identity_source": "per_rally_single_player_side_id",
    }


def initialize_side_evidence_fields(event: dict[str, Any]) -> None:
    """Make side-state fields explicit before any local identification scan."""

    event.setdefault("starter_role_side_hint", starter_role_side_hint(event.get("starter_role")))
    event.setdefault("current_side", "unknown")
    event.setdefault("player_a_current_side", "unknown")
    event.setdefault("player_b_current_side", "unknown")
    event.setdefault("side_evidence_source", "not_scanned")
    event.setdefault("side_evidence_status", "not_scanned")


def reset_side_evidence_fields(
    event: dict[str, Any],
    *,
    player_context: Step3PlayerContext | None,
) -> None:
    """Reset one event to Step 3.1 identity before rerunning Step 3.2 side ID."""

    for key in list(event.keys()):
        if key.startswith("side_") or key.startswith("player_a_current_side") or key.startswith("player_b_current_side"):
            event.pop(key, None)
    for key in (
        "current_side",
        "starter_role_side_hint",
        "side_identified_player_key",
        "side_identified_player_name",
        "side_identified_current_side",
        "side_identified_similarity",
        "side_identified_avg_similarity",
        "side_identified_margin",
        "side_identified_t_sec",
        "side_identified_first_t_sec",
        "side_identified_last_t_sec",
        "side_identified_sample_count",
        "side_evidence_candidate_player_key",
        "side_evidence_candidate_player_name",
        "side_evidence_candidate_current_side",
        "side_evidence_candidate_similarity",
        "side_evidence_candidate_avg_similarity",
        "side_evidence_candidate_margin",
        "side_evidence_candidate_sample_count",
        "side_evidence_reason",
    ):
        event.pop(key, None)

    initial_identity = server_identity_for_starter_role(event.get("starter_role"), player_context)
    event.update(initial_identity)
    event["initial_server_player_key"] = initial_identity.get("server_player_key", "")
    event["initial_server_player_name"] = initial_identity.get("server_player_name", "unknown")
    event["initial_server_identity_source"] = initial_identity.get("server_identity_source", "")
    initialize_side_evidence_fields(event)


def _norm_player_name(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _trusted_records_from_face_db(face_db: Any, player_context: Step3PlayerContext) -> tuple[dict[str, Any], list[str]]:
    wanted = {
        "player_a": _norm_player_name(player_context.player_a_name),
        "player_b": _norm_player_name(player_context.player_b_name),
    }
    found: dict[str, Any] = {}
    for record in getattr(face_db, "records", []) or []:
        name = _norm_player_name(getattr(record, "name", ""))
        for key, wanted_name in wanted.items():
            if wanted_name and name == wanted_name and key not in found:
                found[key] = record
    missing = [
        _player_name_for_key(key, player_context)
        for key in ("player_a", "player_b")
        if key not in found
    ]
    return found, missing


def _side_from_body_rank(body_rank: Any) -> str:
    try:
        rank = int(body_rank)
    except Exception:
        return "unknown"
    if rank == 0:
        return "NEAR"
    if rank == 1:
        return "FAR"
    return "unknown"


def select_single_player_side_evidence(
    face_results: list[dict[str, Any]],
    trusted_records: dict[str, Any],
    *,
    match_threshold: float,
    match_margin: float,
    min_best_similarity: float = 0.45,
    min_avg_similarity: float = 0.38,
    min_face_score: float,
    min_accepted_samples: int,
) -> dict[str, Any]:
    """Pick the best local evidence: one trusted player identified on one side."""

    from backend.player_identity import face_similarity

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    usable_samples = 0
    rejected_samples = 0
    for sample in face_results:
        side = _side_from_body_rank(sample.get("body_rank"))
        if side not in {"NEAR", "FAR"}:
            rejected_samples += 1
            continue
        face_score = float(sample.get("face_score", 0.0) or 0.0)
        if face_score < float(min_face_score):
            rejected_samples += 1
            continue
        embedding = sample.get("embedding")
        if embedding is None:
            rejected_samples += 1
            continue

        sims: list[tuple[str, Any, float]] = []
        for player_key, record in trusted_records.items():
            sims.append((player_key, record, face_similarity(embedding, record.embedding_array())))
        if not sims:
            rejected_samples += 1
            continue
        sims.sort(key=lambda row: row[2], reverse=True)
        best_key, best_record, best_sim = sims[0]
        second_sim = sims[1][2] if len(sims) > 1 else -1.0
        margin = float(best_sim - second_sim)
        usable_samples += 1
        if float(best_sim) < float(match_threshold) or margin < float(match_margin):
            rejected_samples += 1
            continue

        group_key = (best_key, side)
        group = groups.setdefault(
            group_key,
            {
                "player_key": best_key,
                "player_name": str(getattr(best_record, "name", "") or "unknown"),
                "side": side,
                "accepted_samples": 0,
                "similarity_sum": 0.0,
                "best_similarity": -1.0,
                "best_margin": -1.0,
                "best_t_sec": None,
                "first_t_sec": None,
                "last_t_sec": None,
            },
        )
        group["accepted_samples"] = int(group["accepted_samples"]) + 1
        group["similarity_sum"] = float(group["similarity_sum"]) + float(best_sim)
        if float(best_sim) > float(group["best_similarity"]):
            group["best_similarity"] = float(best_sim)
            group["best_margin"] = float(margin)
            group["best_t_sec"] = float(sample.get("t_sec", 0.0) or 0.0)
        t_sec = float(sample.get("t_sec", 0.0) or 0.0)
        group["first_t_sec"] = t_sec if group["first_t_sec"] is None else min(float(group["first_t_sec"]), t_sec)
        group["last_t_sec"] = t_sec if group["last_t_sec"] is None else max(float(group["last_t_sec"]), t_sec)

    if not groups:
        return {
            "side_evidence_source": "single_player_face_id_at_start",
            "side_evidence_status": "unknown",
            "side_evidence_sample_count": len(face_results),
            "side_evidence_usable_samples": usable_samples,
            "side_evidence_rejected_samples": rejected_samples,
            "side_evidence_reason": "no_trusted_player_match",
        }

    ranked = sorted(
        groups.values(),
        key=lambda row: (
            int(row["accepted_samples"]),
            float(row["similarity_sum"]) / max(1, int(row["accepted_samples"])),
            float(row["best_similarity"]),
        ),
        reverse=True,
    )
    best = ranked[0]
    avg_similarity = float(best["similarity_sum"]) / max(1, int(best["accepted_samples"]))
    weak_reasons: list[str] = []
    if int(best["accepted_samples"]) < int(min_accepted_samples):
        weak_reasons.append("not_enough_accepted_samples")
    if float(best["best_similarity"]) < float(min_best_similarity):
        weak_reasons.append("best_similarity_too_low")
    if float(avg_similarity) < float(min_avg_similarity):
        weak_reasons.append("avg_similarity_too_low")
    if weak_reasons:
        return {
            "side_evidence_source": "single_player_face_id_at_start",
            "side_evidence_status": "unknown",
            "side_evidence_candidate_player_key": best["player_key"],
            "side_evidence_candidate_player_name": best["player_name"],
            "side_evidence_candidate_current_side": best["side"],
            "side_evidence_candidate_similarity": float(best["best_similarity"]),
            "side_evidence_candidate_avg_similarity": avg_similarity,
            "side_evidence_candidate_margin": float(best["best_margin"]),
            "side_evidence_candidate_sample_count": int(best["accepted_samples"]),
            "side_evidence_sample_count": len(face_results),
            "side_evidence_usable_samples": usable_samples,
            "side_evidence_rejected_samples": rejected_samples,
            "side_evidence_reason": "|".join(weak_reasons),
        }

    return {
        "side_evidence_source": "single_player_face_id_at_start",
        "side_evidence_status": "identified",
        "side_identified_player_key": best["player_key"],
        "side_identified_player_name": best["player_name"],
        "side_identified_current_side": best["side"],
        "side_identified_similarity": float(best["best_similarity"]),
        "side_identified_avg_similarity": avg_similarity,
        "side_identified_margin": float(best["best_margin"]),
        "side_identified_t_sec": best["best_t_sec"],
        "side_identified_first_t_sec": best["first_t_sec"],
        "side_identified_last_t_sec": best["last_t_sec"],
        "side_identified_sample_count": int(best["accepted_samples"]),
        "side_evidence_sample_count": len(face_results),
        "side_evidence_usable_samples": usable_samples,
        "side_evidence_rejected_samples": rejected_samples,
    }


def _average_hists(hists: list[Any]) -> Any:
    if not hists:
        return None
    import numpy as np

    arr = np.stack(hists, axis=0).astype("float32")
    avg = arr.mean(axis=0)
    total = float(avg.sum())
    return avg / total if total > 1e-9 else avg


def _collect_jersey_hists_in_window(
    video_path: str | Path,
    *,
    t_start: float,
    t_end: float,
    yolo: Any,
    sample_fps: float,
    roi_xyxy: tuple[float, float, float, float] | None,
) -> list[dict[str, Any]]:
    import cv2
    from backend.player_identification import _detect_bodies_and_faces, extract_jersey_hist

    rows: list[dict[str, Any]] = []
    cap = cv2.VideoCapture(str(video_path))
    step_sec = 1.0 / max(0.1, float(sample_fps))
    t = max(0.0, float(t_start))
    hi = max(t, float(t_end))
    try:
        while t <= hi:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            dets = _detect_bodies_and_faces(frame, yolo, roi_xyxy=roi_xyxy)
            for rank, det in enumerate(dets[:2]):
                hist = extract_jersey_hist(frame, det["bbox_xyxy"])
                if hist is None:
                    continue
                rows.append(
                    {
                        "t_sec": t,
                        "body_rank": rank,
                        "side": _side_from_body_rank(rank),
                        "hist": hist,
                    }
                )
            t += step_sec
    finally:
        cap.release()
    return rows


def build_initial_jersey_anchors(
    video_path: str | Path,
    *,
    player_context: Step3PlayerContext,
    yolo: Any,
    roi_xyxy: tuple[float, float, float, float] | None,
    sample_fps: float,
    anchor_window_sec: float,
) -> dict[str, Any]:
    """Build player jersey anchors from the trusted initial Step 2 side state."""

    rows = _collect_jersey_hists_in_window(
        video_path,
        t_start=0.5,
        t_end=max(0.6, float(anchor_window_sec)),
        yolo=yolo,
        sample_fps=sample_fps,
        roi_xyxy=roi_xyxy,
    )
    by_side: dict[str, list[Any]] = {"NEAR": [], "FAR": []}
    for row in rows:
        side = str(row.get("side", "unknown"))
        if side in by_side:
            by_side[side].append(row["hist"])

    player_a_initial_side = "NEAR" if bool(player_context.player_a_starts_near) else "FAR"
    player_b_initial_side = _opposite_side(player_a_initial_side)
    anchors = {
        "player_a": _average_hists(by_side[player_a_initial_side]),
        "player_b": _average_hists(by_side[player_b_initial_side]),
    }
    return {
        "anchors": anchors,
        "sample_counts_by_side": {side: len(values) for side, values in by_side.items()},
        "player_a_initial_side": player_a_initial_side,
        "player_b_initial_side": player_b_initial_side,
    }


def select_single_player_side_evidence_from_jersey(
    jersey_rows: list[dict[str, Any]],
    jersey_anchors: dict[str, Any],
    player_context: Step3PlayerContext,
    *,
    match_margin: float,
    max_distance: float,
    min_accepted_samples: int,
) -> dict[str, Any]:
    """Pick the best local jersey evidence against trusted Step 2 anchors."""

    from backend.player_identification import jersey_distance

    anchors = dict(jersey_anchors.get("anchors") or {})
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    usable = 0
    rejected = 0
    for row in jersey_rows:
        side = str(row.get("side", "unknown") or "unknown").upper()
        hist = row.get("hist")
        if side not in {"NEAR", "FAR"} or hist is None:
            rejected += 1
            continue
        distances: list[tuple[str, float]] = []
        for player_key in ("player_a", "player_b"):
            anchor = anchors.get(player_key)
            if anchor is None:
                continue
            distances.append((player_key, jersey_distance(hist, anchor)))
        if len(distances) < 2:
            rejected += 1
            continue
        distances.sort(key=lambda item: item[1])
        best_key, best_dist = distances[0]
        second_dist = distances[1][1]
        margin = float(second_dist - best_dist)
        usable += 1
        if float(best_dist) > float(max_distance) or margin < float(match_margin):
            rejected += 1
            continue
        group_key = (best_key, side)
        group = groups.setdefault(
            group_key,
            {
                "player_key": best_key,
                "player_name": _player_name_for_key(best_key, player_context),
                "side": side,
                "accepted_samples": 0,
                "distance_sum": 0.0,
                "best_distance": 999.0,
                "best_margin": -1.0,
                "best_t_sec": None,
            },
        )
        group["accepted_samples"] = int(group["accepted_samples"]) + 1
        group["distance_sum"] = float(group["distance_sum"]) + float(best_dist)
        if float(best_dist) < float(group["best_distance"]):
            group["best_distance"] = float(best_dist)
            group["best_margin"] = float(margin)
            group["best_t_sec"] = float(row.get("t_sec", 0.0) or 0.0)

    if not groups:
        return {
            "side_evidence_source": "single_player_jersey_anchor_at_start",
            "side_evidence_status": "unknown",
            "side_evidence_sample_count": len(jersey_rows),
            "side_evidence_usable_samples": usable,
            "side_evidence_rejected_samples": rejected,
            "side_evidence_reason": "no_trusted_jersey_anchor_match",
        }

    ranked = sorted(
        groups.values(),
        key=lambda row: (
            int(row["accepted_samples"]),
            -float(row["distance_sum"]) / max(1, int(row["accepted_samples"])),
            float(row["best_margin"]),
        ),
        reverse=True,
    )
    best = ranked[0]
    if int(best["accepted_samples"]) < int(min_accepted_samples):
        return {
            "side_evidence_source": "single_player_jersey_anchor_at_start",
            "side_evidence_status": "unknown",
            "side_evidence_sample_count": len(jersey_rows),
            "side_evidence_usable_samples": usable,
            "side_evidence_rejected_samples": rejected,
            "side_evidence_reason": "not_enough_accepted_jersey_samples",
        }

    avg_distance = float(best["distance_sum"]) / max(1, int(best["accepted_samples"]))
    return {
        "side_evidence_source": "single_player_jersey_anchor_at_start",
        "side_evidence_status": "identified",
        "side_identified_player_key": best["player_key"],
        "side_identified_player_name": best["player_name"],
        "side_identified_current_side": best["side"],
        "side_identified_jersey_distance": float(best["best_distance"]),
        "side_identified_jersey_avg_distance": avg_distance,
        "side_identified_jersey_margin": float(best["best_margin"]),
        "side_identified_t_sec": best["best_t_sec"],
        "side_identified_sample_count": int(best["accepted_samples"]),
        "side_evidence_sample_count": len(jersey_rows),
        "side_evidence_usable_samples": usable,
        "side_evidence_rejected_samples": rejected,
    }


def apply_single_player_side_evidence(
    event: dict[str, Any],
    evidence: dict[str, Any],
    *,
    player_context: Step3PlayerContext | None,
) -> None:
    """Attach local side evidence and remap the server for one event."""

    event.update(evidence)
    if evidence.get("side_evidence_status") != "identified":
        event["current_side"] = "unknown"
        event["server_player_key"] = ""
        event["server_player_name"] = "unknown"
        event["server_identity_source"] = "unknown_current_side_evidence"
        return

    side_map = infer_current_sides_from_single_player(
        identified_player_key=str(evidence.get("side_identified_player_key", "") or ""),
        identified_side=str(evidence.get("side_identified_current_side", "") or ""),
    )
    event.update(side_map)
    event.update(server_identity_from_event_side_evidence(event, player_context=player_context))


def expected_server_identity_for_event_role(
    event: dict[str, Any],
    expected_role: str,
    player_context: Step3PlayerContext | None,
) -> dict[str, str]:
    """Use local side evidence when present; otherwise fall back to initial Step 2 mapping."""

    evidence_identity = server_identity_from_event_side_evidence(
        event,
        starter_role=expected_role,
        player_context=player_context,
    )
    if evidence_identity.get("server_identity_source") == "per_rally_single_player_side_id":
        return evidence_identity
    return server_identity_for_starter_role(expected_role, player_context)


def _derive_player_zone_from_table_roi(
    video_path: str | Path,
    table_roi: Any,
    *,
    expand_x: float = 0.25,
    expand_y: float = 1.10,
) -> tuple[float, float, float, float] | None:
    if table_roi is None:
        return None
    try:
        x = float(getattr(table_roi, "x"))
        y = float(getattr(table_roi, "y"))
        w = float(getattr(table_roi, "w"))
        h = float(getattr(table_roi, "h"))
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if frame_w <= 0 or frame_h <= 0:
        return None
    return (
        max(0.0, x - w * float(expand_x)),
        max(0.0, y - h * float(expand_y)),
        min(float(frame_w), x + w + w * float(expand_x)),
        min(float(frame_h), y + h + h * float(expand_y)),
    )


def _event_time(event: dict[str, Any], field: str, fallback_field: str, default: float = 0.0) -> float:
    value = event.get(field, event.get(fallback_field, default))
    try:
        return float(value if value is not None else default)
    except Exception:
        return float(default)


def _next_start_by_event_index(
    events: list[dict[str, Any]],
    *,
    time_field: str,
) -> dict[int, float | None]:
    ordered: list[tuple[int, float]] = [
        (idx, _event_time(event, time_field, "t_start"))
        for idx, event in enumerate(events)
    ]
    ordered.sort(key=lambda item: item[1])
    result: dict[int, float | None] = {}
    for pos, (idx, _t_start) in enumerate(ordered):
        result[idx] = ordered[pos + 1][1] if pos + 1 < len(ordered) else None
    return result


def _append_evidence_reason(evidence: dict[str, Any], reason: str) -> None:
    existing = str(evidence.get("side_evidence_reason", "") or "").strip()
    if not existing:
        evidence["side_evidence_reason"] = reason
        return
    parts = [part for part in existing.split("|") if part]
    if reason not in parts:
        parts.append(reason)
    evidence["side_evidence_reason"] = "|".join(parts)


def _side_identification_scan_plan(
    event: dict[str, Any],
    *,
    next_start: float | None,
    config: Step3SideIdentificationConfig,
    time_field: str = "t_start",
    end_time_field: str = "t_end",
) -> dict[str, Any]:
    """Return safe local scan windows for side ID at one rally start.

    The primary window is anchored at the rally start.  Post-rally extension is
    only allowed when the next rally starts soon enough that the current rally
    is not likely followed by a set break or side swap.
    """

    t_start = _event_time(event, time_field, "t_start")
    t_end = max(t_start, _event_time(event, end_time_field, "t_end", t_start))
    lo = max(0.0, t_start - float(config.window_before_sec))

    gap_after_sec = None
    if next_start is not None:
        gap_after_sec = float(next_start) - t_end
    blocks_post_rally = next_start is None
    block_reason = "terminal_event_blocks_post_rally_extension" if blocks_post_rally else ""
    if gap_after_sec is not None and gap_after_sec >= float(config.break_gap_sec):
        blocks_post_rally = True
        block_reason = "long_gap_blocks_post_rally_extension"

    primary_hi = t_start + float(config.window_after_sec)
    if blocks_post_rally:
        primary_hi = min(primary_hi, t_end)
    if next_start is not None:
        primary_hi = min(primary_hi, max(t_start, float(next_start) - float(config.next_event_guard_sec)))
    primary_hi = max(lo + 0.05, primary_hi)

    fallback_hi = max(primary_hi, t_end + float(config.window_after_sec))
    if next_start is not None:
        fallback_hi = min(fallback_hi, max(primary_hi, float(next_start) - float(config.next_event_guard_sec)))
    fallback_allowed = (not blocks_post_rally) and (fallback_hi > primary_hi + 1e-6)
    return {
        "t_start": t_start,
        "t_end": t_end,
        "next_start": next_start,
        "gap_after_sec": gap_after_sec,
        "blocks_post_rally": blocks_post_rally,
        "block_reason": block_reason,
        "primary_lo": lo,
        "primary_hi": primary_hi,
        "fallback_lo": lo,
        "fallback_hi": fallback_hi,
        "fallback_allowed": fallback_allowed,
    }


def _side_identification_retry_scan_window(
    plan: dict[str, Any],
    *,
    config: Step3SideIdentificationConfig,
) -> tuple[float, float] | None:
    """Return a longer start-anchored retry window for unresolved rows."""

    t_start = float(plan.get("t_start", 0.0) or 0.0)
    t_end = float(plan.get("t_end", t_start) or t_start)
    next_start = plan.get("next_start")
    lo = max(0.0, t_start - float(config.window_before_sec))
    hi = t_start + float(config.retry_window_after_sec)

    if bool(plan.get("blocks_post_rally")):
        hi = min(hi, t_end)
    if next_start is not None:
        hi = min(hi, max(t_start, float(next_start) - float(config.next_event_guard_sec)))

    hi = max(lo + 0.05, hi)
    if hi <= float(plan.get("primary_hi", lo)) + 1e-6:
        return None
    return lo, hi


def _promote_strong_candidate_if_safe(
    evidence: dict[str, Any],
    *,
    config: Step3SideIdentificationConfig,
) -> dict[str, Any]:
    """Accept a very strong candidate that only missed strict sample count."""

    if not bool(config.promote_strong_candidate_enabled):
        return evidence
    if evidence.get("side_evidence_status") == "identified":
        return evidence
    reason = str(evidence.get("side_evidence_reason", "") or "")
    if "not_enough_accepted_samples" not in reason:
        return evidence
    if "best_similarity_too_low" in reason or "avg_similarity_too_low" in reason:
        return evidence
    player_key = str(evidence.get("side_evidence_candidate_player_key", "") or "")
    side = str(evidence.get("side_evidence_candidate_current_side", "") or "").upper()
    if player_key not in {"player_a", "player_b"} or side not in {"NEAR", "FAR"}:
        return evidence
    sample_count = int(evidence.get("side_evidence_candidate_sample_count", 0) or 0)
    best = float(evidence.get("side_evidence_candidate_similarity", 0.0) or 0.0)
    avg = float(evidence.get("side_evidence_candidate_avg_similarity", 0.0) or 0.0)
    margin = float(evidence.get("side_evidence_candidate_margin", 0.0) or 0.0)
    if sample_count < int(config.promote_min_samples):
        return evidence
    if best < float(config.promote_min_best_similarity):
        return evidence
    if avg < float(config.promote_min_avg_similarity):
        return evidence
    if margin < float(config.promote_min_margin):
        return evidence

    promoted = dict(evidence)
    promoted["side_evidence_status"] = "identified"
    promoted["side_evidence_source"] = "single_player_face_id_at_start"
    promoted["side_evidence_reason"] = "promoted_strong_candidate"
    promoted["side_identified_player_key"] = player_key
    promoted["side_identified_player_name"] = evidence.get("side_evidence_candidate_player_name", "unknown")
    promoted["side_identified_current_side"] = side
    promoted["side_identified_similarity"] = best
    promoted["side_identified_avg_similarity"] = avg
    promoted["side_identified_margin"] = margin
    promoted["side_identified_sample_count"] = sample_count
    return promoted


def _current_side_map(event: dict[str, Any]) -> dict[str, str] | None:
    a_side = str(event.get("player_a_current_side", "") or "").upper()
    b_side = str(event.get("player_b_current_side", "") or "").upper()
    if a_side not in {"NEAR", "FAR"} or b_side not in {"NEAR", "FAR"}:
        return None
    if a_side == b_side:
        return None
    return {
        "player_a_current_side": a_side,
        "player_b_current_side": b_side,
    }


def _same_side_map(left: dict[str, str] | None, right: dict[str, str] | None) -> bool:
    return bool(left and right and left == right)


def _start_gap_chain_is_short(
    ordered: list[tuple[int, dict[str, Any]]],
    *,
    max_gap_sec: float,
    time_field: str,
) -> bool:
    if len(ordered) < 2:
        return True
    for pos in range(len(ordered) - 1):
        left_t = _event_time(ordered[pos][1], time_field, "t_start")
        right_t = _event_time(ordered[pos + 1][1], time_field, "t_start")
        if right_t - left_t > float(max_gap_sec):
            return False
    return True


def fill_unknown_side_state_by_continuity(
    events: list[dict[str, Any]],
    *,
    player_context: Step3PlayerContext | None,
    time_field: str,
    max_terminal_gap_sec: float,
) -> int:
    """Fill unresolved side state only when neighboring side maps are safe.

    This is not face identification. It is a conservative Step 3.2 inference
    layer after direct scan evidence has failed.
    """

    ordered = sorted(enumerate(events), key=lambda item: _event_time(item[1], time_field, "t_start"))
    filled_count = 0
    pos = 0
    while pos < len(ordered):
        _idx, event = ordered[pos]
        if _current_side_map(event) is not None:
            pos += 1
            continue

        group_start = pos
        while pos < len(ordered) and _current_side_map(ordered[pos][1]) is None:
            pos += 1
        group = ordered[group_start:pos]
        prev_item = ordered[group_start - 1] if group_start > 0 else None
        next_item = ordered[pos] if pos < len(ordered) else None
        prev_map = _current_side_map(prev_item[1]) if prev_item is not None else None
        next_map = _current_side_map(next_item[1]) if next_item is not None else None

        fill_map: dict[str, str] | None = None
        reason = ""
        if _same_side_map(prev_map, next_map):
            fill_map = prev_map
            reason = "side_continuity_between_matching_known_neighbors"
        elif prev_map is not None and next_item is None:
            chain = [prev_item, *group] if prev_item is not None else group
            if _start_gap_chain_is_short(chain, max_gap_sec=max_terminal_gap_sec, time_field=time_field):
                fill_map = prev_map
                reason = "side_continuity_terminal_fill"
        elif next_map is not None and prev_item is None:
            chain = [*group, next_item] if next_item is not None else group
            if _start_gap_chain_is_short(chain, max_gap_sec=max_terminal_gap_sec, time_field=time_field):
                fill_map = next_map
                reason = "side_continuity_leading_fill"

        if fill_map is None:
            continue

        for _event_idx, target in group:
            target.update(fill_map)
            target["side_evidence_status"] = "inferred"
            target["side_evidence_source"] = "side_continuity_fill"
            target["side_evidence_window_mode"] = "continuity_fill"
            target["side_evidence_reason"] = reason
            target.update(server_identity_from_event_side_evidence(target, player_context=player_context))
            filled_count += 1
    return filled_count


def annotate_events_with_single_player_side_identification(
    video_path: str | Path,
    events: list[dict[str, Any]],
    *,
    player_context: Step3PlayerContext | None,
    face_db_path: str | Path | None = None,
    face_model_path: str | Path | None = None,
    pose_weights_path: str | Path | None = None,
    table_roi: Any = None,
    player_zone_xyxy: tuple[float, float, float, float] | None = None,
    config: Step3SideIdentificationConfig | None = None,
    time_field: str = "t_start",
    end_time_field: str = "t_end",
    only_event_ids: set[str] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Identify one trusted Step 2 player near each rally start and infer sides."""

    config = config or Step3SideIdentificationConfig()
    if player_context is None:
        for event in events:
            apply_single_player_side_evidence(
                event,
                {
                    "side_evidence_source": "single_player_face_id_at_start",
                    "side_evidence_status": "unknown",
                    "side_evidence_reason": "missing_step2_player_context",
                },
                player_context=player_context,
            )
        return {"enabled": False, "reason": "missing_step2_player_context"}

    face_db_path = Path(face_db_path) if face_db_path is not None else PROJECT_ROOT / "data" / "players" / "faces.json"
    face_model_path = (
        Path(face_model_path)
        if face_model_path is not None
        else PROJECT_ROOT / "data" / "models" / "face" / "w600k_r50.onnx"
    )
    if pose_weights_path is None:
        pose_weights_path = PROJECT_ROOT / "weights" / "yolov8x-pose.pt"

    from backend.player_identity import FaceDB, FaceEmbedder
    from backend.player_identification import _collect_face_embeddings_in_window
    from ultralytics import YOLO

    face_db = FaceDB(face_db_path)
    trusted_records, missing = _trusted_records_from_face_db(face_db, player_context)
    if missing:
        for event in events:
            apply_single_player_side_evidence(
                event,
                {
                    "side_evidence_source": "single_player_face_id_at_start",
                    "side_evidence_status": "unknown",
                    "side_evidence_reason": "step2_player_missing_from_face_db",
                    "side_evidence_missing_players": missing,
                },
                player_context=player_context,
            )
        return {"enabled": False, "reason": "step2_player_missing_from_face_db", "missing_players": missing}

    derived_player_zone = _derive_player_zone_from_table_roi(
        video_path,
        table_roi,
        expand_x=float(config.player_zone_expand_x),
        expand_y=float(config.player_zone_expand_y),
    )
    if derived_player_zone is not None:
        player_zone_xyxy = derived_player_zone
    if player_zone_xyxy is None:
        for event in events:
            apply_single_player_side_evidence(
                event,
                {
                    "side_evidence_source": "single_player_side_id_at_start",
                    "side_evidence_status": "unknown",
                    "side_evidence_reason": "missing_table_roi_player_zone",
                },
                player_context=player_context,
            )
        return {"enabled": False, "reason": "missing_table_roi_player_zone"}

    if log_fn:
        log_fn("Step 3.2: local side evidence - loading GPU face/pose models")
    embedder = FaceEmbedder(face_model_path)
    yolo = YOLO(str(Path(pose_weights_path).resolve()))
    jersey_anchors = build_initial_jersey_anchors(
        video_path,
        player_context=player_context,
        yolo=yolo,
        roi_xyxy=player_zone_xyxy,
        sample_fps=float(config.jersey_sample_fps),
        anchor_window_sec=float(config.jersey_anchor_window_sec),
    )
    if log_fn:
        counts = jersey_anchors.get("sample_counts_by_side", {})
        log_fn(
            "Step 3.2: local side evidence - jersey anchors "
            f"NEAR={counts.get('NEAR', 0)} FAR={counts.get('FAR', 0)}"
        )

    def scan_window(
        lo: float,
        hi: float,
        *,
        window_mode: str,
        sample_fps: float,
        min_accepted_samples: int,
    ) -> dict[str, Any]:
        face_results = _collect_face_embeddings_in_window(
            str(video_path),
            lo,
            hi,
            yolo,
            embedder,
            sample_fps=float(sample_fps),
            roi_xyxy=player_zone_xyxy,
        )
        evidence = select_single_player_side_evidence(
            face_results,
            trusted_records,
            match_threshold=float(config.match_threshold),
            match_margin=float(config.match_margin),
            min_best_similarity=float(config.min_best_similarity),
            min_avg_similarity=float(config.min_avg_similarity),
            min_face_score=float(config.min_face_score),
            min_accepted_samples=int(min_accepted_samples),
        )
        if config.enable_jersey_fallback and evidence.get("side_evidence_status") != "identified":
            jersey_rows = _collect_jersey_hists_in_window(
                video_path,
                t_start=lo,
                t_end=hi,
                yolo=yolo,
                sample_fps=float(config.jersey_sample_fps),
                roi_xyxy=player_zone_xyxy,
            )
            evidence = select_single_player_side_evidence_from_jersey(
                jersey_rows,
                jersey_anchors,
                player_context,
                match_margin=float(config.jersey_match_margin),
                max_distance=float(config.jersey_max_distance),
                min_accepted_samples=int(min_accepted_samples),
            )
        evidence["side_evidence_window_start"] = lo
        evidence["side_evidence_window_end"] = hi
        evidence["side_evidence_window_mode"] = window_mode
        evidence["side_evidence_scan_fps"] = float(sample_fps)
        return evidence

    next_start_by_index = _next_start_by_event_index(events, time_field=time_field)
    identified_count = 0
    unknown_count = 0
    retry_attempted_count = 0
    retry_identified_count = 0
    promoted_count = 0
    requested_event_ids = {str(item) for item in only_event_ids or set()}
    for idx, event in enumerate(events, start=1):
        if requested_event_ids and str(event.get("id", "")) not in requested_event_ids:
            continue
        reset_side_evidence_fields(event, player_context=player_context)
        event_index = idx - 1
        plan = _side_identification_scan_plan(
            event,
            next_start=next_start_by_index.get(event_index),
            config=config,
            time_field=time_field,
            end_time_field=end_time_field,
        )
        evidence = scan_window(
            float(plan["primary_lo"]),
            float(plan["primary_hi"]),
            window_mode="start_anchor",
            sample_fps=float(config.sample_fps),
            min_accepted_samples=int(config.min_accepted_samples),
        )
        if evidence.get("side_evidence_status") != "identified":
            if bool(plan.get("fallback_allowed")):
                fallback_evidence = scan_window(
                    float(plan["fallback_lo"]),
                    float(plan["fallback_hi"]),
                    window_mode="safe_post_rally_extension",
                    sample_fps=float(config.sample_fps),
                    min_accepted_samples=int(config.min_accepted_samples),
                )
                if fallback_evidence.get("side_evidence_status") == "identified":
                    evidence = fallback_evidence
                elif not evidence.get("side_evidence_candidate_player_key") and fallback_evidence.get(
                    "side_evidence_candidate_player_key"
                ):
                    evidence = fallback_evidence
            elif plan.get("block_reason"):
                _append_evidence_reason(evidence, str(plan["block_reason"]))
        if config.retry_unknown_enabled and evidence.get("side_evidence_status") != "identified":
            retry_window = _side_identification_retry_scan_window(plan, config=config)
            if retry_window is not None:
                retry_attempted_count += 1
                retry_evidence = scan_window(
                    retry_window[0],
                    retry_window[1],
                    window_mode="unknown_retry_start_anchor",
                    sample_fps=float(config.retry_sample_fps),
                    min_accepted_samples=int(config.retry_min_accepted_samples),
                )
                retry_evidence["side_evidence_retry_of"] = evidence.get("side_evidence_window_mode", "")
                if retry_evidence.get("side_evidence_status") == "identified":
                    evidence = retry_evidence
                    retry_identified_count += 1
                elif not evidence.get("side_evidence_candidate_player_key") and retry_evidence.get(
                    "side_evidence_candidate_player_key"
                ):
                    evidence = retry_evidence
        evidence = _promote_strong_candidate_if_safe(evidence, config=config)
        if evidence.get("side_evidence_reason") == "promoted_strong_candidate":
            promoted_count += 1
        evidence["side_evidence_next_start"] = plan.get("next_start")
        evidence["side_evidence_gap_after_sec"] = plan.get("gap_after_sec")
        evidence["side_evidence_post_rally_extension_blocked"] = bool(plan.get("blocks_post_rally"))
        apply_single_player_side_evidence(event, evidence, player_context=player_context)
        if event.get("side_evidence_status") == "identified":
            identified_count += 1
        else:
            unknown_count += 1
        if log_fn and (idx == 1 or idx == len(events) or idx % 10 == 0):
            log_fn(
                "Step 3.2: local side evidence - "
                f"{idx}/{len(events)} scanned, identified={identified_count}, unknown={unknown_count}"
            )

    continuity_filled_count = 0
    if bool(config.continuity_fill_unknown_enabled) and not requested_event_ids:
        continuity_filled_count = fill_unknown_side_state_by_continuity(
            events,
            player_context=player_context,
            time_field=time_field,
            max_terminal_gap_sec=float(config.continuity_terminal_max_gap_sec),
        )
        if continuity_filled_count and log_fn:
            log_fn(
                "Step 3.2: local side evidence - "
                f"continuity filled {continuity_filled_count} remaining unknown row(s)"
            )

    identified_count = sum(1 for event in events if event.get("side_evidence_status") == "identified")
    inferred_count = sum(1 for event in events if event.get("side_evidence_status") == "inferred")
    unknown_count = sum(
        1 for event in events if event.get("side_evidence_status") in {"unknown", "not_scanned"}
    )

    return {
        "enabled": True,
        "algorithm": "per_rally_single_player_face_side_id_start_anchor_v2",
        "identified": identified_count,
        "inferred": inferred_count,
        "unknown": unknown_count,
        "player_zone_xyxy": list(player_zone_xyxy) if player_zone_xyxy is not None else None,
        "config": {
            "window_before_sec": config.window_before_sec,
            "window_after_sec": config.window_after_sec,
            "break_gap_sec": config.break_gap_sec,
            "next_event_guard_sec": config.next_event_guard_sec,
            "sample_fps": config.sample_fps,
            "match_threshold": config.match_threshold,
            "match_margin": config.match_margin,
            "min_best_similarity": config.min_best_similarity,
            "min_avg_similarity": config.min_avg_similarity,
            "min_face_score": config.min_face_score,
            "min_accepted_samples": config.min_accepted_samples,
            "enable_jersey_fallback": config.enable_jersey_fallback,
            "jersey_anchor_window_sec": config.jersey_anchor_window_sec,
            "jersey_sample_fps": config.jersey_sample_fps,
            "jersey_match_margin": config.jersey_match_margin,
            "jersey_max_distance": config.jersey_max_distance,
            "retry_unknown_enabled": config.retry_unknown_enabled,
            "retry_window_after_sec": config.retry_window_after_sec,
            "retry_sample_fps": config.retry_sample_fps,
            "retry_min_accepted_samples": config.retry_min_accepted_samples,
            "promote_strong_candidate_enabled": config.promote_strong_candidate_enabled,
            "promote_min_samples": config.promote_min_samples,
            "promote_min_best_similarity": config.promote_min_best_similarity,
            "promote_min_avg_similarity": config.promote_min_avg_similarity,
            "promote_min_margin": config.promote_min_margin,
            "continuity_fill_unknown_enabled": config.continuity_fill_unknown_enabled,
            "continuity_terminal_max_gap_sec": config.continuity_terminal_max_gap_sec,
            "player_zone_expand_x": config.player_zone_expand_x,
            "player_zone_expand_y": config.player_zone_expand_y,
            "targeted_event_ids": sorted(requested_event_ids),
        },
        "fast_pass_identified": identified_count - retry_identified_count,
        "retry_attempted": retry_attempted_count,
        "retry_identified": retry_identified_count,
        "promoted_strong_candidate": promoted_count,
        "continuity_filled": continuity_filled_count,
        "jersey_anchor_sample_counts_by_side": jersey_anchors.get("sample_counts_by_side", {}),
    }


def _base_event(
    *,
    kind: str,
    source: str,
    point_id: str,
    t_start: float,
    t_end: float,
    starter_role: str,
    flags: list[str],
    player_context: Step3PlayerContext | None,
    source_time_offset_sec: float = 0.0,
) -> dict[str, Any]:
    source_t_start = float(t_start) + float(source_time_offset_sec)
    source_t_end = float(t_end) + float(source_time_offset_sec)
    event = {
        "id": "",
        "kind": str(kind),
        "source": str(source),
        "point_id": str(point_id),
        "t_start": float(t_start),
        "t_end": float(t_end),
        "source_t_start": source_t_start,
        "source_t_end": source_t_end,
        "starter_role": str(starter_role or ""),
        "flags": list(flags or []),
    }
    initial_identity = server_identity_for_starter_role(event["starter_role"], player_context)
    event.update(initial_identity)
    event["initial_server_player_key"] = initial_identity.get("server_player_key", "")
    event["initial_server_player_name"] = initial_identity.get("server_player_name", "unknown")
    event["initial_server_identity_source"] = initial_identity.get("server_identity_source", "")
    initialize_side_evidence_fields(event)
    return event


def timeline_total_rally_start_events(
    timeline: RallyTimeline,
    *,
    player_context: Step3PlayerContext | None = None,
    include_serve_order_review_markers: bool = True,
    source_time_offset_sec: float = 0.0,
) -> list[dict[str, Any]]:
    """Merge scoring points and existing LET starts into one review list."""

    events: list[dict[str, Any]] = []
    for point in timeline.points:
        is_scoring = counts_toward_score(point)
        events.append(
            _base_event(
                kind="scoring" if is_scoring else "let",
                source="timeline_point",
                point_id=str(point.id),
                t_start=float(point.t_start),
                t_end=float(point.t_end),
                starter_role=point.starter_role or "",
                flags=list(point.flags),
                player_context=player_context,
                source_time_offset_sec=source_time_offset_sec,
            )
        )

    metadata = timeline.analysis_metadata if isinstance(timeline.analysis_metadata, dict) else {}
    for bucket in ("excluded_let_starts", "unattached_trailing_let_starts"):
        for item in metadata.get(bucket, []) or []:
            if not isinstance(item, dict):
                continue
            t_start = float(item.get("t_start", 0.0))
            events.append(
                _base_event(
                    kind="let",
                    source=bucket,
                    point_id="",
                    t_start=t_start,
                    t_end=float(item.get("t_end", t_start)),
                    starter_role=str(item.get("starter_role", "") or ""),
                    flags=list(item.get("flags", []) or []),
                    player_context=player_context,
                    source_time_offset_sec=source_time_offset_sec,
                )
            )

    events.sort(key=lambda row: (float(row["t_start"]), float(row["t_end"])))
    if include_serve_order_review_markers:
        events.extend(
            serve_order_review_markers(
                events,
                player_context=player_context,
                source_time_offset_sec=source_time_offset_sec,
            )
        )
        events.sort(key=lambda row: (float(row["t_start"]), float(row["t_end"])))
    for idx, event in enumerate(events, start=1):
        event["id"] = f"rally_{idx:04d}"
    annotate_serve_order_rule_reviews(events, player_context=player_context)
    return events


def serve_order_review_markers(
    events: list[dict[str, Any]],
    *,
    player_context: Step3PlayerContext | None = None,
    source_time_offset_sec: float = 0.0,
) -> list[dict[str, Any]]:
    """Add review-only markers when the existing serve-order engine sees a gap.

    This does not create a confirmed rally. It only marks a timestamp for the
    operator when double-serve order has a single scoring serve between two
    complete runs from the other player.
    """

    scoring_events = [event for event in events if event.get("kind") == "scoring"]
    starter_roles = [str(event.get("starter_role", "") or "") for event in scoring_events]
    if len(starter_roles) < 5:
        return []
    if _infer_player_serve_mode_from_starter_roles(starter_roles) != "double":
        return []

    runs: list[tuple[str, int, int]] = []
    run_start = 0
    while run_start < len(starter_roles):
        role = starter_roles[run_start]
        run_end = run_start + 1
        while run_end < len(starter_roles) and starter_roles[run_end] == role:
            run_end += 1
        runs.append((role, run_start, run_end))
        run_start = run_end

    markers: list[dict[str, Any]] = []
    for idx in range(1, len(runs) - 1):
        left_role, left_start, left_end = runs[idx - 1]
        mid_role, mid_start, mid_end = runs[idx]
        right_role, right_start, right_end = runs[idx + 1]
        left_len = left_end - left_start
        mid_len = mid_end - mid_start
        right_len = right_end - right_start
        if mid_len != 1:
            continue
        if left_role != right_role or left_role == mid_role:
            continue
        if left_len < 2 or right_len < 2:
            continue

        singleton = scoring_events[mid_start]
        next_event = scoring_events[right_start]
        gap_start = float(singleton.get("t_end", singleton.get("t_start", 0.0)))
        gap_end = float(next_event.get("t_start", gap_start))
        if gap_end - gap_start < 4.0:
            continue
        marker_t = float((gap_start + gap_end) / 2.0)
        marker = _base_event(
            kind="needs_review",
            source="serve_order_singleton_gap",
            point_id="",
            t_start=marker_t,
            t_end=marker_t,
            starter_role=mid_role,
            flags=[
                "serve_order_gap_review",
                "not_confirmed_rally",
                "serve_mode_double",
            ],
            player_context=player_context,
            source_time_offset_sec=source_time_offset_sec,
        )
        marker["review_reason"] = "double_serve_singleton_gap"
        marker["review_note"] = (
            "Double-serve order has a singleton scoring serve between two "
            "complete runs from the other player; operator should verify a "
            "missing rally start in this gap."
        )
        marker["gap_start"] = gap_start
        marker["gap_end"] = gap_end
        marker["source_gap_start"] = gap_start + float(source_time_offset_sec)
        marker["source_gap_end"] = gap_end + float(source_time_offset_sec)
        marker["prev_scoring_event_id"] = singleton.get("id", "")
        marker["next_scoring_event_id"] = next_event.get("id", "")
        markers.append(marker)

    return markers


def _append_unique_flag(event: dict[str, Any], flag: str) -> None:
    flags = list(event.get("flags", []) or [])
    if flag not in flags:
        flags.append(flag)
    event["flags"] = flags


def _expected_serve_role(first_role: str, score_index: int, legal_limit: int) -> str:
    other_role = "A" if first_role == "B" else "B"
    return first_role if (score_index // legal_limit) % 2 == 0 else other_role


def annotate_serve_order_rule_reviews(
    events: list[dict[str, Any]],
    *,
    player_context: Step3PlayerContext | None = None,
) -> None:
    """Flag detected rows that conflict with the inferred serve-order rule.

    Scoring rows and review-only gap markers advance the expected serve order.
    LET rows do not advance it, so a LET should match the next expected server.
    """

    scoring_like_events = [
        event
        for event in events
        if event.get("kind") in {"scoring", "needs_review"}
        and str(event.get("starter_role", "") or "") in {"A", "B"}
    ]
    starter_roles = [str(event.get("starter_role", "") or "") for event in scoring_like_events]
    if not starter_roles:
        return

    serve_mode = _infer_player_serve_mode_from_starter_roles(starter_roles)
    legal_limit = 2 if serve_mode == "double" else 1
    first_role = starter_roles[0]

    for score_index, event in enumerate(scoring_like_events):
        expected_role = _expected_serve_role(first_role, score_index, legal_limit)
        expected_identity = expected_server_identity_for_event_role(event, expected_role, player_context)
        event["serve_order_mode"] = serve_mode
        event["serve_order_index"] = score_index + 1
        event["serve_order_expected_role"] = expected_role
        event["serve_order_expected_server_name"] = expected_identity.get("server_player_name", "unknown")
        if str(event.get("starter_role", "") or "") == expected_role:
            event["serve_order_ok"] = True
            continue
        event["serve_order_ok"] = False
        event["review_reason"] = event.get("review_reason") or "serve_order_role_conflict"
        event["review_note"] = event.get("review_note") or (
            "Detected scoring-like start conflicts with the inferred table-tennis serve order."
        )
        _append_unique_flag(event, "serve_order_role_conflict")

    scoring_like_by_time = sorted(scoring_like_events, key=lambda row: float(row.get("t_start", 0.0)))
    for event in sorted(events, key=lambda row: float(row.get("t_start", 0.0))):
        if event.get("kind") != "let":
            continue
        role = str(event.get("starter_role", "") or "")
        if role not in {"A", "B"}:
            continue

        event_t = float(event.get("t_start", 0.0))
        score_index = 0
        for score_event in scoring_like_by_time:
            if float(score_event.get("t_start", 0.0)) < event_t:
                score_index += 1
                continue
            break

        expected_role = _expected_serve_role(first_role, score_index, legal_limit)
        expected_identity = expected_server_identity_for_event_role(event, expected_role, player_context)
        event["serve_order_mode"] = serve_mode
        event["serve_order_next_score_index"] = score_index + 1
        event["serve_order_expected_role"] = expected_role
        event["serve_order_expected_server_name"] = expected_identity.get("server_player_name", "unknown")
        if role == expected_role:
            event["serve_order_ok"] = True
            continue

        event["serve_order_ok"] = False
        event["review_reason"] = event.get("review_reason") or "let_server_conflicts_with_expected_turn"
        event["review_note"] = (
            "Detected LET does not match the server expected by table-tennis serve order. "
            "LET should replay the same server instead of switching service."
        )
        _append_unique_flag(event, "serve_order_role_conflict")
        _append_unique_flag(event, "let_server_conflict")


def export_rally_start_event_frames(
    video_path: str | Path,
    events: list[dict[str, Any]],
    *,
    table_roi,
    out_dir: Path,
    frame_time_field: str = "t_start",
) -> dict[str, Any]:
    """Export one annotated frame and a CSV row per rally/LET start."""

    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    for old_image in out_dir.glob("*.jpg"):
        try:
            old_image.unlink()
        except OSError:
            pass
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for rally-start frame export: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    roi_tuple = None
    if table_roi is not None:
        roi_tuple = (
            int(getattr(table_roi, "x")),
            int(getattr(table_roi, "y")),
            int(getattr(table_roi, "w")),
            int(getattr(table_roi, "h")),
        )

    rows: list[dict[str, Any]] = []
    try:
        for idx, event in enumerate(events, start=1):
            t_start = float(event["t_start"])
            t_end = float(event.get("t_end", t_start))
            frame_time = float(event.get(frame_time_field, t_start) or t_start)
            frame_idx = max(0, int(round(frame_time * fps)))
            if frame_count > 0:
                frame_idx = min(frame_idx, max(0, frame_count - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            image_file = ""
            if ret and frame is not None:
                if roi_tuple is not None:
                    x, y, w, h = roi_tuple
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
                    cv2.putText(
                        frame,
                        "TABLE ROI",
                        (x, max(30, y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 0, 0),
                        3,
                    )

                kind = str(event.get("kind", "")).upper()
                if kind == "SCORING":
                    color = (80, 255, 120)
                elif kind == "NEEDS_REVIEW":
                    color = (80, 170, 255)
                else:
                    color = (80, 210, 255)
                server_name = str(event.get("server_player_name", "unknown") or "unknown")
                cv2.rectangle(frame, (40, 40), (1540, 250), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"STEP 3.1 START #{idx:04d}  {kind}",
                    (60, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.15,
                    color,
                    3,
                )
                cv2.putText(
                    frame,
                    f"start={t_start:.3f}s  end={t_end:.3f}s  frame={frame_idx}",
                    (60, 124),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )
                source_t_start = event.get("source_t_start")
                if source_t_start is not None and abs(float(source_t_start) - t_start) > 0.001:
                    cv2.putText(
                        frame,
                        f"source video t={float(source_t_start):.3f}s",
                        (860, 124),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.82,
                        (255, 255, 255),
                        2,
                    )
                cv2.putText(
                    frame,
                    f"server={server_name}  current_side={event.get('current_side', '') or '-'}  role={event.get('starter_role', '') or '-'}",
                    (60, 164),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.82,
                    (220, 220, 220),
                    2,
                )
                cv2.putText(
                    frame,
                    f"source={event.get('source', '')}  point={event.get('point_id', '') or '-'}",
                    (60, 204),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.82,
                    (220, 220, 220),
                    2,
                )
                image_file = f"{event['id']}_{event['kind']}_{frame_time:08.3f}s.jpg"
                cv2.imwrite(str(out_dir / image_file), frame)

            event["frame_idx"] = int(frame_idx)
            event["image_file"] = image_file
            rows.append(
                {
                    "id": event["id"],
                    "kind": event["kind"],
                    "source": event["source"],
                    "point_id": event.get("point_id", ""),
                    "t_start": f"{t_start:.6f}",
                    "t_end": f"{t_end:.6f}",
                    "source_t_start": f"{float(event.get('source_t_start', t_start)):.6f}",
                    "source_t_end": f"{float(event.get('source_t_end', t_end)):.6f}",
                    "frame_idx": int(frame_idx),
                    "image_file": image_file,
                    "starter_role": event.get("starter_role", ""),
                    "server_initial_side": event.get("server_initial_side", ""),
                    "server_player_key": event.get("server_player_key", ""),
                    "server_player_name": event.get("server_player_name", ""),
                    "server_identity_source": event.get("server_identity_source", ""),
                    "current_side": event.get("current_side", ""),
                    "starter_role_side_hint": event.get("starter_role_side_hint", ""),
                    "player_a_current_side": event.get("player_a_current_side", ""),
                    "player_b_current_side": event.get("player_b_current_side", ""),
                    "side_evidence_source": event.get("side_evidence_source", ""),
                    "side_evidence_status": event.get("side_evidence_status", ""),
                    "side_identified_player_key": event.get("side_identified_player_key", ""),
                    "side_identified_player_name": event.get("side_identified_player_name", ""),
                    "side_identified_current_side": event.get("side_identified_current_side", ""),
                    "side_identified_similarity": event.get("side_identified_similarity", ""),
                    "side_identified_margin": event.get("side_identified_margin", ""),
                    "side_identified_jersey_distance": event.get("side_identified_jersey_distance", ""),
                    "side_identified_jersey_margin": event.get("side_identified_jersey_margin", ""),
                    "side_identified_sample_count": event.get("side_identified_sample_count", ""),
                    "side_evidence_candidate_player_key": event.get("side_evidence_candidate_player_key", ""),
                    "side_evidence_candidate_player_name": event.get("side_evidence_candidate_player_name", ""),
                    "side_evidence_candidate_current_side": event.get("side_evidence_candidate_current_side", ""),
                    "side_evidence_candidate_similarity": event.get("side_evidence_candidate_similarity", ""),
                    "side_evidence_candidate_avg_similarity": event.get("side_evidence_candidate_avg_similarity", ""),
                    "side_evidence_candidate_sample_count": event.get("side_evidence_candidate_sample_count", ""),
                    "side_evidence_reason": event.get("side_evidence_reason", ""),
                    "serve_order_mode": event.get("serve_order_mode", ""),
                    "serve_order_index": event.get("serve_order_index", ""),
                    "serve_order_next_score_index": event.get("serve_order_next_score_index", ""),
                    "serve_order_expected_role": event.get("serve_order_expected_role", ""),
                    "serve_order_expected_server_name": event.get("serve_order_expected_server_name", ""),
                    "serve_order_ok": event.get("serve_order_ok", ""),
                    "review_reason": event.get("review_reason", ""),
                    "review_note": event.get("review_note", ""),
                    "gap_start": event.get("gap_start", ""),
                    "gap_end": event.get("gap_end", ""),
                    "source_gap_start": event.get("source_gap_start", ""),
                    "source_gap_end": event.get("source_gap_end", ""),
                    "flags": "|".join(str(flag) for flag in event.get("flags", []) or []),
                }
            )
    finally:
        cap.release()

    csv_path = out_dir / "rally_start_times.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "id",
            "kind",
            "source",
            "point_id",
            "t_start",
            "t_end",
            "source_t_start",
            "source_t_end",
            "frame_idx",
            "image_file",
            "starter_role",
            "server_initial_side",
            "server_player_key",
            "server_player_name",
            "server_identity_source",
            "current_side",
            "starter_role_side_hint",
            "player_a_current_side",
            "player_b_current_side",
            "side_evidence_source",
            "side_evidence_status",
            "side_identified_player_key",
            "side_identified_player_name",
            "side_identified_current_side",
            "side_identified_similarity",
            "side_identified_margin",
            "side_identified_jersey_distance",
            "side_identified_jersey_margin",
            "side_identified_sample_count",
            "side_evidence_candidate_player_key",
            "side_evidence_candidate_player_name",
            "side_evidence_candidate_current_side",
            "side_evidence_candidate_similarity",
            "side_evidence_candidate_avg_similarity",
            "side_evidence_candidate_sample_count",
            "side_evidence_reason",
            "serve_order_mode",
            "serve_order_index",
            "serve_order_next_score_index",
            "serve_order_expected_role",
            "serve_order_expected_server_name",
            "serve_order_ok",
            "review_reason",
            "review_note",
            "gap_start",
            "gap_end",
            "source_gap_start",
            "source_gap_end",
            "flags",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "start_frames_dir": str(out_dir.resolve()).replace("\\", "/"),
        "csv_path": str(csv_path.resolve()).replace("\\", "/"),
        "image_count": sum(1 for event in events if event.get("image_file")),
    }


def summarize_rally_start_events(
    events: list[dict[str, Any]],
    *,
    timeline_path: Path,
    events_json_path: Path,
    export_info: dict[str, Any],
) -> dict[str, Any]:
    scoring_count = sum(1 for event in events if event["kind"] == "scoring")
    let_count = sum(1 for event in events if event["kind"] == "let")
    gap_review_count = sum(1 for event in events if event["kind"] == "needs_review")
    rule_conflict_review_count = sum(
        1
        for event in events
        if bool(event.get("review_reason")) and event.get("kind") != "needs_review"
    )
    needs_review_count = gap_review_count + rule_conflict_review_count
    first_server = None
    if events:
        first = events[0]
        first_server = {
            "rally_id": first.get("id", ""),
            "starter_role": first.get("starter_role", ""),
            "server_initial_side": first.get("server_initial_side", ""),
            "server_player_key": first.get("server_player_key", ""),
            "server_player_name": first.get("server_player_name", "unknown"),
            "source": first.get("server_identity_source", ""),
            "current_side": first.get("current_side", "unknown"),
            "side_evidence_source": first.get("side_evidence_source", "unknown"),
            "side_evidence_status": first.get("side_evidence_status", "unknown"),
            "t_start": float(first.get("t_start", 0.0)),
            "source_t_start": float(first.get("source_t_start", first.get("t_start", 0.0))),
        }
    side_identified_count = sum(
        1 for event in events if event.get("side_evidence_status") == "identified"
    )
    side_inferred_count = sum(
        1 for event in events if event.get("side_evidence_status") == "inferred"
    )
    side_unknown_count = sum(
        1 for event in events if event.get("side_evidence_status") in {"unknown", "not_scanned"}
    )

    return {
        "algorithm": STEP3_1_ALGORITHM,
        "total": len(events),
        "detected_total": scoring_count + let_count,
        "scoring": scoring_count,
        "lets": let_count,
        "needs_review": needs_review_count,
        "rule_gap_review_count": gap_review_count,
        "rule_conflict_review_count": rule_conflict_review_count,
        "timeline_path": str(timeline_path.resolve()).replace("\\", "/"),
        "events_json_path": str(events_json_path.resolve()).replace("\\", "/"),
        "first_server": first_server,
        "side_identification": {
            "identified": side_identified_count,
            "inferred": side_inferred_count,
            "unknown": side_unknown_count,
        },
        **export_info,
        "events": events,
    }


def write_rally_start_events_json(
    events_json_path: Path,
    summary: dict[str, Any],
    events: list[dict[str, Any]],
) -> None:
    payload = {
        "summary": {k: v for k, v in summary.items() if k != "events"},
        "events": events,
    }
    events_json_path.parent.mkdir(parents=True, exist_ok=True)
    events_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_or_build_rally_timeline(
    *,
    video_path: str | Path,
    timeline_path: Path,
    legacy_cache_path: Path | None,
    table_weights_path: str,
    pose_weights_path: str,
    best_of: int,
    stride: int,
    mode: str,
    player_margin_px: int,
    player_fuse_gain: float,
    player_signal_source: str,
    ball_fuse_gain: float,
    ball_signal_source: str,
    table_roi,
    force_rebuild: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> RallyTimeline:
    if timeline_path.exists() and not force_rebuild:
        if log_fn:
            log_fn("Step 3.1: total rally start detection - reusing cached total-rally timeline")
        return load_rally_timeline(timeline_path)
    if legacy_cache_path is not None and legacy_cache_path.exists() and not force_rebuild:
        if log_fn:
            log_fn("Step 3.1: total rally start detection - migrating cached full-video detector output")
        timeline = load_rally_timeline(legacy_cache_path)
        save_rally_timeline(timeline_path, timeline)
        return timeline

    build_rally_timeline = _load_build_rally_timeline()
    if log_fn:
        log_fn("Step 3.1: total rally start detection - running existing start-time detector")
    timeline = build_rally_timeline(
        str(video_path),
        table_weights_path,
        pose_weights_path=pose_weights_path,
        best_of=best_of,
        stride=stride,
        mode=mode,
        player_margin_px=player_margin_px,
        player_fuse_gain=player_fuse_gain,
        player_signal_source=player_signal_source,
        ball_fuse_gain=ball_fuse_gain,
        ball_signal_source=ball_signal_source,
        table_roi=table_roi,
        log_fn=log_fn,
    )
    save_rally_timeline(timeline_path, timeline)
    return timeline


def build_step3_1_rally_start_review(
    *,
    video_path: str | Path,
    timeline_path: Path,
    events_json_path: Path,
    frame_dir: Path,
    table_roi,
    table_weights_path: str,
    pose_weights_path: str,
    best_of: int,
    stride: int,
    mode: str,
    player_margin_px: int,
    player_fuse_gain: float,
    player_signal_source: str,
    ball_fuse_gain: float,
    ball_signal_source: str,
    player_context: Step3PlayerContext | None,
    legacy_cache_path: Path | None = None,
    force_rebuild: bool = False,
    source_time_offset_sec: float = 0.0,
    enable_side_identification: bool = False,
    side_identification_video_path: str | Path | None = None,
    side_identification_time_field: str = "t_start",
    side_identification_end_time_field: str = "t_end",
    face_db_path: str | Path | None = None,
    face_model_path: str | Path | None = None,
    player_zone_xyxy: tuple[float, float, float, float] | None = None,
    side_identification_config: Step3SideIdentificationConfig | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> Step3RallyStartReviewResult:
    timeline = load_or_build_rally_timeline(
        video_path=video_path,
        timeline_path=timeline_path,
        legacy_cache_path=legacy_cache_path,
        table_weights_path=table_weights_path,
        pose_weights_path=pose_weights_path,
        best_of=best_of,
        stride=stride,
        mode=mode,
        player_margin_px=player_margin_px,
        player_fuse_gain=player_fuse_gain,
        player_signal_source=player_signal_source,
        ball_fuse_gain=ball_fuse_gain,
        ball_signal_source=ball_signal_source,
        table_roi=table_roi,
        force_rebuild=force_rebuild,
        log_fn=log_fn,
    )
    events = timeline_total_rally_start_events(
        timeline,
        player_context=player_context,
        source_time_offset_sec=source_time_offset_sec,
    )
    side_id_summary = {"enabled": False, "reason": "disabled"}
    if enable_side_identification:
        side_id_summary = annotate_events_with_single_player_side_identification(
            side_identification_video_path or video_path,
            events,
            player_context=player_context,
            face_db_path=face_db_path,
            face_model_path=face_model_path,
            pose_weights_path=pose_weights_path,
            table_roi=table_roi,
            player_zone_xyxy=player_zone_xyxy,
            config=side_identification_config,
            time_field=side_identification_time_field,
            end_time_field=side_identification_end_time_field,
            log_fn=log_fn,
        )
        # Refresh the serve-order expected names after per-rally side evidence
        # remaps the current server identity.
        annotate_serve_order_rule_reviews(events, player_context=player_context)
    export_info = export_rally_start_event_frames(
        video_path,
        events,
        table_roi=table_roi,
        out_dir=frame_dir,
    )
    summary = summarize_rally_start_events(
        events,
        timeline_path=timeline_path,
        events_json_path=events_json_path,
        export_info=export_info,
    )
    summary["side_identification"].update(side_id_summary)
    write_rally_start_events_json(events_json_path, summary, events)
    return Step3RallyStartReviewResult(timeline=timeline, events=events, summary=summary)


def build_step3_2_side_state_review(
    *,
    video_path: str | Path,
    source_events_json_path: Path,
    events_json_path: Path,
    frame_dir: Path,
    table_roi,
    pose_weights_path: str,
    player_context: Step3PlayerContext | None,
    face_db_path: str | Path | None = None,
    face_model_path: str | Path | None = None,
    player_zone_xyxy: tuple[float, float, float, float] | None = None,
    side_identification_config: Step3SideIdentificationConfig | None = None,
    time_field: str = "source_t_start",
    end_time_field: str = "source_t_end",
    log_fn: Callable[[str], None] | None = None,
) -> Step3SideStateReviewResult:
    """Step 3.2: attach per-rally NEAR/FAR side state to Step 3.1 events."""

    payload = json.loads(Path(source_events_json_path).read_text(encoding="utf-8"))
    events = [dict(event) for event in list(payload.get("events") or [])]
    for event in events:
        reset_side_evidence_fields(event, player_context=player_context)

    side_id_summary = annotate_events_with_single_player_side_identification(
        video_path,
        events,
        player_context=player_context,
        face_db_path=face_db_path,
        face_model_path=face_model_path,
        pose_weights_path=pose_weights_path,
        table_roi=table_roi,
        player_zone_xyxy=player_zone_xyxy,
        config=side_identification_config,
        time_field=time_field,
        end_time_field=end_time_field,
        log_fn=log_fn,
    )
    annotate_serve_order_rule_reviews(events, player_context=player_context)
    export_info = export_rally_start_event_frames(
        video_path,
        events,
        table_roi=table_roi,
        out_dir=frame_dir,
        frame_time_field=time_field,
    )
    summary = summarize_rally_start_events(
        events,
        timeline_path=source_events_json_path,
        events_json_path=events_json_path,
        export_info=export_info,
    )
    summary["algorithm"] = "step3_2_side_state_review_v1"
    summary["source_step3_1_events_json_path"] = str(Path(source_events_json_path).resolve()).replace("\\", "/")
    summary["side_identification"].update(side_id_summary)
    write_rally_start_events_json(events_json_path, summary, events)
    return Step3SideStateReviewResult(events=events, summary=summary)


def _logic_event_time(event: dict[str, Any], field: str) -> float:
    return float(event.get(field, event.get("t_start", 0.0)) or 0.0)


def _logic_event_end_time(event: dict[str, Any], field: str) -> float:
    return float(event.get(field, event.get("t_end", event.get("t_start", 0.0))) or 0.0)


def _logic_player_name(player_key: str, player_context: Step3PlayerContext | None) -> str:
    if player_context is None:
        return player_key or "unknown"
    return _player_name_for_key(player_key, player_context)


def _logic_other_player_key(player_key: str) -> str:
    return "player_b" if player_key == "player_a" else "player_a"


def _logic_expected_server_key(first_key: str, score_index: int, config: Step3LogicAuditConfig) -> str:
    other_key = _logic_other_player_key(first_key)
    deuce_after = int(config.deuce_switch_after_scoring_count)
    if deuce_after > 0 and score_index >= deuce_after:
        return first_key if ((score_index - deuce_after) % 2 == 0) else other_key
    return first_key if ((score_index // 2) % 2 == 0) else other_key


def _logic_server_key(event: dict[str, Any]) -> str:
    key = str(event.get("server_player_key", "") or "").strip()
    return key if key in {"player_a", "player_b"} else ""


def _logic_side_signature(event: dict[str, Any]) -> tuple[str, str] | None:
    a_side = str(event.get("player_a_current_side", "") or "").strip().upper()
    b_side = str(event.get("player_b_current_side", "") or "").strip().upper()
    if {a_side, b_side} != {"NEAR", "FAR"}:
        return None
    return (a_side, b_side)


def _logic_is_swapped_signature(prev_sig: tuple[str, str] | None, curr_sig: tuple[str, str] | None) -> bool:
    if prev_sig is None or curr_sig is None:
        return False
    return prev_sig[0] == _opposite_side(curr_sig[0]) and prev_sig[1] == _opposite_side(curr_sig[1])


def _logic_relevant_kind(event: dict[str, Any]) -> str:
    kind = str(event.get("kind", "") or "").strip()
    if kind in {"scoring", "needs_review", "let"}:
        return kind
    return ""


def _clear_step3_3_logic_fields(events: list[dict[str, Any]]) -> None:
    for event in events:
        for key in list(event.keys()):
            if key.startswith("step3_3_"):
                event.pop(key, None)


def _add_step3_3_issue(
    issues: list[dict[str, Any]],
    *,
    issue_type: str,
    message: str,
    event_ids: list[str],
    severity: str = "blocking",
    repair_route: str = "step3_2_side_rescan",
    segment_index: int | None = None,
    expected_server_key: str = "",
    actual_server_key: str = "",
    time_start: float | None = None,
    time_end: float | None = None,
) -> None:
    issue = {
        "id": f"logic_issue_{len(issues) + 1:04d}",
        "type": issue_type,
        "severity": severity,
        "repair_route": repair_route,
        "message": message,
        "event_ids": list(dict.fromkeys(event_ids)),
    }
    if segment_index is not None:
        issue["segment_index"] = int(segment_index)
    if expected_server_key:
        issue["expected_server_key"] = expected_server_key
    if actual_server_key:
        issue["actual_server_key"] = actual_server_key
    if time_start is not None:
        issue["time_start"] = float(time_start)
    if time_end is not None:
        issue["time_end"] = float(time_end)
    issues.append(issue)


def _step3_3_neighbor_event_ids(
    ordered: list[tuple[int, dict[str, Any]]],
    index_in_order: int,
    radius: int,
) -> list[str]:
    lo = max(0, int(index_in_order) - int(radius))
    hi = min(len(ordered), int(index_in_order) + int(radius) + 1)
    ids: list[str] = []
    for _idx, event in ordered[lo:hi]:
        event_id = str(event.get("id", "") or "")
        if event_id:
            ids.append(event_id)
    return ids


def _step3_3_segments(
    events: list[dict[str, Any]],
    *,
    config: Step3LogicAuditConfig,
    time_field: str,
    end_time_field: str,
) -> list[list[tuple[int, dict[str, Any]]]]:
    ordered = sorted(enumerate(events), key=lambda item: _logic_event_time(item[1], time_field))
    segments: list[list[tuple[int, dict[str, Any]]]] = [[]]
    scoring_since_boundary = 0
    prev_scoring_like: dict[str, Any] | None = None

    for item in ordered:
        _idx, event = item
        kind = _logic_relevant_kind(event)
        is_scoring_like = kind in {"scoring", "needs_review"}
        if is_scoring_like and prev_scoring_like is not None:
            gap = _logic_event_time(event, time_field) - _logic_event_end_time(prev_scoring_like, end_time_field)
            if (
                scoring_since_boundary >= int(config.min_scoring_before_set_boundary)
                and gap >= float(config.set_boundary_gap_sec)
                and _logic_is_swapped_signature(_logic_side_signature(prev_scoring_like), _logic_side_signature(event))
            ):
                event["step3_3_set_boundary_before"] = True
                event["step3_3_set_boundary_gap_sec"] = float(gap)
                segments.append([])
                scoring_since_boundary = 0

        segments[-1].append(item)
        if is_scoring_like:
            scoring_since_boundary += 1
            prev_scoring_like = event

    return [segment for segment in segments if segment]


def audit_step3_side_state_logic(
    events: list[dict[str, Any]],
    *,
    player_context: Step3PlayerContext | None,
    config: Step3LogicAuditConfig | None = None,
    time_field: str = "source_t_start",
    end_time_field: str = "source_t_end",
) -> dict[str, Any]:
    """Audit Step 3.2 output against basic table-tennis service rules."""

    config = config or Step3LogicAuditConfig()
    _clear_step3_3_logic_fields(events)
    issues: list[dict[str, Any]] = []
    rescan_event_ids: set[str] = set()
    step3_1_gap_rescan_event_ids: set[str] = set()
    segments = _step3_3_segments(events, config=config, time_field=time_field, end_time_field=end_time_field)

    for segment_index, segment in enumerate(segments, start=1):
        for _idx, event in segment:
            event["step3_3_segment_index"] = segment_index
            event["step3_3_logic_ok"] = True

        scoring_like = [
            (order_idx, event)
            for order_idx, (_idx, event) in enumerate(segment)
            if _logic_relevant_kind(event) in {"scoring", "needs_review"}
        ]
        first_server_key = ""
        for _order_idx, event in scoring_like:
            first_server_key = _logic_server_key(event)
            if first_server_key:
                break
        if not first_server_key:
            event_ids = [str(event.get("id", "") or "") for _idx, event in segment if str(event.get("id", "") or "")]
            _add_step3_3_issue(
                issues,
                issue_type="missing_server_identity_for_segment",
                message="Cannot audit serve order because this segment has no trusted server identity.",
                event_ids=event_ids,
                repair_route="step3_2_side_rescan",
                segment_index=segment_index,
            )
            rescan_event_ids.update(event_ids)
            continue

        score_index = 0
        for order_idx, (_idx, event) in enumerate(segment):
            kind = _logic_relevant_kind(event)
            if not kind:
                continue
            event_id = str(event.get("id", "") or "")
            actual_key = _logic_server_key(event)
            if str(event.get("side_evidence_status", "") or "") in {"unknown", "not_scanned"} or not actual_key:
                event["step3_3_logic_ok"] = False
                _add_step3_3_issue(
                    issues,
                    issue_type="unknown_current_server",
                    message="Current server is unknown; Step 3.2 side-state evidence must be rescanned.",
                    event_ids=[event_id],
                    repair_route="step3_2_side_rescan",
                    segment_index=segment_index,
                    time_start=_logic_event_time(event, time_field),
                    time_end=_logic_event_end_time(event, end_time_field),
                )
                if event_id:
                    rescan_event_ids.update(_step3_3_neighbor_event_ids(segment, order_idx, config.rescan_neighbor_radius))

            expected_key = _logic_expected_server_key(first_server_key, score_index, config)
            event["step3_3_expected_server_key"] = expected_key
            event["step3_3_expected_server_name"] = _logic_player_name(expected_key, player_context)
            event["step3_3_actual_server_key"] = actual_key
            event["step3_3_actual_server_name"] = event.get("server_player_name", "unknown")

            if actual_key and actual_key != expected_key:
                event["step3_3_logic_ok"] = False
                issue_type = "let_server_conflict" if kind == "let" else "serve_order_conflict"
                message = (
                    "LET server conflicts with the expected replay server."
                    if kind == "let"
                    else "Scoring server conflicts with the expected 2-serve table-tennis order."
                )
                _add_step3_3_issue(
                    issues,
                    issue_type=issue_type,
                    message=message,
                    event_ids=[event_id],
                    repair_route="step3_2_side_rescan",
                    segment_index=segment_index,
                    expected_server_key=expected_key,
                    actual_server_key=actual_key,
                    time_start=_logic_event_time(event, time_field),
                    time_end=_logic_event_end_time(event, end_time_field),
                )
                if event_id:
                    rescan_event_ids.update(_step3_3_neighbor_event_ids(segment, order_idx, config.rescan_neighbor_radius))

            if kind in {"scoring", "needs_review"}:
                score_index += 1

        scoring_events = [event for _order_idx, event in scoring_like]
        runs: list[tuple[str, int, int]] = []
        run_start = 0
        while run_start < len(scoring_events):
            key = _logic_server_key(scoring_events[run_start])
            run_end = run_start + 1
            while run_end < len(scoring_events) and _logic_server_key(scoring_events[run_end]) == key:
                run_end += 1
            runs.append((key, run_start, run_end))
            run_start = run_end

        for run_idx, (key, start, end) in enumerate(runs):
            if not key or start >= int(config.deuce_switch_after_scoring_count):
                continue
            run_len = end - start
            is_terminal = run_idx == 0 or run_idx == len(runs) - 1
            if run_len == 2 or (is_terminal and run_len == 1):
                continue
            run_event_ids = [
                str(event.get("id", "") or "")
                for event in scoring_events[start:end]
                if str(event.get("id", "") or "")
            ]
            if not run_event_ids:
                continue
            repair_route = "step3_2_side_rescan"
            if run_len == 1 and not is_terminal:
                repair_route = "step3_2_side_rescan_or_step3_1_gap_rescan"
                step3_1_gap_rescan_event_ids.update(run_event_ids)
            _add_step3_3_issue(
                issues,
                issue_type="serve_run_length_violation",
                message=(
                    f"Server run length is {run_len}; expected 2 before deuce "
                    "(terminal partial runs are allowed only at segment edges)."
                ),
                event_ids=run_event_ids,
                repair_route=repair_route,
                segment_index=segment_index,
            )
            for event_id in run_event_ids:
                rescan_event_ids.add(event_id)

    blocking_issues = [issue for issue in issues if issue.get("severity") == "blocking"]
    return {
        "algorithm": "step3_3_table_tennis_logic_audit_v1",
        "ok": not blocking_issues,
        "segments": len(segments),
        "issue_count": len(issues),
        "blocking_issue_count": len(blocking_issues),
        "issues": issues[: int(config.max_issues_in_summary)],
        "rescan_event_ids": sorted(rescan_event_ids),
        "requires_step3_1_gap_rescan_event_ids": sorted(step3_1_gap_rescan_event_ids),
        "config": {
            "max_repair_iterations": config.max_repair_iterations,
            "rescan_neighbor_radius": config.rescan_neighbor_radius,
            "set_boundary_gap_sec": config.set_boundary_gap_sec,
            "min_scoring_before_set_boundary": config.min_scoring_before_set_boundary,
            "deuce_switch_after_scoring_count": config.deuce_switch_after_scoring_count,
        },
    }


def _step3_3_repair_side_config(config: Step3SideIdentificationConfig | None) -> Step3SideIdentificationConfig:
    base = config or Step3SideIdentificationConfig()
    return replace(
        base,
        window_after_sec=max(float(base.window_after_sec), 6.0),
        sample_fps=max(float(base.sample_fps), 8.0),
        min_accepted_samples=max(int(base.min_accepted_samples), 4),
        retry_unknown_enabled=True,
        retry_window_after_sec=max(float(base.retry_window_after_sec), 12.0),
        retry_sample_fps=max(float(base.retry_sample_fps), 10.0),
        retry_min_accepted_samples=max(int(base.retry_min_accepted_samples), 3),
        continuity_fill_unknown_enabled=False,
    )


def build_step3_3_logic_audit_review(
    *,
    video_path: str | Path,
    source_events_json_path: Path,
    events_json_path: Path,
    table_roi,
    pose_weights_path: str,
    player_context: Step3PlayerContext | None,
    face_db_path: str | Path | None = None,
    face_model_path: str | Path | None = None,
    player_zone_xyxy: tuple[float, float, float, float] | None = None,
    side_identification_config: Step3SideIdentificationConfig | None = None,
    logic_audit_config: Step3LogicAuditConfig | None = None,
    time_field: str = "source_t_start",
    end_time_field: str = "source_t_end",
    log_fn: Callable[[str], None] | None = None,
) -> Step3LogicAuditResult:
    """Step 3.3: validate Step 3.2 and rescan suspicious side-state rows."""

    logic_config = logic_audit_config or Step3LogicAuditConfig()
    payload = json.loads(Path(source_events_json_path).read_text(encoding="utf-8"))
    source_summary = dict(payload.get("summary") or {})
    events = [dict(event) for event in list(payload.get("events") or [])]

    iterations: list[dict[str, Any]] = []
    repair_config = _step3_3_repair_side_config(side_identification_config)
    final_audit: dict[str, Any] = {}
    for iteration in range(int(logic_config.max_repair_iterations) + 1):
        audit = audit_step3_side_state_logic(
            events,
            player_context=player_context,
            config=logic_config,
            time_field=time_field,
            end_time_field=end_time_field,
        )
        iterations.append(
            {
                "iteration": iteration,
                "ok": bool(audit.get("ok")),
                "blocking_issue_count": int(audit.get("blocking_issue_count", 0)),
                "rescan_event_ids": list(audit.get("rescan_event_ids", []) or []),
                "requires_step3_1_gap_rescan_event_ids": list(
                    audit.get("requires_step3_1_gap_rescan_event_ids", []) or []
                ),
            }
        )
        final_audit = audit
        if bool(audit.get("ok")):
            break
        if iteration >= int(logic_config.max_repair_iterations):
            break
        rescan_ids = set(str(item) for item in audit.get("rescan_event_ids", []) or [])
        if not rescan_ids:
            break
        if log_fn:
            log_fn(
                "Step 3.3: logic audit - "
                f"iteration {iteration + 1} rescanning {len(rescan_ids)} suspicious side-state row(s)"
            )
        annotate_events_with_single_player_side_identification(
            video_path,
            events,
            player_context=player_context,
            face_db_path=face_db_path,
            face_model_path=face_model_path,
            pose_weights_path=pose_weights_path,
            table_roi=table_roi,
            player_zone_xyxy=player_zone_xyxy,
            config=repair_config,
            time_field=time_field,
            end_time_field=end_time_field,
            only_event_ids=rescan_ids,
            log_fn=log_fn,
        )

    export_info = {
        "start_frames_dir": source_summary.get("start_frames_dir", ""),
        "csv_path": source_summary.get("csv_path", ""),
        "image_count": source_summary.get("image_count", 0),
    }
    summary = summarize_rally_start_events(
        events,
        timeline_path=source_events_json_path,
        events_json_path=events_json_path,
        export_info=export_info,
    )
    summary["algorithm"] = "step3_3_logic_audit_v1"
    summary["source_step3_2_events_json_path"] = str(Path(source_events_json_path).resolve()).replace("\\", "/")
    summary["logic_audit"] = final_audit
    summary["logic_audit_iterations"] = iterations
    summary["logic_ok"] = bool(final_audit.get("ok", False))
    summary["logic_blocking_issue_count"] = int(final_audit.get("blocking_issue_count", 0) or 0)
    write_rally_start_events_json(events_json_path, summary, events)
    return Step3LogicAuditResult(events=events, summary=summary)
