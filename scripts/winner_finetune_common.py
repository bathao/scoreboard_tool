from __future__ import annotations

import json
import re
from pathlib import Path


CANONICAL_TAXONOMY_DEFINITIONS: dict[str, str] = {
    "clean_winner_no_touch": "the winner hits a legal shot and the loser never touches the final ball",
    "touched_but_out": "the loser touches the final ball but sends the return out",
    "touched_but_no_net_cross": "the loser touches the final ball but the return does not cross the net",
    "attacker_direct_out": "the attacking player makes the final shot and sends it directly out",
    "attacker_into_net": "the attacking player makes the final shot into the net or fails to cross the net",
    "double_bounce_before_return": "the loser allows a second bounce before making a legal return",
    "ball_hits_player_or_body": "the ball hits the losing player's body",
    "ball_hits_non_racket_object": "the ball hits a non-racket object on the losing side",
    "illegal_or_mishit_return": "the final losing return is clearly illegal or mishit",
    "blocked_by_visibility": "the decisive contact is hidden or too unclear in the video",
    "ambiguous_review": "the evidence remains unclear",
}

ACTIVE_PILOT_TAXONOMY_ORDER = [
    "clean_winner_no_touch",
    "touched_but_out",
    "touched_but_no_net_cross",
    "attacker_direct_out",
    "attacker_into_net",
]


def load_manifest_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = str(line).strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def resolve_clip_path(dataset_root: Path, row: dict[str, object]) -> Path:
    clip_relpath = Path(str(row["clip_relpath"]))
    return (dataset_root / clip_relpath).resolve()


def safe_sample_stem(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_id).strip())


def resolve_cached_clip_path(cache_dir: Path, row: dict[str, object]) -> Path:
    return cache_dir / f"{safe_sample_stem(str(row['sample_id']))}.mp4"


def observed_taxonomies(rows: list[dict[str, object]]) -> list[str]:
    present = {str(row.get("taxonomy", "")).strip() for row in rows if str(row.get("taxonomy", "")).strip()}
    ordered = [label for label in ACTIVE_PILOT_TAXONOMY_ORDER if label in present]
    extras = sorted(present.difference(ordered))
    return ordered + extras


def build_training_prompt(active_taxonomies: list[str]) -> str:
    definitions = " ".join(
        f"{label} = {CANONICAL_TAXONOMY_DEFINITIONS.get(label, label)}."
        for label in active_taxonomies
    )
    allowed = ", ".join(active_taxonomies)
    return (
        "Analyze this full table-tennis rally video. "
        "Player A is the near-side player. Player B is the far-side player. "
        "Choose exactly one final losing-event taxonomy from: "
        f"{allowed}. "
        f"Definitions: {definitions} "
        "Return strict JSON only with these keys: "
        '{"winner":"player_a|player_b","loser":"player_a|player_b","taxonomy":"one allowed taxonomy","last_hitter":"player_a|player_b"}.'
    )


def build_target_json(row: dict[str, object]) -> str:
    payload = {
        "winner": str(row["winner"]),
        "loser": str(row["loser"]),
        "taxonomy": str(row["taxonomy"]),
        "last_hitter": str(row["last_hitter"]),
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def parse_prediction_json(text: str) -> dict[str, str]:
    stripped = str(text).strip()
    if not stripped:
        return {}
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    candidate = match.group(0) if match else stripped
    try:
        loaded = json.loads(candidate)
    except json.JSONDecodeError:
        loaded = {}
    if isinstance(loaded, dict):
        return {str(k): str(v) for k, v in loaded.items()}
    return {}
