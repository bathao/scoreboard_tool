from __future__ import annotations

from typing import Any, Dict, List

from backend.ai_contract import DraftMatch, to_core_rally_events
from backend.engine import ScoreEngine
from backend.models import MatchState


def _set_to_dict(s) -> Dict[str, Any]:
    return {
        "player_a": int(s.player_a),
        "player_b": int(s.player_b),
        "winner": s.winner,
    }


def build_score_validation(
    draft: DraftMatch,
    *,
    expected_scope: str = "any",  # any | set | match
    expected_final_set_score: str | None = None,  # e.g. "11-3"
) -> Dict[str, Any]:
    if expected_scope not in {"any", "set", "match"}:
        raise ValueError(f"Invalid expected_scope: {expected_scope}")
    if expected_final_set_score:
        try:
            a_str, b_str = expected_final_set_score.split("-")
            expected_a = int(a_str.strip())
            expected_b = int(b_str.strip())
        except Exception as e:
            raise ValueError(
                f"Invalid expected_final_set_score '{expected_final_set_score}'. Use format A-B, e.g. 11-3."
            ) from e
    else:
        expected_a = None
        expected_b = None

    unknown_count = sum(1 for p in draft.points if p.winner == "unknown")
    events = to_core_rally_events(draft)

    engine = ScoreEngine(MatchState(best_of=draft.best_of))
    snapshots = []
    for e in events:
        snapshots.append(engine.process_event(e))

    match = engine.match
    completed_sets = [s for s in match.sets if s.winner is not None]
    open_set = match.sets[-1] if match.sets and match.sets[-1].winner is None else None

    issues: List[str] = []
    if len(events) == 0:
        issues.append("No scored events available for validation.")
    if unknown_count > 0:
        issues.append(f"Unknown winners remaining: {unknown_count}.")

    if expected_scope == "set":
        if len(completed_sets) < 1:
            issues.append("Expected full set clip, but no completed set was detected.")
    elif expected_scope == "match":
        if not match.is_finished:
            issues.append("Expected full match clip, but match is not finished.")

    unmet_scope = (
        (expected_scope == "set" and len(completed_sets) < 1)
        or (expected_scope == "match" and not match.is_finished)
    )

    last_completed_set = completed_sets[-1] if completed_sets else None
    inferred_scoreline = None
    final_set_a = None
    final_set_b = None
    if last_completed_set is not None:
        final_set_a = int(last_completed_set.player_a)
        final_set_b = int(last_completed_set.player_b)
        inferred_scoreline = f"{last_completed_set.player_a}-{last_completed_set.player_b}"
    elif open_set is not None:
        final_set_a = int(open_set.player_a)
        final_set_b = int(open_set.player_b)
        inferred_scoreline = f"{open_set.player_a}-{open_set.player_b}"

    if expected_a is not None and expected_b is not None:
        if final_set_a is None or final_set_b is None:
            issues.append("Expected final set score provided, but no set score could be inferred.")
        elif final_set_a != expected_a or final_set_b != expected_b:
            issues.append(
                f"Final set score mismatch: expected {expected_a}-{expected_b}, got {final_set_a}-{final_set_b}."
            )

    status = "pass"
    if issues:
        status = "warn"
    if unmet_scope:
        status = "fail"
    if expected_a is not None and expected_b is not None:
        if final_set_a is None or final_set_b is None or final_set_a != expected_a or final_set_b != expected_b:
            status = "fail"

    return {
        "status": status,
        "expected_scope": expected_scope,
        "best_of": int(draft.best_of),
        "event_count": len(draft.points),
        "known_winner_count": len(events),
        "unknown_winner_count": unknown_count,
        "is_match_finished": bool(match.is_finished),
        "match_winner": match.winner,
        "completed_sets": [_set_to_dict(s) for s in completed_sets],
        "open_set": _set_to_dict(open_set) if open_set is not None else None,
        "inferred_scoreline": inferred_scoreline,
        "expected_final_set_score": expected_final_set_score,
        "issues": issues,
    }
