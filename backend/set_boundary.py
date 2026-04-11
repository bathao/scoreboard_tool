"""Set boundary detection using multiple independent signals.

Signal 1 — Score rule:
    ScoreEngine determines a set ends when one player reaches 11+ points with a 2-point
    lead (or deuce). Reliable only when winner assignments are mostly correct.

Signal 2 — Inter-rally gap:
    The gap between rally[i].t_end and rally[i+1].t_start.
    Within-set gaps are typically < 15 seconds.
    Between-set breaks (water, coaching) are typically 60–120 seconds.
    This signal is independent of winner correctness.

Signal 3 — YOLO player X-position:
    Compare mean player X positions in the last rally of set N vs the first rally of
    set N+1. If players have swapped sides, this confirms a real set boundary.
    Requires near_mean_x / far_mean_x populated in RallyTimelinePoint.
    (Infrastructure deferred — fields exist in the schema, populated by future pipeline step.)

Usage:
    set_nums = assign_set_numbers(timeline, best_of=5)
    for point, sn in zip(timeline.points, set_nums):
        point.set_number = sn
"""
from __future__ import annotations

from typing import List, Optional

from backend.rally_timeline_contract import RallyTimeline, RallyTimelinePoint, counts_toward_score

# Default threshold: gaps longer than this (in seconds) are set boundary candidates.
# Between-set breaks in table tennis are typically 60-120s.
GAP_THRESHOLD_SECONDS: float = 60.0

# For Signal 3: minimum absolute difference in mean X to call a side swap.
# Expressed as a fraction of frame width (0.0-1.0). Placeholder until calibrated.
SIDE_SWAP_X_THRESHOLD: float = 0.15


# ---------------------------------------------------------------------------
# Signal 2 — inter-rally gap
# ---------------------------------------------------------------------------

def detect_boundaries_by_gap(
    timeline: RallyTimeline,
    min_gap_sec: float = GAP_THRESHOLD_SECONDS,
) -> List[int]:
    """Return rally indices where a new set likely starts, based on inter-rally gap.

    Index i means timeline.points[i] is the first rally of a new set.
    """
    candidates: List[int] = []
    pts = timeline.points
    for i in range(len(pts) - 1):
        gap = pts[i + 1].t_start - pts[i].t_end
        if gap >= min_gap_sec:
            candidates.append(i + 1)
    return candidates


def get_inter_rally_gaps(timeline: RallyTimeline) -> List[float]:
    """Return list of gaps in seconds between consecutive rallies (len = n_points - 1)."""
    pts = timeline.points
    return [pts[i + 1].t_start - pts[i].t_end for i in range(len(pts) - 1)]


# ---------------------------------------------------------------------------
# Signal 1 — score rule
# ---------------------------------------------------------------------------

def detect_boundaries_by_score(
    timeline: RallyTimeline,
    best_of: int = 5,
) -> List[int]:
    """Return rally indices where a new set starts, based on score accumulation.

    Uses the effective winner of each scoring rally (human correction > AI candidate).
    Skips unknown winners. Less reliable when many winners are unresolved.
    """
    from backend.production_jobs import effective_rally_winner

    candidates: List[int] = []
    score_a, score_b = 0, 0
    sets_a, sets_b = 0, 0
    sets_needed = (best_of + 1) // 2

    for i, point in enumerate(timeline.points):
        if not counts_toward_score(point):
            continue
        winner = effective_rally_winner(point)
        if winner == "player_a":
            score_a += 1
        elif winner == "player_b":
            score_b += 1
        else:
            continue  # unknown — cannot advance score

        a_wins_set = score_a >= 11 and (score_a - score_b) >= 2
        b_wins_set = score_b >= 11 and (score_b - score_a) >= 2
        if a_wins_set or b_wins_set:
            if a_wins_set:
                sets_a += 1
            else:
                sets_b += 1
            score_a, score_b = 0, 0
            if i + 1 < len(timeline.points):
                # Match is not over: next rally starts a new set
                if sets_a < sets_needed and sets_b < sets_needed:
                    candidates.append(i + 1)

    return candidates


# ---------------------------------------------------------------------------
# Signal 3 — YOLO side-swap (near_mean_x / far_mean_x)
# ---------------------------------------------------------------------------

def detect_boundaries_by_position(
    timeline: RallyTimeline,
    x_threshold: float = SIDE_SWAP_X_THRESHOLD,
) -> List[int]:
    """Return rally indices where players appear to have swapped sides.

    Compares near_mean_x of rally[i] vs rally[i+1]. A large shift suggests a
    side swap (set boundary or mid-deciding-set swap).

    Returns empty list if near_mean_x is not populated in the timeline.
    """
    candidates: List[int] = []
    pts = timeline.points
    for i in range(len(pts) - 1):
        nx_before = pts[i].near_mean_x
        nx_after = pts[i + 1].near_mean_x
        if nx_before is None or nx_after is None:
            continue
        if abs(nx_after - nx_before) >= x_threshold:
            candidates.append(i + 1)
    return candidates


# ---------------------------------------------------------------------------
# Cross-validation and assignment
# ---------------------------------------------------------------------------

def cross_validate(
    gap_candidates: List[int],
    score_candidates: List[int],
    position_candidates: Optional[List[int]] = None,
    tolerance: int = 3,
) -> List[dict]:
    """Annotate each detected boundary with which signals agree.

    tolerance: accept score/position candidate as matching a gap candidate if
    within this many rally indices.
    """
    all_indices = sorted(set(gap_candidates) | set(score_candidates) | set(position_candidates or []))
    results = []
    for idx in all_indices:
        near_gap = any(abs(idx - g) <= tolerance for g in gap_candidates)
        near_score = any(abs(idx - s) <= tolerance for s in score_candidates)
        near_pos = any(abs(idx - p) <= tolerance for p in (position_candidates or []))
        signal_count = sum([near_gap, near_score, near_pos])
        confidence = "high" if signal_count >= 2 else "low"
        results.append(
            {
                "rally_index": idx,
                "gap_signal": near_gap,
                "score_signal": near_score,
                "position_signal": near_pos,
                "signal_count": signal_count,
                "confidence": confidence,
            }
        )
    return results


def assign_set_numbers(
    timeline: RallyTimeline,
    best_of: int = 5,
    min_gap_sec: float = GAP_THRESHOLD_SECONDS,
) -> List[int]:
    """Compute set_number (1-indexed) for each rally in timeline.

    Strategy:
    - Gap signal (Signal 2) is the primary independent signal — it doesn't depend
      on winner correctness.
    - Score signal (Signal 1) confirms or refines gap-detected boundaries. Also adds
      boundaries in cases where the gap is just under threshold (e.g. quick between-set).
    - Position signal (Signal 3) used when available; ignored otherwise.

    A boundary is accepted if:
    - Gap signal fires (most reliable), OR
    - Score signal fires without a contradicting gap (score confirms a boundary that
      gap missed, plausible for fast transitions)

    Returns a list of len(timeline.points) ints, each the 1-indexed set number.
    """
    gap_candidates = detect_boundaries_by_gap(timeline, min_gap_sec)
    score_candidates = detect_boundaries_by_score(timeline, best_of)
    position_candidates = detect_boundaries_by_position(timeline)

    # Accept: gap-detected, OR score-detected with no nearby contradiction from gap
    accepted: set[int] = set(gap_candidates)
    for sc in score_candidates:
        # If score fires but no gap is anywhere nearby (within 5 rallies), still accept
        near_gap_exists = any(abs(sc - g) <= 5 for g in gap_candidates)
        if not near_gap_exists:
            accepted.add(sc)
        # If score fires and there IS a gap nearby but at a slightly different index,
        # prefer the gap index (already added above)

    # Refine: if position signal fires near a gap boundary, confirm it (already accepted)
    # If position fires with NO gap/score nearby: caution — could be mid-set swap. Skip.

    boundary_set = sorted(accepted)

    set_numbers: List[int] = []
    current_set = 1
    for i in range(len(timeline.points)):
        if i in boundary_set:
            current_set = min(current_set + 1, best_of)  # cap at best_of
        set_numbers.append(current_set)

    return set_numbers


def apply_set_numbers(timeline: RallyTimeline, best_of: int = 5, min_gap_sec: float = GAP_THRESHOLD_SECONDS) -> RallyTimeline:
    """Assign set_number to each point in-place and return the timeline."""
    set_nums = assign_set_numbers(timeline, best_of=best_of, min_gap_sec=min_gap_sec)
    for point, sn in zip(timeline.points, set_nums):
        point.set_number = sn
    return timeline
