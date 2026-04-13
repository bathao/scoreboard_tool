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
    Requires player_a_mean_x / player_b_mean_x populated in RallyTimelinePoint.
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

# For Signal 3: minimum jump in near-player X between consecutive rallies (as fraction of
# estimated frame width) to call a side swap.  A jump > 20% of frame width indicates
# the "near" (largest-bounding-box) player crossed to the other side of the frame.
SWAP_X_JUMP_FRACTION: float = 0.20

# Pre-boundary stability filter: the region BEFORE a boundary candidate must be
# consistently on one side (near_x range < this fraction of frame width).
# Filters out false positives from sets where both players appear similar in size
# (near player assignment flips randomly, causing large but spurious jumps).
PRE_STABILITY_FRACTION: float = 0.13  # ~370px for a 2688px frame
PRE_STABILITY_WINDOW: int = 4   # how many rallies to check
PRE_STABILITY_SKIP: int = 2     # skip the N rallies closest to the candidate (may be transitional)


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
# Signal 3 — YOLO side-swap (player_a_mean_x / player_b_mean_x)
# ---------------------------------------------------------------------------

def populate_player_positions(
    timeline: RallyTimeline,
    video_path: str,
    pose_weights_path: str,
    samples_per_rally: int = 4,
) -> None:
    """Fill player_a_mean_x / player_b_mean_x on each RallyTimelinePoint in-place.

    Samples a few frames per rally, runs YOLO pose inference to locate the two
    players, then tracks player identity across rallies using positional continuity
    (EMA smoothing on expected X).

    player_a is initialised as the LEFT-side player in the first rally.
    After a side swap, player_a_mean_x will shift from ~left to ~right of frame.
    This crossing of the frame midline is the set-boundary signal.

    The function does NOT re-run the full tracking pipeline — it is a lightweight
    post-hoc pass (~seconds on GPU for a full match).
    """
    import cv2
    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        return  # gracefully skip if ultralytics not available

    model = YOLO(pose_weights_path)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return

    # Area-based assignment (no EMA identity tracking):
    # player_a_mean_x = mean X of the LARGEST bounding box (near player — closest to camera)
    # player_b_mean_x = mean X of the SECOND-LARGEST bounding box (far player)
    #
    # After a side swap, the near player moves to the other side of the frame, causing
    # player_a_mean_x to jump dramatically (e.g. left→right or right→left).
    # This is more reliable than EMA identity tracking, which fails at the swap point by
    # always following the left-side player regardless of which physical player is there.

    for point in timeline.points:
        duration = max(0.2, point.t_end - point.t_start)
        n = min(samples_per_rally, max(1, int(duration * 2)))
        sample_times = [point.t_start + duration * (k + 0.5) / n for k in range(n)]

        near_xs: list[float] = []
        far_xs: list[float] = []

        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if not ret:
                continue
            results = model.predict(frame, verbose=False, half=True)
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                continue
            boxes = results[0].boxes.xyxy.cpu().numpy()
            if len(boxes) < 2:
                continue
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            top2 = sorted(range(len(boxes)), key=lambda i: -areas[i])[:2]
            # top2[0] = largest area = near player; top2[1] = second-largest = far player
            near_cx = (boxes[top2[0]][0] + boxes[top2[0]][2]) / 2.0
            far_cx = (boxes[top2[1]][0] + boxes[top2[1]][2]) / 2.0
            near_xs.append(near_cx)
            far_xs.append(far_cx)

        if not near_xs:
            continue

        point.player_a_mean_x = float(sum(near_xs) / len(near_xs))
        point.player_b_mean_x = float(sum(far_xs) / len(far_xs))

    cap.release()


def detect_boundaries_by_position(
    timeline: RallyTimeline,
    x_jump_fraction: float = SWAP_X_JUMP_FRACTION,
) -> List[int]:
    """Return rally indices where the near player (largest bounding box) jumped sides.

    player_a_mean_x is populated by populate_player_positions() as the mean X of the
    LARGEST bounding box per sample frame (= the player nearest to the camera).

    After a side swap, the near player moves to the other side of the frame, so
    player_a_mean_x jumps by a large fraction of the estimated frame width.

    Algorithm:
    1. For each consecutive rally pair: compute |ax_after - ax_before|.
    2. If jump > x_jump_fraction * frame_w_est → raw boundary candidate.
    3. Pre-boundary stability filter: the region BEFORE the candidate (PRE_STABILITY_WINDOW
       rallies, skipping the PRE_STABILITY_SKIP closest to the boundary) must have
       near_x range < PRE_STABILITY_FRACTION * frame_w_est.
       This filters out false positives from sets where both players appear similar in
       size (near player assignment flips randomly → large but spurious jumps).
    4. Consecutive candidates that remain are merged: use the later index.

    Returns empty list if player_a_mean_x is not populated.
    """
    pts = timeline.points
    # Estimate frame width from all observed X positions (near + far players)
    all_xs = [p.player_a_mean_x for p in pts if p.player_a_mean_x is not None]
    all_xs += [p.player_b_mean_x for p in pts if p.player_b_mean_x is not None]
    if not all_xs:
        return []
    frame_w_est = max(all_xs) * 1.15
    jump_threshold = frame_w_est * x_jump_fraction
    stability_threshold = frame_w_est * PRE_STABILITY_FRACTION

    def _pre_stable(i: int, last_boundary: int) -> bool:
        """True if near_x is stable in the window just before rally i.

        Window is bounded below by last_boundary (we only look within the current set).
        Skips the PRE_STABILITY_SKIP rallies closest to the candidate (may be transitional).
        Requires at least 2 data points; returns False (reject) if insufficient data.
        """
        end = max(last_boundary, i - PRE_STABILITY_SKIP)
        start = max(last_boundary, i - PRE_STABILITY_SKIP - PRE_STABILITY_WINDOW)
        if end <= start:
            return False  # insufficient data within this set → reject
        xs = [pts[j].player_a_mean_x for j in range(start, end) if pts[j].player_a_mean_x is not None]
        if len(xs) < 2:
            return False  # insufficient data → reject
        return (max(xs) - min(xs)) < stability_threshold

    # Sequential: evaluate each raw candidate using the most recent accepted boundary
    accepted: List[int] = []
    last_boundary = 0
    for i in range(len(pts) - 1):
        ax_before = pts[i].player_a_mean_x
        ax_after = pts[i + 1].player_a_mean_x
        if ax_before is None or ax_after is None:
            continue
        if abs(ax_after - ax_before) > jump_threshold and _pre_stable(i, last_boundary):
            accepted.append(i + 1)
            last_boundary = i + 1  # next check is relative to this new boundary

    # Merge consecutive accepted boundaries: [i, i+1] → keep i+1
    merged: List[int] = []
    j = 0
    while j < len(accepted):
        if j + 1 < len(accepted) and accepted[j + 1] == accepted[j] + 1:
            merged.append(accepted[j + 1])
            j += 2
        else:
            merged.append(accepted[j])
            j += 1
    return merged


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

    Priority:
    - Signal 3 (side swap via YOLO player X): PRIMARY when available.
      Players ALWAYS swap sides at each set boundary — this is a guaranteed rule.
      Accept all Signal 3 candidates.
    - Signal 1 (score rule): Accept score candidates that have no nearby Signal 3
      (to catch any missed position detections). Also used when Signal 3 not available.
    - Signal 2 (inter-rally gap): Supporting evidence only.
      Gap is NOT required between sets (can be as short as 10-20s).
      Used as tiebreaker when Signal 3 is unavailable and score is also absent.

    Returns a list of len(timeline.points) ints, each the 1-indexed set number.
    """
    gap_candidates = detect_boundaries_by_gap(timeline, min_gap_sec)
    score_candidates = detect_boundaries_by_score(timeline, best_of)
    position_candidates = detect_boundaries_by_position(timeline)

    # Check if player positions were populated at all (Signal 3 data available)
    has_position_data = any(p.player_a_mean_x is not None for p in timeline.points)

    accepted: set[int] = set()

    if has_position_data:
        # Signal 3 is available: use it as primary source of truth
        accepted.update(position_candidates)
        # Also accept score candidates with no nearby position signal
        # (catches set boundaries where position detection failed)
        for sc in score_candidates:
            near_pos = any(abs(sc - p) <= 5 for p in position_candidates)
            if not near_pos:
                accepted.add(sc)
    else:
        # Signal 3 unavailable: fall back to score + gap
        accepted.update(score_candidates)
        for g in gap_candidates:
            near_score = any(abs(g - sc) <= 5 for sc in score_candidates)
            if not near_score:
                accepted.add(g)

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
