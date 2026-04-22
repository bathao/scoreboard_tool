"""Detect the side-swap timestamp in a table tennis video.

Algorithm (independent of rally detection):
  1. Detect Table ROI and derive player zone.
  2. Sample frames at fixed step through the video.
  3. Per frame: detect bodies inside player zone, classify each as Side L
     (cx < table_center_x) or Side R, identify via face DB.
  4. Build per-identity timeline of (t, side).
  5. Smooth the side per identity over a sliding window.
  6. Establish baseline state (Set 1 sides) from the early window.
  7. Walk forward to find the earliest timestamp T* where BOTH players have
     swapped to the opposite side AND the new state is stable for >= 15 s.
  8. Backtrack from T* to find the earliest moment the swapped state began.

Output: T_swap_start (timestamp where Set 2 begins) plus a summary of how
the input timeline splits into Set 1 / Set 2 ranges.

Usage:
    python scripts/detect_side_swap.py --video inputs/raw_matches/2_sets.mp4
    python scripts/detect_side_swap.py --video inputs/raw_matches/2_sets.mp4 --sample-step 1.5 --baseline-end 60
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

from backend.player_identity import FaceDB, FaceEmbedder, face_similarity
from backend.player_identification import (
    DEFAULT_MATCH_THRESHOLD,
    detect_table_roi_and_player_zone,
    _detect_bodies_and_faces,
    _try_embed_face,
)
from backend.production_pipeline import ProductionPipelineConfig


# --- side classification --------------------------------------------------

SIDE_L = "L"
SIDE_R = "R"
SIDE_UNK = "?"


def classify_side(cx: float, table_center_x: float) -> str:
    return SIDE_L if cx < table_center_x else SIDE_R


# --- per-frame sampling ---------------------------------------------------

def sample_positions(
    video_path: str,
    yolo,
    embedder: FaceEmbedder,
    face_db: FaceDB,
    player_zone: tuple,
    table_center_x: float,
    sample_step: float,
    match_threshold: float,
    log_fn=None,
) -> list[dict]:
    """Walk the video at sample_step intervals and return per-detection records.

    Each record:
        {
            "t": float seconds,
            "identity": str | None,   player name from face DB or None
            "side": "L" | "R",
            "sim": float | None,      cosine similarity to matched record
            "rank": int,              0 = largest bbox, 1 = second largest
            "face_score": float,
        }
    """
    def _log(msg):
        if log_fn:
            log_fn(msg)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n_frames / fps if fps > 0 else 0.0
    _log(f"video: fps={fps:.2f} frames={n_frames} duration={duration:.1f}s")

    records: list[dict] = []
    t = 0.0
    while t <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if not ret:
            break
        dets = _detect_bodies_and_faces(frame, yolo, roi_xyxy=player_zone)
        for rank, det in enumerate(dets[:2]):
            cx = float((det["bbox_xyxy"][0] + det["bbox_xyxy"][2]) / 2.0)
            side = classify_side(cx, table_center_x)
            embed_result = _try_embed_face(frame, det, embedder)
            if embed_result is None:
                # Body present but no usable face this frame.
                records.append({
                    "t": t, "identity": None, "side": side,
                    "sim": None, "rank": rank, "face_score": 0.0,
                })
                continue
            emb, _disp, face_score = embed_result
            best_record = None
            best_sim = -1.0
            for rec in face_db.records:
                sim = face_similarity(emb, rec.embedding_array())
                if sim > best_sim:
                    best_sim = sim
                    best_record = rec
            identity = best_record.name if best_record is not None and best_sim >= match_threshold else None
            records.append({
                "t": t, "identity": identity, "side": side,
                "sim": float(best_sim), "rank": rank, "face_score": float(face_score),
            })
        t += sample_step
    cap.release()
    return records


# --- smoothing / state detection -----------------------------------------

def smoothed_side(timeline_for_player: list[tuple[float, str]],
                  t: float,
                  window: float = 10.0,
                  min_samples: int = 2,
                  min_majority_frac: float = 0.6) -> str | None:
    """Return the dominant side (L/R) in [t-window, t+window] for one player.

    Returns None when there are too few samples or when the majority is not
    strong enough (likely transitioning).
    """
    nearby = [s for (ts, s) in timeline_for_player if abs(ts - t) <= window]
    if len(nearby) < min_samples:
        return None
    counts = Counter(nearby)
    dominant_side, dominant_count = counts.most_common(1)[0]
    if dominant_count / len(nearby) < min_majority_frac:
        return None
    return dominant_side


def baseline_state(timeline_for_player: list[tuple[float, str]],
                   t_start: float, t_end: float) -> str | None:
    """Return the dominant side observed in [t_start, t_end]."""
    nearby = [s for (ts, s) in timeline_for_player if t_start <= ts <= t_end]
    if not nearby:
        return None
    counts = Counter(nearby)
    return counts.most_common(1)[0][0]


def _flipped_or_consistent(side: str | None, init: str, flipped: str) -> bool:
    """True if side is None (no info) or matches the flipped state.
    Used to allow the partner-symmetry inference: when one player has clearly
    flipped and the other has no contrary evidence, treat as a swap."""
    return side is None or side == flipped


def find_swap(tl_a: list[tuple[float, str]],
              tl_b: list[tuple[float, str]],
              search_start: float,
              search_end: float,
              step: float,
              init_a: str,
              init_b: str,
              stability_seconds: float,
              window: float) -> tuple[float, str] | None:
    """Walk forward from search_start; return (t, mode) at the earliest moment
    one or both players have flipped side AND state remains stable for
    >= stability_seconds.

    mode = "both"   when both players' smoothed sides are observed flipped
    mode = "a-only" when only player A has observed flipped state and B has
                    no contrary evidence (symmetry inference)
    mode = "b-only" mirror of the above
    """
    flipped_a = SIDE_R if init_a == SIDE_L else SIDE_L
    flipped_b = SIDE_R if init_b == SIDE_L else SIDE_L

    t = search_start
    while t <= search_end:
        sa = smoothed_side(tl_a, t, window=window)
        sb = smoothed_side(tl_b, t, window=window)

        a_flipped = sa == flipped_a
        b_flipped = sb == flipped_b

        candidate_mode = None
        if a_flipped and b_flipped:
            candidate_mode = "both"
        elif a_flipped and _flipped_or_consistent(sb, init_b, flipped_b):
            candidate_mode = "a-only"
        elif b_flipped and _flipped_or_consistent(sa, init_a, flipped_a):
            candidate_mode = "b-only"

        if candidate_mode is not None:
            stable = True
            tt = t
            while tt < t + stability_seconds:
                tt += step
                ts_check = smoothed_side(tl_a, tt, window=window)
                vs_check = smoothed_side(tl_b, tt, window=window)
                # Reject only if we observe an explicit return to baseline.
                if (ts_check is not None and ts_check == init_a) or \
                   (vs_check is not None and vs_check == init_b):
                    stable = False
                    break
            if stable:
                return t, candidate_mode
        t += step
    return None


def _compute_table_energy_window(
    video_path: str,
    table_roi,
    t_start: float,
    t_end: float,
    dense_step: float = 0.5,
):
    """Sample frames densely between t_start..t_end; for each consecutive pair,
    compute mean absolute pixel difference inside the table ROI.  Returns a
    list of (t, energy) tuples (one per pair, indexed by the LATER frame's t).

    High energy = rally activity in table region (ball + arms in motion).
    Low energy = nothing happening at the table (set break / inter-rally idle).
    """
    import cv2 as _cv2
    cap = _cv2.VideoCapture(str(video_path))
    samples: list[tuple[float, float]] = []
    prev_gray = None
    x, y, w, h = int(table_roi.x), int(table_roi.y), int(table_roi.w), int(table_roi.h)
    t = t_start
    while t <= t_end:
        cap.set(_cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            t += dense_step
            continue
        gray = _cv2.cvtColor(crop, _cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and prev_gray.shape == gray.shape:
            diff = _cv2.absdiff(gray, prev_gray)
            energy = float(np.mean(diff))
            samples.append((t, energy))
        prev_gray = gray
        t += dense_step
    cap.release()
    return samples


def dominant_side_in_range(
    timeline_for_player: list[tuple[float, str]],
    t_start: float,
    t_end: float,
    *,
    min_samples: int = 1,
    min_majority_frac: float = 0.60,
) -> dict:
    """Return dominant side evidence for one trusted player in a time range."""
    samples = [s for (ts, s) in timeline_for_player if t_start <= ts <= t_end]
    if len(samples) < min_samples:
        return {"side": None, "samples": len(samples), "majority_frac": 0.0}
    counts = Counter(samples)
    side, count = counts.most_common(1)[0]
    frac = float(count / max(1, len(samples)))
    if frac < min_majority_frac:
        return {"side": None, "samples": len(samples), "majority_frac": frac}
    return {"side": side, "samples": len(samples), "majority_frac": frac}


def infer_opposite_side(side: str | None) -> str | None:
    if side == SIDE_L:
        return SIDE_R
    if side == SIDE_R:
        return SIDE_L
    return None


def _video_duration_sec(video_path: str) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return float(n_frames / fps) if fps > 0 else 0.0


def detect_table_break_candidates(
    video_path: str,
    table_roi,
    *,
    dense_step: float = 0.5,
    min_break_sec: float = 8.0,
    min_start_sec: float = 20.0,
    merge_gap_sec: float = 2.0,
    resume_search_sec: float = 45.0,
    log_fn=None,
) -> list[dict]:
    """Find long low-table-motion windows that may be set breaks.

    This intentionally does NOT decide whether a candidate is a real side swap.
    Step 3.1 validates candidates against trusted Step 2 identities.
    """
    if table_roi is None:
        return []

    duration = _video_duration_sec(video_path)
    samples = _compute_table_energy_window(
        video_path,
        table_roi,
        0.0,
        duration,
        dense_step=dense_step,
    )
    if len(samples) < 10:
        return []

    times = [s[0] for s in samples]
    energies = [s[1] for s in samples]

    smooth_half = 3.0
    smoothed: list[float] = []
    for t, _energy in samples:
        vals = [energies[j] for j in range(len(times)) if abs(times[j] - t) <= smooth_half]
        smoothed.append(float(sum(vals) / len(vals)) if vals else 0.0)

    arr = np.asarray(smoothed, dtype=np.float32)
    p10 = float(np.percentile(arr, 10))
    p30 = float(np.percentile(arr, 30))
    median = float(np.percentile(arr, 50))
    p75 = float(np.percentile(arr, 75))
    # Conservative low-energy threshold: enough to collect real idle windows,
    # but not so high that ordinary slow rallies become break candidates.
    low_threshold = min(p30, median * 0.45)
    low_threshold = max(low_threshold, p10 + ((p30 - p10) * 0.35))

    if log_fn:
        log_fn(
            f"Step 3.1: break candidates — table energy samples={len(samples)} "
            f"low_threshold={low_threshold:.2f} median={median:.2f} p75={p75:.2f}"
        )

    candidates: list[dict] = []
    break_window_sec = 15.0
    n_win = max(1, int(break_window_sec / dense_step))
    if len(smoothed) < n_win:
        return []

    window_means: list[tuple[int, float]] = []
    for idx in range(0, len(smoothed) - n_win + 1):
        if times[idx] < min_start_sec:
            continue
        window_means.append((idx, float(np.mean(arr[idx:idx + n_win]))))
    if not window_means:
        return []

    mean_values = np.asarray([mean for _idx, mean in window_means], dtype=np.float32)
    window_threshold = min(float(np.percentile(mean_values, 25)), median * 0.75)
    # Keep local minima so a real set break with modest table motion still
    # becomes a candidate. Identity-side validation rejects false positives.
    local_minima: list[tuple[int, float]] = []
    for pos, (idx, mean) in enumerate(window_means):
        prev_mean = window_means[pos - 1][1] if pos > 0 else float("inf")
        next_mean = window_means[pos + 1][1] if pos + 1 < len(window_means) else float("inf")
        if mean <= window_threshold and mean <= prev_mean and mean <= next_mean:
            local_minima.append((idx, mean))

    selected_minima: list[tuple[int, float]] = []
    min_sep_samples = max(1, int(30.0 / dense_step))
    for idx, mean in sorted(local_minima, key=lambda item: item[1]):
        if any(abs(idx - kept_idx) < min_sep_samples for kept_idx, _kept_mean in selected_minima):
            continue
        selected_minima.append((idx, mean))
    selected_minima.sort(key=lambda item: item[0])

    required_high = max(1, int(3.0 / dense_step))
    for best_idx, best_mean in selected_minima:
        expand_threshold = min(median * 0.75, max(low_threshold, best_mean * 1.60))
        lo_idx = best_idx
        hi_idx = min(len(smoothed) - 1, best_idx + n_win - 1)
        while lo_idx > 0 and times[lo_idx - 1] >= min_start_sec and smoothed[lo_idx - 1] <= expand_threshold:
            lo_idx -= 1
        while hi_idx < len(smoothed) - 1 and smoothed[hi_idx + 1] <= expand_threshold:
            hi_idx += 1

        search_hi = min(len(smoothed) - 1, hi_idx + max(1, int(resume_search_sec / dense_step)))
        resume_idx = None
        run = 0
        for idx in range(hi_idx, search_hi + 1):
            if smoothed[idx] >= p75:
                run += 1
                if run >= required_high:
                    resume_idx = idx - required_high + 1
                    break
            else:
                run = 0
        t_start = float(times[lo_idx])
        core_end = float(times[hi_idx])
        t_end = float(times[resume_idx]) if resume_idx is not None else core_end
        duration_sec = t_end - t_start
        if duration_sec < min_break_sec:
            continue

        candidates.append(
            {
                "t_break_start": t_start,
                "t_break_core_end": core_end,
                "t_break_end": t_end,
                "duration": float(duration_sec),
                "avg_energy": float(np.mean(arr[lo_idx:hi_idx + 1])),
                "threshold": float(low_threshold),
                "window_mean": float(best_mean),
                "window_threshold": float(window_threshold),
            }
        )

    # De-duplicate overlapping/nearby candidates; keep the longer lower-energy one.
    merged: list[dict] = []
    for cand in sorted(candidates, key=lambda c: (c["t_break_start"], c["avg_energy"])):
        if not merged:
            merged.append(cand)
            continue
        prev = merged[-1]
        if cand["t_break_start"] <= prev["t_break_end"] + merge_gap_sec:
            prev_score = (prev["duration"], -prev["avg_energy"])
            cand_score = (cand["duration"], -cand["avg_energy"])
            if cand_score > prev_score:
                merged[-1] = cand
            continue
        merged.append(cand)

    if log_fn:
        log_fn(f"Step 3.1: break candidates — {len(merged)} candidate(s)")
        for idx, cand in enumerate(merged, start=1):
            log_fn(
                f"  candidate {idx}: [{cand['t_break_start']:.1f}s .. "
                f"{cand['t_break_end']:.1f}s] dur={cand['duration']:.1f}s "
                f"avg_energy={cand['avg_energy']:.2f}"
            )
    return merged


def validate_known_player_swaps(
    candidates: list[dict],
    tl_a: list[tuple[float, str]],
    tl_b: list[tuple[float, str]],
    *,
    init_a: str,
    init_b: str,
    player_a_name: str,
    player_b_name: str,
    pre_window_sec: float = 45.0,
    post_window_sec: float = 45.0,
    guard_sec: float = 3.0,
    refine_start_with_last_old: bool = True,
    log_fn=None,
) -> list[dict]:
    """Accept only break candidates validated by trusted Step 2 player names."""
    swaps: list[dict] = []
    cur_a, cur_b = init_a, init_b
    cursor = 0.0

    for cand in candidates:
        t_start = float(cand["t_break_start"])
        t_end = float(cand["t_break_end"])
        if t_start < cursor:
            continue

        expected_a = infer_opposite_side(cur_a)
        expected_b = infer_opposite_side(cur_b)
        pre_a = dominant_side_in_range(tl_a, max(0.0, t_start - pre_window_sec), max(0.0, t_start - guard_sec))
        pre_b = dominant_side_in_range(tl_b, max(0.0, t_start - pre_window_sec), max(0.0, t_start - guard_sec))
        post_a = dominant_side_in_range(tl_a, t_end + guard_sec, t_end + post_window_sec)
        post_b = dominant_side_in_range(tl_b, t_end + guard_sec, t_end + post_window_sec)

        pre_a_side, pre_b_side = pre_a["side"], pre_b["side"]
        post_a_side, post_b_side = post_a["side"], post_b["side"]

        contradictions: list[str] = []
        if pre_a_side is not None and pre_a_side != cur_a:
            contradictions.append(f"{player_a_name} pre={pre_a_side} expected={cur_a}")
        if pre_b_side is not None and pre_b_side != cur_b:
            contradictions.append(f"{player_b_name} pre={pre_b_side} expected={cur_b}")
        if post_a_side is not None and post_a_side != expected_a:
            contradictions.append(f"{player_a_name} post={post_a_side} expected={expected_a}")
        if post_b_side is not None and post_b_side != expected_b:
            contradictions.append(f"{player_b_name} post={post_b_side} expected={expected_b}")

        a_flipped = pre_a_side == cur_a and post_a_side == expected_a
        b_flipped = pre_b_side == cur_b and post_b_side == expected_b
        one_sided_a = post_a_side == expected_a and pre_b_side in {None, cur_b} and post_b_side in {None, expected_b}
        one_sided_b = post_b_side == expected_b and pre_a_side in {None, cur_a} and post_a_side in {None, expected_a}

        if a_flipped and b_flipped:
            mode = "both"
        elif one_sided_a:
            mode = "a-only"
        elif one_sided_b:
            mode = "b-only"
        else:
            mode = ""

        validation = {
            "player_a_name": player_a_name,
            "player_b_name": player_b_name,
            "current_a": cur_a,
            "current_b": cur_b,
            "expected_a_after": expected_a,
            "expected_b_after": expected_b,
            "pre_a": pre_a,
            "pre_b": pre_b,
            "post_a": post_a,
            "post_b": post_b,
            "contradictions": contradictions,
        }

        if mode and not contradictions:
            last_old: list[float] = []
            for ts, side in tl_a:
                if t_start - guard_sec <= ts <= t_end and side == cur_a:
                    last_old.append(float(ts))
            for ts, side in tl_b:
                if t_start - guard_sec <= ts <= t_end and side == cur_b:
                    last_old.append(float(ts))
            refined_start = max(t_start, max(last_old)) if (refine_start_with_last_old and last_old) else t_start

            swap = {
                "t_swap": float((refined_start + t_end) / 2.0),
                "t_break_start": float(refined_start),
                "t_break_end": float(t_end),
                "t_cutoff": float(t_end),
                "mode": mode,
                "source": "break_candidate_plus_step2_identity",
                "validation": validation,
                "break_candidate": cand,
            }
            swaps.append(swap)
            if log_fn:
                log_fn(
                    f"Step 3.1: accepted swap [{refined_start:.1f}s .. {t_end:.1f}s] "
                    f"mode={mode} ({player_a_name}: {cur_a}->{expected_a}, "
                    f"{player_b_name}: {cur_b}->{expected_b})"
                )
            cur_a, cur_b = expected_a, expected_b
            cursor = t_end + 20.0
        elif log_fn:
            reason = "; ".join(contradictions) if contradictions else "insufficient side evidence"
            log_fn(f"Step 3.1: rejected break candidate [{t_start:.1f}s .. {t_end:.1f}s] — {reason}")

    return swaps


def _point_value(point, name: str, default=None):
    if isinstance(point, dict):
        return point.get(name, default)
    return getattr(point, name, default)


def _point_counts_toward_score(point) -> bool:
    flags = list(_point_value(point, "flags", []) or [])
    return not any(flag in {"rally_label_let", "let_no_score"} for flag in flags)


def _point_id(point, idx: int) -> str:
    return str(_point_value(point, "id", f"pt_{idx + 1:04d}"))


def rally_anchor_side_state(
    point,
    tl_a: list[tuple[float, str]],
    tl_b: list[tuple[float, str]],
    *,
    current_a: str,
    current_b: str,
    window_before_sec: float = 3.0,
    window_after_sec: float = 6.0,
) -> dict:
    """Classify trusted-player side state around one rally anchor."""
    t_start = float(_point_value(point, "active_start", None) or _point_value(point, "t_start", 0.0))
    t_end = float(_point_value(point, "active_end", None) or _point_value(point, "t_end", t_start))
    lo = max(0.0, t_start - float(window_before_sec))
    hi = max(t_end, t_start + float(window_after_sec))
    ev_a = dominant_side_in_range(tl_a, lo, hi, min_samples=1, min_majority_frac=0.55)
    ev_b = dominant_side_in_range(tl_b, lo, hi, min_samples=1, min_majority_frac=0.55)

    side_a = ev_a["side"]
    side_b = ev_b["side"]
    expected_a = infer_opposite_side(current_a)
    expected_b = infer_opposite_side(current_b)

    contradictions: list[str] = []
    if side_a is not None and side_b is not None and side_a == side_b:
        contradictions.append(f"both_players_same_side={side_a}")

    observed = int(side_a is not None) + int(side_b is not None)
    matches_current = observed > 0
    matches_flipped = observed > 0
    if side_a is not None:
        matches_current = matches_current and side_a == current_a
        matches_flipped = matches_flipped and side_a == expected_a
    if side_b is not None:
        matches_current = matches_current and side_b == current_b
        matches_flipped = matches_flipped and side_b == expected_b

    if contradictions:
        state = "contradictory"
    elif matches_flipped:
        state = "flipped"
    elif matches_current:
        state = "current"
    else:
        state = "unknown"

    return {
        "state": state,
        "t_start": t_start,
        "t_end": t_end,
        "window": {"lo": lo, "hi": hi},
        "player_a": ev_a,
        "player_b": ev_b,
        "current_a": current_a,
        "current_b": current_b,
        "expected_a_after": expected_a,
        "expected_b_after": expected_b,
        "contradictions": contradictions,
    }


def detect_rally_anchor_side_swaps(
    points: list,
    tl_a: list[tuple[float, str]],
    tl_b: list[tuple[float, str]],
    *,
    init_a: str,
    init_b: str,
    player_a_name: str,
    player_b_name: str,
    best_of: int,
    log_fn=None,
) -> dict:
    """Detect side swaps by scanning existing rally anchors after safe jumps.

    This consumes already-debugged RallyTimelinePoint outputs: t_start/t_end and
    LET flags. It does not reimplement rally segmentation.
    """
    ordered = sorted(list(points), key=lambda p: float(_point_value(p, "t_start", 0.0)))
    swaps: list[dict] = []
    mid_set_swaps: list[dict] = []
    checked: list[dict] = []

    current_a, current_b = init_a, init_b
    current_set = 1
    scoring_in_set = 0
    deciding_mid_swap_seen = False
    prev_point = None

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    for idx, point in enumerate(ordered):
        is_scoring = _point_counts_toward_score(point)
        prospective_score = scoring_in_set + (1 if is_scoring else 0)
        is_deciding = int(best_of) > 0 and current_set == int(best_of)
        safe_threshold = 5 if (is_deciding and not deciding_mid_swap_seen) else 11
        should_check = prospective_score >= safe_threshold
        can_accept_swap = scoring_in_set >= safe_threshold

        if should_check:
            evidence = rally_anchor_side_state(
                point,
                tl_a,
                tl_b,
                current_a=current_a,
                current_b=current_b,
            )
            evidence_record = {
                "point_index": idx,
                "point_id": _point_id(point, idx),
                "set_number": current_set,
                "scoring_in_set_before": scoring_in_set,
                "prospective_score": prospective_score,
                "is_scoring": is_scoring,
                "safe_threshold": safe_threshold,
                "can_accept_swap": can_accept_swap,
                "evidence": evidence,
            }
            checked.append(evidence_record)

            if evidence["state"] == "flipped":
                if not can_accept_swap:
                    evidence_record["ignored_reason"] = "flipped_before_minimum_completed_score"
                    if is_scoring:
                        scoring_in_set += 1
                    prev_point = point
                    continue

                prev_t_end = float(_point_value(prev_point, "t_end", evidence["t_start"])) if prev_point is not None else evidence["t_start"]
                break_start = prev_t_end
                break_end = float(_point_value(point, "t_start", evidence["t_start"]))
                event = {
                    "t_swap": float((break_start + break_end) / 2.0),
                    "t_break_start": float(break_start),
                    "t_break_end": float(break_end),
                    "t_cutoff": float(break_end),
                    "mode": "rally-anchor",
                    "source": "rally_anchor_side_scan",
                    "point_before_id": (_point_id(prev_point, idx - 1) if prev_point is not None else ""),
                    "point_after_id": _point_id(point, idx),
                    "point_after_index": idx,
                    "scoring_in_set_before": scoring_in_set,
                    "prospective_score": prospective_score,
                    "validation": {
                        "player_a_name": player_a_name,
                        "player_b_name": player_b_name,
                        "evidence": evidence,
                    },
                }

                if is_deciding and not deciding_mid_swap_seen:
                    event["kind"] = "mid_deciding_set_swap"
                    mid_set_swaps.append(event)
                    _log(
                        f"Step 3.1: mid-set side swap at Set {current_set}, "
                        f"{break_start:.1f}s..{break_end:.1f}s"
                    )
                    current_a = str(evidence["expected_a_after"])
                    current_b = str(evidence["expected_b_after"])
                    deciding_mid_swap_seen = True
                else:
                    event["kind"] = "set_boundary"
                    swaps.append(event)
                    _log(
                        f"Step 3.1: rally-anchor set boundary between "
                        f"{event['point_before_id']} and {event['point_after_id']} "
                        f"[{break_start:.1f}s .. {break_end:.1f}s]"
                    )
                    current_a = str(evidence["expected_a_after"])
                    current_b = str(evidence["expected_b_after"])
                    current_set += 1
                    scoring_in_set = 1 if is_scoring else 0
                    deciding_mid_swap_seen = False
                    prev_point = point
                    continue

        if is_scoring:
            scoring_in_set += 1
        prev_point = point

    return {
        "swaps": swaps,
        "mid_set_swaps": mid_set_swaps,
        "checked_anchor_count": len(checked),
        "checked_anchors": checked[:80],
        "total_anchor_count": len(ordered),
        "scoring_anchor_count": sum(1 for point in ordered if _point_counts_toward_score(point)),
    }


def refine_swap_to_transition_window(
    video_path: str,
    yolo,
    player_zone: tuple,
    table_center_x: float,
    coarse_t: float,
    table_roi=None,
    search_before: float = 90.0,
    search_after: float = 10.0,
    dense_step: float = 0.5,
    log_fn=None,
) -> tuple[float | None, float | None]:
    """Refine a coarse swap timestamp to the actual set-break window using
    **table motion energy** — the mean absolute pixel difference between
    consecutive frames within the table ROI region.

    During rally play the ball and arms create high motion energy on the table.
    During a set break (players walking to chair, drinking water, discussing
    with coach) the table region is essentially static → energy drops.

    The break is identified as the longest contiguous low-energy period within
    the asymmetric search window [coarse_t - search_before, coarse_t + search_after].

    Returns:
        (t_break_start, t_break_end):
          - t_break_start ≈ t_end of last rally of the previous set
          - t_break_end   ≈ t_start of first rally of the next set
        Returns (None, None) if table_roi is not available or no break found.
    """
    if table_roi is None:
        return None, None

    import cv2 as _cv2
    cap = _cv2.VideoCapture(str(video_path))
    fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n_frames / fps if fps > 0 else 0.0
    cap.release()

    t_lo = max(0.0, coarse_t - search_before)
    t_hi = min(duration, coarse_t + search_after)

    # 1. Compute frame-to-frame table energy in the search window.
    energy_samples = _compute_table_energy_window(
        video_path, table_roi, t_lo, t_hi, dense_step=dense_step,
    )
    if len(energy_samples) < 10:
        return None, None

    times = [s[0] for s in energy_samples]
    energies = [s[1] for s in energy_samples]

    # 2. Smooth energy over a sliding window (mean of ±smooth_half seconds).
    smooth_half = 3.0   # 6 s total sliding window
    smoothed = []
    for i, (t, _) in enumerate(energy_samples):
        window_vals = [energies[j] for j in range(len(times))
                       if abs(times[j] - t) <= smooth_half]
        smoothed.append(sum(window_vals) / len(window_vals) if window_vals else 0.0)

    if log_fn:
        log_fn(f"  table energy: {len(energy_samples)} samples in "
               f"[{t_lo:.0f}s..{t_hi:.0f}s]")
        # Compact pattern: 'H' = high energy, '.' = low, scale relative to median
        bucket_step = 2.0
        median_e = sorted(smoothed)[len(smoothed) // 2] if smoothed else 1.0
        bucket_t = t_lo
        line = []
        while bucket_t < t_hi:
            bucket_vals = [smoothed[j] for j in range(len(times))
                           if bucket_t <= times[j] < bucket_t + bucket_step]
            if not bucket_vals:
                line.append("?")
            else:
                avg = sum(bucket_vals) / len(bucket_vals)
                line.append("H" if avg > median_e * 0.5 else ".")
            bucket_t += bucket_step
        log_fn(f"  pattern @ {t_lo:.0f}s..{t_hi:.0f}s (each char = {bucket_step}s):")
        log_fn(f"    {''.join(line)}")

    # 3. Find the set break as the sliding window with the LOWEST average
    #    energy.  This is robust against brief energy spikes during the break
    #    (e.g. a player's arm crossing the table ROI while walking past).
    #
    #    Sweep a window of `break_window_sec` seconds across the search range
    #    and pick the position with minimum mean energy.  Then expand the
    #    window outward as long as energy stays below an adaptive threshold.
    break_window_sec = 15.0
    n_win = max(1, int(break_window_sec / dense_step))
    if len(smoothed) < n_win:
        return None, None

    # Sweep for minimum-energy window
    best_mean = float("inf")
    best_idx = 0
    for i in range(len(smoothed) - n_win + 1):
        win_mean = sum(smoothed[i:i + n_win]) / n_win
        if win_mean < best_mean:
            best_mean = win_mean
            best_idx = i

    # Adaptive threshold: break energy should be well below the median
    median_e = sorted(smoothed)[len(smoothed) // 2]
    expand_threshold = max(median_e * 0.4, best_mean * 2.0)

    # Expand outward from the best window as long as energy stays low.
    lo_idx = best_idx
    hi_idx = best_idx + n_win - 1
    while lo_idx > 0 and smoothed[lo_idx - 1] <= expand_threshold:
        lo_idx -= 1
    while hi_idx < len(smoothed) - 1 and smoothed[hi_idx + 1] <= expand_threshold:
        hi_idx += 1

    t_break_start = times[lo_idx]

    # Break END refinement: the core low-energy window ends when energy rises
    # slightly (e.g. player starts walking back to table), but actual play
    # doesn't resume for several more seconds.  To find when play truly
    # resumes, walk forward from the core end and look for the first SUSTAINED
    # high-energy period (>= 75th percentile for >= 3 s).  That is the first
    # rally of the new set.
    p75 = sorted(smoothed)[min(len(smoothed) - 1, int(len(smoothed) * 0.75))]
    rally_resume_idx = None
    required_consecutive = max(1, int(3.0 / dense_step))  # ~6 samples at 0.5s
    run = 0
    for i in range(hi_idx, len(smoothed)):
        if smoothed[i] >= p75:
            run += 1
            if run >= required_consecutive:
                rally_resume_idx = i - required_consecutive + 1
                break
        else:
            run = 0

    if rally_resume_idx is not None:
        t_break_end = times[rally_resume_idx]
    else:
        t_break_end = times[hi_idx]  # fallback to core end

    break_dur = t_break_end - t_break_start

    if log_fn:
        log_fn(f"  break found: [{t_break_start:.1f}s .. {t_break_end:.1f}s] "
               f"duration={break_dur:.1f}s  avg_energy={best_mean:.2f} "
               f"(median={median_e:.2f}, p75={p75:.2f})")

    if break_dur < 8.0:
        return None, None
    return t_break_start, t_break_end


def backtrack_swap_start(tl_a: list[tuple[float, str]],
                         tl_b: list[tuple[float, str]],
                         t_swap: float,
                         after_a: str,
                         after_b: str,
                         init_a: str,
                         init_b: str,
                         step: float,
                         max_lookback: float,
                         window: float) -> float:
    """Walk backward from t_swap as long as the state is consistent with the
    swapped (post-swap) configuration.  Stops when an explicit return to the
    baseline state is observed.  Treats None (no data) as compatible."""
    t = t_swap
    while t - step >= t_swap - max_lookback:
        prev = t - step
        sa = smoothed_side(tl_a, prev, window=window)
        sb = smoothed_side(tl_b, prev, window=window)
        # Reject only on explicit return to baseline state.
        if (sa is not None and sa == init_a) or (sb is not None and sb == init_b):
            break
        t = prev
    return t


# --- main -----------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--sample-step", type=float, default=2.0,
                        help="Frame sampling interval in seconds (default 2.0)")
    parser.add_argument("--baseline-end", type=float, default=60.0,
                        help="End of baseline window for Set 1 reference (default 60s)")
    parser.add_argument("--baseline-start", type=float, default=10.0)
    parser.add_argument("--smooth-window", type=float, default=10.0,
                        help="±seconds for sliding-window majority vote (default 10)")
    parser.add_argument("--stability-seconds", type=float, default=15.0,
                        help="Post-swap state must persist for this many seconds (default 15)")
    parser.add_argument("--match-threshold", type=float, default=DEFAULT_MATCH_THRESHOLD,
                        help=f"Face-DB match threshold (default {DEFAULT_MATCH_THRESHOLD})")
    parser.add_argument("--best-of", type=int, default=None,
                        help="If set (e.g. 3, 5, 7), infer number of sets played from swap count")
    parser.add_argument("--no-refine", action="store_true",
                        help="Skip dense-sample refinement step (faster, less precise)")
    parser.add_argument("--refine-before", type=float, default=90.0,
                        help="Dense-sample window: seconds BEFORE coarse swap (default 90)")
    parser.add_argument("--refine-after", type=float, default=10.0,
                        help="Dense-sample window: seconds AFTER coarse swap (default 10)")
    parser.add_argument("--refine-step", type=float, default=0.5,
                        help="Dense-sample step in seconds (default 0.5)")
    parser.add_argument("--save-csv", type=str, default=None,
                        help="Optional path to dump per-sample records as CSV")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        return 1

    config = ProductionPipelineConfig()
    face_db_path = ROOT / "data" / "players" / "faces.json"
    face_model_path = ROOT / "data" / "models" / "face" / "w600k_r50.onnx"

    print(f"Video:           {video_path}")
    print(f"Table weights:   {config.table_weights_path}")
    print(f"Pose weights:    {config.pose_weights_path}")
    print(f"Face DB:         {face_db_path}")
    print()

    # Load FaceDB + face embedder
    face_db = FaceDB(face_db_path)
    if len(face_db) < 2:
        print(f"WARNING: face DB has only {len(face_db)} player(s); need at least 2 for swap detection.")
    print(f"Enrolled players: {[r.name for r in face_db.records]}")
    embedder = FaceEmbedder(face_model_path)

    # Detect table ROI + player zone
    print("\n[1/5] Detecting table ROI + player zone...")
    table_roi, player_zone = detect_table_roi_and_player_zone(
        str(video_path), config.table_weights_path,
    )
    if table_roi is None:
        print("ERROR: table ROI detection failed.")
        return 1
    table_center_x = table_roi.x + table_roi.w / 2.0
    print(f"  table ROI:    x={table_roi.x} y={table_roi.y} w={table_roi.w} h={table_roi.h}")
    print(f"  table center: x={table_center_x:.0f}")
    print(f"  player zone:  x=[{player_zone[0]:.0f},{player_zone[2]:.0f}] y=[{player_zone[1]:.0f},{player_zone[3]:.0f}]")

    # Load YOLO pose
    print("\n[2/5] Loading YOLO pose...")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required: torch.cuda.is_available() returned False")
    from ultralytics import YOLO
    yolo = YOLO(str(config.pose_weights_path))

    # Sample frames
    print(f"\n[3/5] Sampling frames every {args.sample_step}s...")
    records = sample_positions(
        str(video_path), yolo, embedder, face_db,
        player_zone=player_zone,
        table_center_x=table_center_x,
        sample_step=args.sample_step,
        match_threshold=args.match_threshold,
        log_fn=lambda m: print(f"  {m}"),
    )
    print(f"  collected {len(records)} body detections")

    # Optional CSV dump for inspection
    if args.save_csv:
        csv_path = Path(args.save_csv)
        with csv_path.open("w", encoding="utf-8") as fh:
            fh.write("t,identity,side,sim,rank,face_score\n")
            for r in records:
                fh.write(f"{r['t']:.2f},{r['identity'] or ''},{r['side']},"
                         f"{'' if r['sim'] is None else f'{r['sim']:.3f}'},"
                         f"{r['rank']},{r['face_score']:.2f}\n")
        print(f"  CSV saved: {csv_path}")

    # Determine the two main players from the actual sample distribution.
    # The face DB may contain many enrolled players; the two playing in this
    # video are the two identities that account for the most samples.
    print(f"\n[4/5] Identifying the two main players from sampled detections...")
    identity_counts = Counter(r["identity"] for r in records if r["identity"])
    if not identity_counts:
        print("ABORT: no face matched any DB record — cannot determine players.")
        print("Re-enroll players or lower --match-threshold.")
        return 1
    print("  Detection counts per identity:")
    for n, c in identity_counts.most_common():
        print(f"    {n}: {c}")
    top2 = identity_counts.most_common(2)
    if len(top2) < 2:
        print(f"ABORT: only one identity matched ({top2[0][0]} x {top2[0][1]}). "
              "Need 2 distinct players for swap detection.")
        return 1
    name_a, count_a = top2[0]
    name_b, count_b = top2[1]
    tl_a = [(r["t"], r["side"]) for r in records if r["identity"] == name_a]
    tl_b = [(r["t"], r["side"]) for r in records if r["identity"] == name_b]
    print(f"  Main players selected: A={name_a} ({count_a} samples)  B={name_b} ({count_b} samples)")
    if len(tl_a) < 5 or len(tl_b) < 3:
        print("ABORT: too few identified samples for one of the main players.")
        return 1

    # Baseline (Set 1 sides)
    init_a = baseline_state(tl_a, args.baseline_start, args.baseline_end)
    init_b = baseline_state(tl_b, args.baseline_start, args.baseline_end)
    print(f"  baseline ({args.baseline_start:.0f}s..{args.baseline_end:.0f}s):"
          f"  {name_a}={init_a}  {name_b}={init_b}")
    if init_a is None or init_b is None or init_a == init_b:
        print("ABORT: baseline sides are not opposite — cannot determine swap.")
        print("Try a different --baseline-start / --baseline-end window.")
        return 1

    # Find swap
    print(f"\n[5/5] Searching for ALL swaps (stability >= {args.stability_seconds}s)...")
    duration = max(r["t"] for r in records) if records else 0.0

    # Iteratively detect every swap by walking forward.  After each swap, the
    # current expected state becomes the swapped state, and we continue
    # searching from past the stability window.
    swaps: list[dict] = []   # each: {t_start, t_detect, mode, before_a/b, after_a/b}
    current_a, current_b = init_a, init_b
    search_cursor = args.baseline_end

    while search_cursor <= duration:
        swap_result = find_swap(
            tl_a, tl_b,
            search_start=search_cursor,
            search_end=duration,
            step=args.sample_step,
            init_a=current_a, init_b=current_b,
            stability_seconds=args.stability_seconds,
            window=args.smooth_window,
        )
        if swap_result is None:
            break
        t_swap, swap_mode = swap_result

        flipped_a = SIDE_R if current_a == SIDE_L else SIDE_L
        flipped_b = SIDE_R if current_b == SIDE_L else SIDE_L
        after_a = smoothed_side(tl_a, t_swap, window=args.smooth_window) or flipped_a
        after_b = smoothed_side(tl_b, t_swap, window=args.smooth_window) or flipped_b

        t_swap_start = backtrack_swap_start(
            tl_a, tl_b, t_swap, after_a, after_b,
            init_a=current_a, init_b=current_b,
            step=args.sample_step,
            max_lookback=30.0,
            window=args.smooth_window,
        )

        # Refinement: dense-sample around the coarse swap point to locate
        # the actual transition window (no rally configuration).
        t_break_start, t_break_end = (None, None)
        if not args.no_refine:
            t_break_start, t_break_end = refine_swap_to_transition_window(
                str(video_path), yolo, player_zone, table_center_x,
                coarse_t=t_swap_start,
                table_roi=table_roi,
                search_before=args.refine_before,
                search_after=args.refine_after,
                dense_step=args.refine_step,
                log_fn=lambda m: print(f"  [refine swap @ ~{t_swap_start:.0f}s] {m}"),
            )

        swaps.append({
            "t_detect": t_swap,
            "t_start": t_swap_start,
            "mode": swap_mode,
            "before_a": current_a, "before_b": current_b,
            "after_a": after_a, "after_b": after_b,
            "t_break_start": t_break_start,
            "t_break_end": t_break_end,
        })

        # Advance: new "current" state is the swapped state; resume searching
        # past the stability window so we don't immediately re-detect the same
        # swap as a candidate.
        current_a, current_b = after_a, after_b
        search_cursor = t_swap + args.stability_seconds + args.sample_step

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    if not swaps:
        print(f"NO SWAP DETECTED in window [{args.baseline_end:.0f}s..{duration:.0f}s].")
        print(f"  Baseline state: {name_a}={init_a}  {name_b}={init_b}")
        print(f"  Inferred: only 1 set played (or video too short).")
        return 0

    # Detailed per-swap report
    print(f"  Initial state ({args.baseline_start:.0f}s..{args.baseline_end:.0f}s):"
          f"  {name_a}={init_a}  {name_b}={init_b}")
    print()
    for i, s in enumerate(swaps, 1):
        print(f"  Swap #{i}:  coarse t_start={s['t_start']:.2f}s  "
              f"(detected at {s['t_detect']:.2f}s, mode={s['mode']})")
        print(f"    before:  {name_a}={s['before_a']}  {name_b}={s['before_b']}")
        print(f"    after :  {name_a}={s['after_a']}  {name_b}={s['after_b']}")
        if s.get("t_break_start") is not None and s.get("t_break_end") is not None:
            br_dur = s["t_break_end"] - s["t_break_start"]
            print(f"    break window (refined):"
                  f"  [{s['t_break_start']:.2f}s .. {s['t_break_end']:.2f}s]"
                  f"  duration={br_dur:.1f}s")
            print(f"      → t_end (last rally of previous set):  {s['t_break_start']:.2f}s")
            print(f"      → t_start (first rally of next set):   {s['t_break_end']:.2f}s")
    print()

    # Build timeline ranges per period (each period = between swaps).
    # Use the refined break END as the cutoff so each period is
    # "first rally of set N → first rally of set N+1 (exclusive)".
    n_swaps = len(swaps)
    print(f"  Total swaps: {n_swaps}")
    cutoffs = []
    for s in swaps:
        # Prefer refined break end (start of next set's first rally).
        # Fall back to coarse t_start if refinement failed.
        cutoffs.append(s["t_break_end"] if s.get("t_break_end") is not None else s["t_start"])
    boundaries = [0.0] + cutoffs + [duration]
    print(f"  Periods (separated by swap):")
    for i in range(len(boundaries) - 1):
        print(f"    Period {i+1}:  [{boundaries[i]:.2f}s .. {boundaries[i+1]:.2f}s]"
              f"   duration={boundaries[i+1] - boundaries[i]:.1f}s")
    print()

    # Infer number of sets played, given best-of N from the user.
    best_of = args.best_of
    if best_of is not None and best_of > 0:
        sets_max = best_of
        # In a deciding set (set N for BO_N where N is odd), there is an
        # additional mid-set swap when the players' total score reaches 5.
        # So number of sets played and number of swaps relate as:
        #   no mid-set swap: n_swaps == n_sets - 1
        #   with mid-set swap: n_swaps == n_sets   (only possible when reached deciding set)
        candidates = []
        for n_sets in range(1, sets_max + 1):
            if n_swaps == n_sets - 1:
                candidates.append((n_sets, "no mid-set swap"))
            if n_sets == sets_max and n_swaps == n_sets:
                candidates.append((n_sets, "with mid-set swap in deciding set"))
        print(f"  Inference (best_of={best_of}):")
        if candidates:
            for n_sets, note in candidates:
                print(f"    -> {n_sets} sets played   ({note})")
        else:
            print(f"    -> n_swaps={n_swaps} does not match any valid (sets, mid-swap) pattern for BO{best_of}.")
            print(f"       Possible reasons: noisy data, missed swap, or false swap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
