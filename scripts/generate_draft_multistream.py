from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_contract import DraftMatch, DraftPointEvent, save_draft_match
from backend.ai_multistream_rally import detect_multistream_rallies, extract_multistream_signals


def _smooth_and_normalize(values: List[float]) -> np.ndarray:
    if not values:
        return np.array([], dtype=np.float32)

    arr = np.asarray(values, dtype=np.float32)
    if arr.size < 5:
        p10, p95 = float(arr.min()), float(arr.max())
        return np.clip((arr - p10) / (p95 - p10 + 1e-6), 0.0, 1.0)

    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    smooth = np.convolve(arr, kernel, mode="same")
    p10, p95 = np.percentile(smooth, 10), np.percentile(smooth, 95)
    return np.clip((smooth - p10) / (p95 - p10 + 1e-6), 0.0, 1.0)


def _signal_array(values: List[float], sample_count: int) -> np.ndarray:
    if values and len(values) == sample_count:
        return np.asarray(values, dtype=np.float32)
    return np.zeros(sample_count, dtype=np.float32)


def _normalize_segment_flags(seg, mode: str, signals) -> tuple[list[str], str | None]:
    flags = list(seg.flags)
    if mode == "fused":
        flags.append("multistream_fused")
    elif mode == "player":
        flags.append("player_only")
    elif mode == "ball":
        flags.append("ball_only")
    elif mode == "table":
        flags.append("table_only")
    elif mode == "table_refined":
        flags.append("table_role_refined")
    elif mode == "table_ball_refined":
        flags.append("table_ball_refined")
    flags.append(f"player_signal_{signals.player_signal_source}")
    flags.append(f"ball_signal_{signals.ball_signal_source}")
    starter_role = str(seg.server_role) if getattr(seg, "server_role", None) in {"A", "B"} else None
    return sorted(set(flags)), starter_role


def _build_points_with_active_windows(
    segments,
    *,
    mode: str,
    signals,
    video_end_sec: float,
) -> tuple[list[DraftPointEvent], list[dict], list[dict]]:
    segments = sorted(segments, key=lambda seg: float(seg.t_start))
    points: List[DraftPointEvent] = []
    excluded_let_starts: List[dict] = []
    pending_let_starts: List[dict] = []

    for seg in segments:
        normalized_flags, starter_role = _normalize_segment_flags(seg, mode, signals)
        record = {
            "t_start": float(seg.t_start),
            "t_end": float(seg.t_end),
            "starter_role": starter_role,
            "flags": normalized_flags,
        }
        if "let_no_score" in normalized_flags or "rally_label_let" in normalized_flags:
            excluded_let_starts.append(record)
            pending_let_starts.append(record)
            continue

        preceding_let_starts = [float(item["t_start"]) for item in pending_let_starts]
        points.append(
            DraftPointEvent(
                id=f"pt_{len(points) + 1:04d}",
                t_start=float(seg.t_start),
                t_end=float(seg.t_end),
                active_start=float(seg.t_start),
                starter_role=starter_role,
                preceding_let_count=len(preceding_let_starts),
                preceding_let_starts=preceding_let_starts,
                service_attempt_index=len(preceding_let_starts) + 1,
                winner="unknown",
                confidence=float(seg.confidence),
                flags=normalized_flags,
                source="ai",
            )
        )
        pending_let_starts = []

    unattached_trailing_let_starts = list(pending_let_starts)
    for idx, point in enumerate(points):
        if idx + 1 < len(points):
            point.active_end = float(points[idx + 1].t_start)
            point.boundary_mode = "next_start_exclusive"
        else:
            point.active_end = float(max(video_end_sec, point.t_end))
            point.boundary_mode = "video_end_open_tail"

    return points, excluded_let_starts, unattached_trailing_let_starts


def _fill_short_false_gaps(mask: np.ndarray, max_gap_samples: int = 1) -> np.ndarray:
    if mask.size == 0 or max_gap_samples < 1:
        return mask
    out = mask.astype(bool).copy()
    gap_start = None
    for idx, value in enumerate(out):
        if not value and gap_start is None:
            gap_start = idx
        elif value and gap_start is not None:
            gap_len = idx - gap_start
            left_live = gap_start > 0 and out[gap_start - 1]
            right_live = True
            if left_live and right_live and gap_len <= max_gap_samples:
                out[gap_start:idx] = True
            gap_start = None
    return out


def _compute_search_upper_bound(points: List[DraftPointEvent], idx: int, video_end_sec: float) -> float:
    if idx + 1 >= len(points):
        return float(video_end_sec)
    next_point = points[idx + 1]
    if next_point.preceding_let_starts:
        return float(next_point.preceding_let_starts[0])
    return float(next_point.t_start)


def _build_endpoint_support_series(signals) -> dict[str, np.ndarray]:
    sample_count = len(signals.timestamps)
    table_norm = _smooth_and_normalize(list(signals.table_energies))
    ball_norm = _smooth_and_normalize(list(signals.ball_energies))
    if ball_norm.size == 0:
        ball_norm = np.zeros_like(table_norm)

    motion_a = _smooth_and_normalize(list(signals.player_a_energies))
    motion_b = _smooth_and_normalize(list(signals.player_b_energies))
    crouch_a = _signal_array(list(signals.player_a_crouch_scores), sample_count)
    crouch_b = _signal_array(list(signals.player_b_crouch_scores), sample_count)
    serve_a = _signal_array(list(signals.player_a_serve_scores), sample_count)
    serve_b = _signal_array(list(signals.player_b_serve_scores), sample_count)
    upper_a = _signal_array(list(signals.player_a_upper_body_scores), sample_count)
    upper_b = _signal_array(list(signals.player_b_upper_body_scores), sample_count)
    foot_a = _signal_array(list(signals.player_a_footwork_scores), sample_count)
    foot_b = _signal_array(list(signals.player_b_footwork_scores), sample_count)

    competitive_a = np.maximum.reduce(
        [
            motion_a * (0.35 + (0.65 * crouch_a)),
            upper_a * (0.45 + (0.55 * crouch_a)),
            foot_a * (0.42 + (0.58 * crouch_a)),
        ]
    )
    competitive_b = np.maximum.reduce(
        [
            motion_b * (0.35 + (0.65 * crouch_b)),
            upper_b * (0.45 + (0.55 * crouch_b)),
            foot_b * (0.42 + (0.58 * crouch_b)),
        ]
    )
    live_pair = np.maximum(competitive_a, competitive_b)

    stand_a = np.clip((1.0 - crouch_a) * (1.0 - (0.35 * np.maximum(upper_a, foot_a))), 0.0, 1.0)
    stand_b = np.clip((1.0 - crouch_b) * (1.0 - (0.35 * np.maximum(upper_b, foot_b))), 0.0, 1.0)
    stand_pair = np.minimum(stand_a, stand_b)
    casual_a = np.clip(
        1.0 - np.maximum.reduce([0.90 * upper_a, 0.80 * foot_a, 0.70 * crouch_a, 0.55 * serve_a]),
        0.0,
        1.0,
    )
    casual_b = np.clip(
        1.0 - np.maximum.reduce([0.90 * upper_b, 0.80 * foot_b, 0.70 * crouch_b, 0.55 * serve_b]),
        0.0,
        1.0,
    )
    casual_pair = np.minimum(casual_a, casual_b)
    reset_pair = np.maximum(stand_pair, 0.92 * casual_pair)
    shared_activity = np.maximum.reduce([motion_a, motion_b, upper_a, upper_b, foot_a, foot_b])

    return {
        "table_norm": table_norm,
        "ball_norm": ball_norm,
        "live_pair": live_pair,
        "reset_pair": reset_pair,
        "shared_activity": shared_activity,
    }


def _refine_endpoint_from_signals(
    timestamps: np.ndarray,
    table_norm: np.ndarray,
    ball_norm: np.ndarray,
    live_pair: np.ndarray,
    reset_pair: np.ndarray,
    shared_activity: np.ndarray,
    *,
    t_start: float,
    detector_end: float,
    search_upper_bound: float,
) -> tuple[float, str, float]:
    safe_start = float(t_start)
    safe_upper = float(max(search_upper_bound, safe_start + 0.01))
    baseline_end = float(np.clip(detector_end, safe_start + 0.01, safe_upper))

    if timestamps.size == 0:
        return baseline_end, "detector_end_clamped", 0.15

    start_idx = int(np.searchsorted(timestamps, safe_start, side="left"))
    upper_idx = int(np.searchsorted(timestamps, safe_upper, side="left")) - 1
    if upper_idx < start_idx or start_idx >= timestamps.size:
        return baseline_end, "detector_end_clamped", 0.15

    interval_table = table_norm[start_idx : upper_idx + 1]
    interval_ball = ball_norm[start_idx : upper_idx + 1]
    if interval_table.size == 0:
        return baseline_end, "detector_end_clamped", 0.15

    interval_times = timestamps[start_idx : upper_idx + 1]
    interval_live = live_pair[start_idx : upper_idx + 1]
    interval_reset = reset_pair[start_idx : upper_idx + 1]
    interval_shared = shared_activity[start_idx : upper_idx + 1]
    interval_duration = float(max(0.0, safe_upper - safe_start))
    combined_live = np.maximum.reduce(
        [
            (0.44 * interval_ball) + (0.34 * interval_table) + (0.22 * interval_live),
            np.minimum(interval_ball * 1.05, np.maximum(interval_table, 0.85 * interval_live)),
            np.minimum(interval_live * 1.08, np.maximum(interval_ball, 0.70 * interval_table)),
        ]
    )

    exchange_mask = (
        ((interval_ball >= 0.18) & ((interval_table >= 0.16) | (interval_live >= 0.16)))
        | ((interval_live >= 0.26) & (interval_ball >= 0.10))
        | ((combined_live >= 0.28) & (interval_ball >= 0.08))
    )
    exchange_mask = _fill_short_false_gaps(exchange_mask, max_gap_samples=1)
    prior_exchange_peak = np.maximum.accumulate(np.where(exchange_mask, combined_live, 0.0))
    strong_exchange_mask = (
        ((interval_ball >= 0.22) & ((interval_table >= 0.12) | (interval_live >= 0.14)))
        | ((interval_ball >= 0.14) & (interval_live >= 0.26) & (interval_table >= 0.10))
        | ((combined_live >= 0.32) & (interval_ball >= 0.10) & (interval_live >= 0.20))
    )
    strong_exchange_mask = _fill_short_false_gaps(strong_exchange_mask, max_gap_samples=1)
    future_exchange_peak = np.maximum.accumulate(np.where(strong_exchange_mask[::-1], combined_live[::-1], 0.0))[::-1]
    future_ball_peak = np.maximum.accumulate(interval_ball[::-1])[::-1]

    dead_reset_mask = (
        (interval_ball <= 0.08)
        & (
            ((interval_reset >= 0.58) & (interval_live <= 0.28))
            | ((interval_table <= 0.16) & (interval_live <= 0.24))
            | ((interval_reset >= 0.70) & (interval_shared <= 0.42))
            | ((interval_reset >= 0.56) & (interval_live <= 0.22) & (interval_shared >= 0.24))
        )
    )
    dead_reset_mask = _fill_short_false_gaps(dead_reset_mask, max_gap_samples=1)
    earliest_dead_time = safe_start + min(1.05, 0.45 * max(0.0, safe_upper - safe_start))
    future_guard_t = safe_upper - min(0.95, max(0.55, 0.18 * interval_duration))
    future_guard_idx = int(np.searchsorted(interval_times, future_guard_t, side="left")) - 1
    terminal_dead_runs: list[tuple[int, int, int]] = []
    run_start = None
    for idx_local, is_dead in enumerate(dead_reset_mask):
        if is_dead and run_start is None:
            run_start = idx_local
            continue
        if (not is_dead) and run_start is not None:
            run_end = idx_local - 1
            dead_start_t = float(interval_times[run_start])
            has_exchange_before = bool(run_start > 0 and prior_exchange_peak[run_start - 1] >= 0.26)
            future_idx = run_end + 1
            if future_idx <= future_guard_idx:
                future_exchange = float(future_exchange_peak[future_idx]) if future_idx < len(future_exchange_peak) else 0.0
                future_ball = float(future_ball_peak[future_idx]) if future_idx < len(future_ball_peak) else 0.0
            else:
                future_exchange = 0.0
                future_ball = 0.0
            is_terminal = future_exchange < 0.24 or future_ball < 0.10
            if dead_start_t >= earliest_dead_time and has_exchange_before and is_terminal:
                terminal_dead_runs.append((run_start, run_end, run_end - run_start + 1))
            run_start = None
    if run_start is not None:
        run_end = len(dead_reset_mask) - 1
        dead_start_t = float(interval_times[run_start])
        has_exchange_before = bool(run_start > 0 and prior_exchange_peak[run_start - 1] >= 0.26)
        future_idx = run_end + 1
        if future_idx <= future_guard_idx:
            future_exchange = float(future_exchange_peak[future_idx]) if future_idx < len(future_exchange_peak) else 0.0
            future_ball = float(future_ball_peak[future_idx]) if future_idx < len(future_ball_peak) else 0.0
        else:
            future_exchange = 0.0
            future_ball = 0.0
        is_terminal = future_exchange < 0.24 or future_ball < 0.10
        if dead_start_t >= earliest_dead_time and has_exchange_before and is_terminal:
            terminal_dead_runs.append((run_start, run_end, run_end - run_start + 1))

    if terminal_dead_runs:
        dead_start_local, dead_end_local, dead_len = terminal_dead_runs[0]
        dead_start_t = float(interval_times[dead_start_local])
        if dead_start_t > safe_start + 0.10:
            refined_end = float(np.clip(dead_start_t, safe_start + 0.01, baseline_end))
            endpoint_confidence = float(
                np.clip(
                    0.42
                    + (0.07 * dead_len)
                    + (0.18 * float(interval_reset[dead_start_local]))
                    + (0.12 * float(1.0 - interval_ball[dead_start_local])),
                    0.28,
                    0.95,
                )
            )
            return refined_end, "dead_reset_run_start", endpoint_confidence

    if np.any(exchange_mask):
        last_live_local = int(np.flatnonzero(exchange_mask)[-1])
        refined_end = float(np.clip(float(interval_times[last_live_local]), safe_start + 0.01, baseline_end))
        endpoint_confidence = float(np.clip(0.25 + (0.65 * combined_live[last_live_local]), 0.20, 0.90))
        return refined_end, "last_exchange_support", endpoint_confidence

    return baseline_end, "detector_end_clamped", 0.20


def _refine_points_with_endpoint_signals(
    points: List[DraftPointEvent],
    *,
    signals,
    video_end_sec: float,
) -> None:
    timestamps = np.asarray(signals.timestamps, dtype=np.float32)
    support_series = _build_endpoint_support_series(signals)

    for idx, point in enumerate(points):
        search_upper_bound = _compute_search_upper_bound(points, idx, video_end_sec)
        point.search_upper_bound = float(search_upper_bound)
        refined_end, endpoint_mode, endpoint_confidence = _refine_endpoint_from_signals(
            timestamps,
            support_series["table_norm"],
            support_series["ball_norm"],
            support_series["live_pair"],
            support_series["reset_pair"],
            support_series["shared_activity"],
            t_start=float(point.t_start),
            detector_end=float(point.t_end),
            search_upper_bound=float(search_upper_bound),
        )
        point.t_end = float(refined_end)
        point.endpoint_mode = endpoint_mode
        point.endpoint_confidence = float(endpoint_confidence)


def build_draft(
    video_path: str,
    table_weights_path: str,
    *,
    pose_weights_path: str = "weights/yolov8x-pose.pt",
    best_of: int = 5,
    stride: int = 2,
    mode: str = "fused",
    player_margin_px: int = 220,
    player_fuse_gain: float = 1.0,
    player_signal_source: str = "role_tracker",
    ball_fuse_gain: float = 1.15,
    ball_signal_source: str = "none",
) -> DraftMatch:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for multi-stream draft generation.")

    if best_of <= 0 or best_of % 2 == 0:
        raise ValueError("best_of must be a positive odd number.")
    if mode == "player" and player_signal_source == "none":
        raise ValueError("player mode requires a real --player-signal-source")
    if mode == "ball" and ball_signal_source == "none":
        raise ValueError("ball mode requires --ball-signal-source classical")

    effective_player_signal_source = player_signal_source
    effective_ball_signal_source = ball_signal_source
    if mode == "table":
        effective_player_signal_source = "none"
        effective_ball_signal_source = "none"
    elif mode == "ball":
        effective_player_signal_source = "none"
    ball_tracking_profile = "standalone" if mode == "ball" else "support"

    signals = extract_multistream_signals(
        video_path,
        table_weights_path,
        pose_weights_path=pose_weights_path,
        stride=max(1, int(stride)),
        player_margin_px=int(player_margin_px),
        player_fuse_gain=float(player_fuse_gain),
        player_signal_source=effective_player_signal_source,
        ball_fuse_gain=float(ball_fuse_gain),
        ball_signal_source=effective_ball_signal_source,
        ball_tracking_profile=ball_tracking_profile,
        device="cuda",
    )
    segments = detect_multistream_rallies(signals, mode=mode)

    v_path = Path(video_path).resolve()
    video_end_sec = 0.0
    if signals.timestamps:
        video_end_sec = float(signals.timestamps[-1])
    if segments:
        video_end_sec = float(max(video_end_sec, max(float(seg.t_end) for seg in segments)))

    points, excluded_let_starts, unattached_trailing_let_starts = _build_points_with_active_windows(
        segments,
        mode=mode,
        signals=signals,
        video_end_sec=video_end_sec,
    )
    _refine_points_with_endpoint_signals(
        points,
        signals=signals,
        video_end_sec=video_end_sec,
    )

    return DraftMatch(
        video_path=str(v_path),
        video_fps=float(signals.effective_fps * max(1, int(stride))),
        best_of=int(best_of),
        created_at=datetime.now(timezone.utc).isoformat(),
        roi=signals.roi.to_dict() | {
            "x": int(signals.roi.x),
            "y": int(signals.roi.y),
            "w": int(signals.roi.w),
            "h": int(signals.roi.h),
        },
        points=points,
        analysis_metadata={
            "detector_mode": mode,
            "detector_group": "independent" if mode in {"table", "player", "ball"} else "experimental",
            "player_signal_source": signals.player_signal_source,
            "ball_signal_source": signals.ball_signal_source,
            "stride": max(1, int(stride)),
            "active_window_mode": "accepted_start_to_next_accepted_start",
            "endpoint_refine_mode": "roi_plus_ball_bounded_search",
            "excluded_let_count": len(excluded_let_starts),
            "excluded_let_starts": excluded_let_starts,
            "unattached_trailing_let_count": len(unattached_trailing_let_starts),
            "unattached_trailing_let_starts": unattached_trailing_let_starts,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate draft JSON using independent table/player/ball or multi-stream segmentation.")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--weights", default="weights/yolov8x_table.pt", help="Path to YOLO table weights")
    parser.add_argument("--pose-weights", default="weights/yolov8x-pose.pt", help="Path to YOLO pose weights")
    parser.add_argument("--out", required=True, help="Output draft JSON path")
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--mode", choices=["table", "player", "ball", "fused", "table_refined", "table_ball_refined"], default="fused")
    parser.add_argument("--player-margin-px", type=int, default=220)
    parser.add_argument("--player-fuse-gain", type=float, default=1.0)
    parser.add_argument("--player-signal-source", choices=["role_tracker", "nearest_two", "none"], default="role_tracker")
    parser.add_argument("--ball-fuse-gain", type=float, default=1.15)
    parser.add_argument("--ball-signal-source", choices=["none", "classical"], default="none")
    args = parser.parse_args()

    draft = build_draft(
        args.video,
        args.weights,
        pose_weights_path=args.pose_weights,
        best_of=args.best_of,
        stride=args.stride,
        mode=args.mode,
        player_margin_px=args.player_margin_px,
        player_fuse_gain=args.player_fuse_gain,
        player_signal_source=args.player_signal_source,
        ball_fuse_gain=args.ball_fuse_gain,
        ball_signal_source=args.ball_signal_source,
    )
    out_path = Path(args.out)
    save_draft_match(out_path, draft)
    print(f"[OK] Saved {args.mode} draft: {out_path} | total_rallies={len(draft.points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
