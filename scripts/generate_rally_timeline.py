from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))

from backend.production_defaults import PRODUCTION_RALLY_DEFAULTS
from backend.rally_timeline_contract import RallyTimeline, RallyTimelinePoint, save_rally_timeline
from backend.ai_multistream_rally import detect_multistream_rallies, extract_multistream_signals


def _probe_video_duration_sec(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps <= 0.0 or frame_count <= 0.0:
            return 0.0
        return float(frame_count / fps)
    finally:
        cap.release()


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
) -> tuple[list[RallyTimelinePoint], list[dict], list[dict]]:
    segments = sorted(segments, key=lambda seg: float(seg.t_start))
    points: List[RallyTimelinePoint] = []
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
            RallyTimelinePoint(
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


def _compute_search_upper_bound(points: List[RallyTimelinePoint], idx: int, video_end_sec: float) -> float:
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
    reach_a = _signal_array(list(getattr(signals, "player_a_reach_scores", [])), sample_count)
    reach_b = _signal_array(list(getattr(signals, "player_b_reach_scores", [])), sample_count)
    approach_a = _signal_array(list(getattr(signals, "player_a_net_approach_scores", [])), sample_count)
    approach_b = _signal_array(list(getattr(signals, "player_b_net_approach_scores", [])), sample_count)
    face_hidden_a = _signal_array(list(getattr(signals, "player_a_face_hidden_scores", [])), sample_count)
    face_hidden_b = _signal_array(list(getattr(signals, "player_b_face_hidden_scores", [])), sample_count)
    face_touch_a = _signal_array(list(getattr(signals, "player_a_face_touch_scores", [])), sample_count)
    face_touch_b = _signal_array(list(getattr(signals, "player_b_face_touch_scores", [])), sample_count)

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
    action_a = np.maximum.reduce(
        [
            competitive_a,
            0.68 * reach_a,
            0.58 * serve_a,
            0.42 * approach_a,
        ]
    )
    action_b = np.maximum.reduce(
        [
            competitive_b,
            0.68 * reach_b,
            0.58 * serve_b,
            0.42 * approach_b,
        ]
    )
    live_pair = np.maximum(competitive_a, competitive_b)
    interaction_pair = np.sqrt(np.clip(competitive_a, 0.0, None) * np.clip(competitive_b, 0.0, None))
    one_sided_motion = np.clip(live_pair - interaction_pair, 0.0, 1.0)

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
    terminal_body_a = np.maximum.reduce([face_hidden_a, 0.92 * face_touch_a, 0.35 * casual_a])
    terminal_body_b = np.maximum.reduce([face_hidden_b, 0.92 * face_touch_b, 0.35 * casual_b])
    partner_reset_a = np.maximum(casual_b, stand_b)
    partner_reset_b = np.maximum(casual_a, stand_a)
    terminal_body_pair = np.maximum.reduce(
        [
            np.minimum(terminal_body_a, terminal_body_b),
            np.minimum(terminal_body_a, partner_reset_a),
            np.minimum(terminal_body_b, partner_reset_b),
        ]
    )

    return {
        "table_norm": table_norm,
        "ball_norm": ball_norm,
        "competitive_a": competitive_a,
        "competitive_b": competitive_b,
        "action_a": action_a,
        "action_b": action_b,
        "live_pair": live_pair,
        "interaction_pair": interaction_pair,
        "one_sided_motion": one_sided_motion,
        "reset_pair": reset_pair,
        "shared_activity": shared_activity,
        "terminal_body_pair": terminal_body_pair,
    }


def _refine_endpoint_from_signals(
    timestamps: np.ndarray,
    table_norm: np.ndarray,
    ball_norm: np.ndarray,
    live_pair: np.ndarray,
    interaction_pair: np.ndarray,
    one_sided_motion: np.ndarray,
    reset_pair: np.ndarray,
    shared_activity: np.ndarray,
    terminal_body_pair: np.ndarray,
    *,
    t_start: float,
    detector_end: float,
    search_upper_bound: float,
    is_open_tail: bool = False,
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
    sample_dt = float(np.median(np.diff(interval_times))) if interval_times.size > 1 else (1.0 / 30.0)
    interval_live = live_pair[start_idx : upper_idx + 1]
    interval_interaction = interaction_pair[start_idx : upper_idx + 1]
    interval_one_sided = one_sided_motion[start_idx : upper_idx + 1]
    interval_reset = reset_pair[start_idx : upper_idx + 1]
    interval_shared = shared_activity[start_idx : upper_idx + 1]
    interval_terminal_body = terminal_body_pair[start_idx : upper_idx + 1]
    interaction_discount = np.clip(1.0 - (0.60 * interval_terminal_body), 0.35, 1.0)
    effective_interaction = interval_interaction * interaction_discount
    combined_live = np.maximum.reduce(
        [
            (0.44 * interval_ball) + (0.34 * interval_table) + (0.22 * interval_live),
            np.minimum(interval_ball * 1.05, np.maximum(interval_table, 0.85 * interval_live)),
            np.minimum(interval_live * 1.08, np.maximum(interval_ball, 0.70 * interval_table)),
        ]
    )
    no_ball = np.clip(1.0 - interval_ball, 0.0, 1.0)
    no_live = np.clip(1.0 - interval_live, 0.0, 1.0)
    no_table = np.clip(1.0 - interval_table, 0.0, 1.0)
    terminal_reset_score = np.maximum.reduce(
        [
            (0.42 * no_ball) + (0.24 * interval_reset) + (0.16 * no_live) + (0.18 * (1.0 - interval_interaction)),
            (0.36 * no_ball) + (0.22 * interval_reset) + (0.16 * no_table) + (0.16 * (1.0 - effective_interaction)) + (0.10 * interval_one_sided),
            (0.36 * no_ball) + (0.20 * interval_reset) + (0.12 * no_live) + (0.16 * no_table) + (0.16 * (1.0 - effective_interaction)),
            (0.26 * no_ball) + (0.20 * interval_reset) + (0.12 * no_live) + (0.12 * no_table) + (0.12 * (1.0 - effective_interaction)) + (0.18 * interval_terminal_body),
        ]
    )

    exchange_mask = (
        ((interval_ball >= 0.18) & ((interval_table >= 0.16) | (effective_interaction >= 0.12) | ((interval_live >= 0.24) & (interval_reset <= 0.62))))
        | ((effective_interaction >= 0.16) & (interval_live >= 0.24) & (interval_ball >= 0.08))
        | ((combined_live >= 0.32) & (interval_ball >= 0.12) & (effective_interaction >= 0.10))
    )
    exchange_mask = _fill_short_false_gaps(exchange_mask, max_gap_samples=1)
    competitive_exchange_mask = (
        ((interval_ball >= 0.20) & (effective_interaction >= 0.12) & (interval_live >= 0.22) & (interval_reset <= 0.68))
        | ((combined_live >= 0.38) & (interval_ball >= 0.18) & (effective_interaction >= 0.16) & (interval_reset <= 0.64))
        | ((interval_ball >= 0.50) & (effective_interaction >= 0.08) & (interval_table >= 0.16) & (interval_reset <= 0.68))
    )
    competitive_exchange_mask = _fill_short_false_gaps(competitive_exchange_mask, max_gap_samples=1)
    terminal_body_mask = ((interval_terminal_body >= 0.34) & (interval_reset >= 0.44))
    terminal_body_mask = _fill_short_false_gaps(terminal_body_mask, max_gap_samples=1)
    prior_exchange_peak = np.maximum.accumulate(np.where(competitive_exchange_mask, combined_live, 0.0))

    dead_reset_mask = (
        (interval_ball <= 0.24)
        & (
            ((terminal_reset_score >= 0.62) & (effective_interaction <= 0.16))
            | ((interval_ball <= 0.16) & (interval_reset >= 0.60) & (effective_interaction <= 0.16))
            | ((interval_ball <= 0.30) & (interval_reset >= 0.52) & (effective_interaction <= 0.10) & (interval_one_sided >= 0.12))
            | ((interval_ball <= 0.22) & (interval_reset >= 0.66) & (interval_one_sided >= 0.18))
            | ((interval_reset >= 0.74) & (interval_shared <= 0.52) & (effective_interaction <= 0.20))
            | ((interval_terminal_body >= 0.46) & (interval_reset >= 0.40) & (effective_interaction <= 0.24) & (combined_live <= 0.62))
        )
    )
    ball_only_false_tail_mask = (
        (interval_ball >= 0.52)
        & (terminal_reset_score >= 0.66)
        & (interval_table <= 0.08)
        & (interval_live <= 0.14)
        & (effective_interaction <= 0.06)
        & (interval_reset >= 0.72)
        & (interval_shared <= 0.30)
        & (interval_terminal_body >= 0.58)
    )
    dead_gap_samples = max(1, int(round(0.20 / max(sample_dt, 1e-6))))
    dead_reset_mask = _fill_short_false_gaps(dead_reset_mask, max_gap_samples=dead_gap_samples)
    interval_duration = float(max(1e-6, safe_upper - safe_start))
    earliest_dead_time = safe_start + min(1.05, 0.45 * interval_duration)
    future_guard_t = safe_upper - min(0.95, max(0.55, 0.18 * interval_duration))
    future_guard_idx = int(np.searchsorted(interval_times, future_guard_t, side="left")) - 1
    def extract_runs(mask: np.ndarray) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        run_start_idx: int | None = None
        for idx_local, value in enumerate(mask):
            if value and run_start_idx is None:
                run_start_idx = idx_local
                continue
            if (not value) and run_start_idx is not None:
                runs.append((run_start_idx, idx_local - 1))
                run_start_idx = None
        if run_start_idx is not None:
            runs.append((run_start_idx, len(mask) - 1))
        return runs

    def run_duration_sec(run_start_idx: int, run_end_idx: int) -> float:
        return float(interval_times[run_end_idx] - interval_times[run_start_idx] + sample_dt)

    competitive_runs = extract_runs(competitive_exchange_mask)
    exchange_runs = extract_runs(exchange_mask)
    dead_runs = extract_runs(dead_reset_mask)
    terminal_body_runs = extract_runs(terminal_body_mask)
    ball_only_false_tail_runs = extract_runs(
        _fill_short_false_gaps(ball_only_false_tail_mask, max_gap_samples=max(1, int(round(0.10 / max(sample_dt, 1e-6)))))
    )

    def run_mean(arr: np.ndarray, run_start_idx: int, run_end_idx: int) -> float:
        return float(np.mean(arr[run_start_idx : run_end_idx + 1]))

    def run_peak(arr: np.ndarray, run_start_idx: int, run_end_idx: int) -> float:
        return float(np.max(arr[run_start_idx : run_end_idx + 1]))

    def slice_runs_between(
        runs: list[tuple[int, int]],
        *,
        after_idx: int,
        before_idx: int,
    ) -> list[tuple[int, int]]:
        clipped_runs: list[tuple[int, int]] = []
        for run_start_idx, run_end_idx in runs:
            if run_end_idx <= after_idx:
                continue
            if run_start_idx >= before_idx:
                break
            clipped_start_idx = max(run_start_idx, after_idx + 1)
            clipped_end_idx = min(run_end_idx, before_idx - 1)
            if clipped_start_idx <= clipped_end_idx:
                clipped_runs.append((clipped_start_idx, clipped_end_idx))
        return clipped_runs

    def runs_total_duration_sec(runs: list[tuple[int, int]]) -> float:
        return float(sum(run_duration_sec(run_start_idx, run_end_idx) for run_start_idx, run_end_idx in runs))

    def runs_span_sec(runs: list[tuple[int, int]]) -> float:
        if not runs:
            return 0.0
        first_start_idx = runs[0][0]
        last_end_idx = runs[-1][1]
        return float(interval_times[last_end_idx] - interval_times[first_start_idx] + sample_dt)

    def is_weak_tail_fragment(run_start_idx: int, run_end_idx: int) -> bool:
        run_duration_value = run_duration_sec(run_start_idx, run_end_idx)
        mean_table = run_mean(interval_table, run_start_idx, run_end_idx)
        mean_interaction = run_mean(interval_interaction, run_start_idx, run_end_idx)
        mean_reset = run_mean(interval_reset, run_start_idx, run_end_idx)
        peak_interaction = run_peak(interval_interaction, run_start_idx, run_end_idx)
        term_start = float(terminal_reset_score[run_start_idx])
        micro_fragment = bool(run_duration_value <= max(0.12, 3.5 * sample_dt))
        return bool(
            run_duration_value <= max(1.15, 36.0 * sample_dt)
            and mean_reset >= 0.50
            and (
                mean_interaction <= 0.34
                or (
                    micro_fragment
                    and term_start >= 0.50
                    and mean_reset >= 0.40
                    and peak_interaction <= 0.65
                )
            )
            and (
                mean_table <= 0.28
                or term_start >= 0.64
                or (micro_fragment and term_start >= 0.50)
            )
        )

    def is_long_gap_pseudo_resume_fragment(
        run_start_idx: int,
        run_end_idx: int,
        *,
        anchor_t: float,
    ) -> bool:
        run_start_t = float(interval_times[run_start_idx])
        quiet_gap_sec = run_start_t - anchor_t
        if quiet_gap_sec < max(1.80, 0.18 * interval_duration):
            return False
        run_duration_value = run_duration_sec(run_start_idx, run_end_idx)
        mean_table = run_mean(interval_table, run_start_idx, run_end_idx)
        mean_live = run_mean(interval_live, run_start_idx, run_end_idx)
        mean_effective_interaction = run_mean(effective_interaction, run_start_idx, run_end_idx)
        mean_reset = run_mean(interval_reset, run_start_idx, run_end_idx)
        mean_shared = run_mean(interval_shared, run_start_idx, run_end_idx)
        return bool(
            run_duration_value <= max(1.25, 38.0 * sample_dt)
            and mean_table <= 0.16
            and mean_live <= 0.34
            and mean_effective_interaction <= 0.20
            and mean_reset >= 0.56
            and mean_shared <= 0.52
        )

    def is_post_body_disengaged_fragment(run_start_idx: int, run_end_idx: int) -> bool:
        run_duration_value = run_duration_sec(run_start_idx, run_end_idx)
        mean_table = run_mean(interval_table, run_start_idx, run_end_idx)
        mean_live = run_mean(interval_live, run_start_idx, run_end_idx)
        mean_interaction = run_mean(interval_interaction, run_start_idx, run_end_idx)
        mean_effective_interaction = run_mean(effective_interaction, run_start_idx, run_end_idx)
        mean_reset = run_mean(interval_reset, run_start_idx, run_end_idx)
        mean_shared = run_mean(interval_shared, run_start_idx, run_end_idx)
        mean_terminal_body = run_mean(interval_terminal_body, run_start_idx, run_end_idx)
        peak_interaction = run_peak(interval_interaction, run_start_idx, run_end_idx)
        micro_fragment = bool(run_duration_value <= max(0.18, 5.5 * sample_dt))
        return bool(
            run_duration_value <= max(1.50, 46.0 * sample_dt)
            and (
                (
                    mean_effective_interaction <= 0.18
                    and mean_reset >= 0.58
                    and mean_terminal_body >= 0.45
                    and mean_shared <= 0.64
                )
                or (
                    micro_fragment
                    and mean_live <= 0.30
                    and peak_interaction <= 0.35
                    and mean_table <= 0.20
                    and mean_reset >= 0.40
                )
                or (
                    mean_effective_interaction <= 0.10
                    and mean_reset >= 0.60
                    and mean_table <= 0.30
                    and mean_shared <= 0.70
                )
            )
        )

    def is_strong_post_body_continuation_fragment(run_start_idx: int, run_end_idx: int) -> bool:
        run_duration_value = run_duration_sec(run_start_idx, run_end_idx)
        mean_table = run_mean(interval_table, run_start_idx, run_end_idx)
        mean_live = run_mean(interval_live, run_start_idx, run_end_idx)
        mean_effective_interaction = run_mean(effective_interaction, run_start_idx, run_end_idx)
        mean_reset = run_mean(interval_reset, run_start_idx, run_end_idx)
        peak_table = run_peak(interval_table, run_start_idx, run_end_idx)
        peak_live = run_peak(interval_live, run_start_idx, run_end_idx)
        peak_interaction = run_peak(interval_interaction, run_start_idx, run_end_idx)
        return bool(
            (
                run_duration_value >= max(0.70, 22.0 * sample_dt)
                and mean_effective_interaction >= 0.18
                and mean_live >= 0.38
                and mean_reset <= 0.58
                and peak_live >= 0.58
                and (
                    mean_table >= 0.18
                    or peak_table >= 0.30
                    or peak_interaction >= 0.36
                )
            )
            or (
                run_duration_value >= max(0.45, 12.0 * sample_dt)
                and mean_effective_interaction >= 0.20
                and mean_reset <= 0.56
                and peak_live >= 0.55
                and peak_table >= 0.58
                and peak_interaction >= 0.32
            )
        )

    def is_post_body_pseudo_live_fragment(run_start_idx: int, run_end_idx: int) -> bool:
        run_duration_value = run_duration_sec(run_start_idx, run_end_idx)
        mean_table = run_mean(interval_table, run_start_idx, run_end_idx)
        mean_live = run_mean(interval_live, run_start_idx, run_end_idx)
        mean_interaction = run_mean(interval_interaction, run_start_idx, run_end_idx)
        mean_effective_interaction = run_mean(effective_interaction, run_start_idx, run_end_idx)
        mean_reset = run_mean(interval_reset, run_start_idx, run_end_idx)
        mean_ball = run_mean(interval_ball, run_start_idx, run_end_idx)
        peak_interaction = run_peak(interval_interaction, run_start_idx, run_end_idx)
        return bool(
            run_duration_value <= max(1.40, 42.0 * sample_dt)
            and mean_ball >= 0.55
            and (
                (
                    mean_table <= 0.12
                    and mean_live >= 0.40
                    and mean_effective_interaction <= 0.38
                    and mean_interaction <= 0.48
                )
                or (
                    mean_reset >= 0.56
                    and mean_effective_interaction <= 0.30
                    and peak_interaction <= 0.55
                )
            )
        )

    first_viable_dead_start_local: int | None = None
    for dead_start_local, _dead_end_local in dead_runs:
        dead_start_t = float(interval_times[dead_start_local])
        if dead_start_t >= earliest_dead_time:
            first_viable_dead_start_local = dead_start_local
            break

    if competitive_runs and dead_runs and terminal_body_runs:
        for body_start_local, body_end_local in terminal_body_runs:
            body_start_t = float(interval_times[body_start_local])
            if body_start_t < earliest_dead_time:
                continue
            if body_start_local > future_guard_idx:
                break
            if body_start_t >= baseline_end - max(0.20, 4.0 * sample_dt):
                continue
            if first_viable_dead_start_local is not None and body_start_local >= first_viable_dead_start_local:
                continue

            body_peak = run_peak(interval_terminal_body, body_start_local, body_end_local)
            if body_peak < 0.58:
                continue

            body_mean_ball = run_mean(interval_ball, body_start_local, body_end_local)
            body_mean_table = run_mean(interval_table, body_start_local, body_end_local)
            body_mean_eff = run_mean(effective_interaction, body_start_local, body_end_local)
            if body_mean_ball >= 0.32 or body_mean_table >= 0.16 or body_mean_eff >= 0.18:
                continue

            prior_peak = float(prior_exchange_peak[body_start_local - 1]) if body_start_local > 0 else 0.0
            if prior_peak < 0.40:
                continue

            if (
                effective_interaction[body_start_local] > 0.34
                and interval_one_sided[body_start_local] < 0.24
            ):
                continue

            if combined_live[body_start_local] > 0.80 and interval_reset[body_start_local] < 0.32:
                continue

            next_dead_after_body: tuple[int, int] | None = None
            for dead_start_local, dead_end_local in dead_runs:
                if dead_start_local > body_start_local:
                    next_dead_after_body = (dead_start_local, dead_end_local)
                    break
            if next_dead_after_body is None:
                continue

            next_dead_start_local, _ = next_dead_after_body
            next_dead_start_t = float(interval_times[next_dead_start_local])
            if next_dead_start_t - body_start_t < max(1.10, 0.11 * interval_duration):
                continue

            weak_tail_only = True
            for run_start_local, run_end_local in competitive_runs:
                if run_end_local <= body_start_local:
                    continue
                if run_start_local >= next_dead_start_local:
                    break
                tail_run_start = max(run_start_local, body_start_local)
                if (
                    is_strong_post_body_continuation_fragment(tail_run_start, run_end_local)
                    or not is_weak_tail_fragment(tail_run_start, run_end_local)
                    and not is_long_gap_pseudo_resume_fragment(
                        tail_run_start,
                        run_end_local,
                        anchor_t=body_start_t,
                    )
                    and not is_post_body_disengaged_fragment(tail_run_start, run_end_local)
                ):
                    weak_tail_only = False
                    break

            if weak_tail_only:
                refined_end = float(np.clip(body_start_t, safe_start + 0.01, baseline_end))
                endpoint_confidence = float(
                    np.clip(
                        0.44
                        + (0.16 * body_peak)
                        + (0.10 * float(interval_reset[body_start_local]))
                        + (0.10 * min(1.0, (next_dead_start_t - body_start_t) / max(1.10, 0.11 * interval_duration))),
                        0.34,
                        0.94,
                    )
                )
                return refined_end, "terminal_body_split_start", endpoint_confidence

    if competitive_runs and dead_runs and terminal_body_runs:
        for body_start_local, body_end_local in terminal_body_runs:
            body_start_t = float(interval_times[body_start_local])
            if body_start_t < earliest_dead_time:
                continue
            if body_start_local > future_guard_idx:
                break
            if body_start_t >= baseline_end - max(0.20, 4.0 * sample_dt):
                continue

            body_duration_value = run_duration_sec(body_start_local, body_end_local)
            body_mean_term = run_mean(interval_terminal_body, body_start_local, body_end_local)
            body_mean_reset = run_mean(interval_reset, body_start_local, body_end_local)
            body_mean_eff = run_mean(effective_interaction, body_start_local, body_end_local)
            body_mean_ball = run_mean(interval_ball, body_start_local, body_end_local)
            body_mean_table = run_mean(interval_table, body_start_local, body_end_local)
            if body_duration_value < max(0.55, 16.0 * sample_dt):
                continue
            if body_mean_term < 0.34 or body_mean_reset < 0.60 or body_mean_eff > 0.10:
                continue
            if body_mean_ball < 0.60 or body_mean_table > 0.32:
                continue

            prior_peak = float(prior_exchange_peak[body_start_local - 1]) if body_start_local > 0 else 0.0
            if prior_peak < 0.48:
                continue

            next_dead_after_body: tuple[int, int] | None = None
            for dead_start_local, dead_end_local in dead_runs:
                if dead_start_local > body_start_local:
                    next_dead_after_body = (dead_start_local, dead_end_local)
                    break
            if next_dead_after_body is None:
                continue

            next_dead_start_local, _ = next_dead_after_body
            next_dead_start_t = float(interval_times[next_dead_start_local])
            if next_dead_start_t - body_start_t < max(1.15, 0.14 * interval_duration):
                continue

            saw_tail_run = False
            saw_table_dominant_pseudo_tail = False
            pseudo_tail_only = True
            for run_start_local, run_end_local in competitive_runs:
                if run_end_local <= body_start_local:
                    continue
                if run_start_local >= next_dead_start_local:
                    break
                tail_run_start = max(run_start_local, body_end_local + 1)
                if tail_run_start > run_end_local:
                    continue
                saw_tail_run = True
                tail_mean_table = run_mean(interval_table, tail_run_start, run_end_local)
                if tail_mean_table >= 0.24:
                    saw_table_dominant_pseudo_tail = True
                if (
                    is_strong_post_body_continuation_fragment(tail_run_start, run_end_local)
                    or not is_post_body_pseudo_live_fragment(tail_run_start, run_end_local)
                ):
                    pseudo_tail_only = False
                    break

            if saw_tail_run and pseudo_tail_only and saw_table_dominant_pseudo_tail:
                refined_end = float(np.clip(body_start_t, safe_start + 0.01, baseline_end))
                endpoint_confidence = float(
                    np.clip(
                        0.46
                        + (0.10 * body_mean_term)
                        + (0.12 * body_mean_reset)
                        + (0.10 * min(1.0, body_duration_value / max(0.55, 16.0 * sample_dt)))
                        + (0.08 * prior_peak),
                        0.36,
                        0.94,
                    )
                )
                return refined_end, "post_body_pseudo_live_start", endpoint_confidence

    if competitive_runs and dead_runs and terminal_body_runs:
        for body_start_local, body_end_local in terminal_body_runs:
            body_start_t = float(interval_times[body_start_local])
            if body_start_t < earliest_dead_time:
                continue
            if body_start_local > future_guard_idx:
                break
            if body_start_t >= baseline_end - max(0.20, 4.0 * sample_dt):
                continue

            body_duration_value = run_duration_sec(body_start_local, body_end_local)
            body_mean_term = run_mean(interval_terminal_body, body_start_local, body_end_local)
            body_mean_reset = run_mean(interval_reset, body_start_local, body_end_local)
            body_mean_eff = run_mean(effective_interaction, body_start_local, body_end_local)
            body_mean_ball = run_mean(interval_ball, body_start_local, body_end_local)
            body_mean_table = run_mean(interval_table, body_start_local, body_end_local)
            if body_duration_value < max(1.20, 32.0 * sample_dt):
                continue
            if body_mean_term < 0.34 or body_mean_reset < 0.60 or body_mean_eff > 0.08:
                continue
            if body_mean_ball > 0.28 or body_mean_table > 0.14:
                continue

            prior_peak = float(prior_exchange_peak[body_start_local - 1]) if body_start_local > 0 else 0.0
            if prior_peak < 0.48:
                continue

            next_dead_after_body: tuple[int, int] | None = None
            for dead_start_local, dead_end_local in dead_runs:
                if dead_start_local > body_start_local:
                    next_dead_after_body = (dead_start_local, dead_end_local)
                    break
            if next_dead_after_body is None:
                continue

            next_dead_start_local, _ = next_dead_after_body
            next_dead_start_t = float(interval_times[next_dead_start_local])
            if next_dead_start_t - body_start_t < max(1.60, 0.18 * interval_duration):
                continue

            saw_tail_run = False
            pseudo_tail_only = True
            for run_start_local, run_end_local in competitive_runs:
                if run_end_local <= body_start_local:
                    continue
                if run_start_local >= next_dead_start_local:
                    break
                tail_run_start = max(run_start_local, body_end_local + 1)
                if tail_run_start > run_end_local:
                    continue
                saw_tail_run = True
                tail_duration_value = run_duration_sec(tail_run_start, run_end_local)
                tail_mean_table = run_mean(interval_table, tail_run_start, run_end_local)
                tail_mean_ball = run_mean(interval_ball, tail_run_start, run_end_local)
                tail_mean_eff = run_mean(effective_interaction, tail_run_start, run_end_local)
                if not (
                    is_weak_tail_fragment(tail_run_start, run_end_local)
                    or is_post_body_disengaged_fragment(tail_run_start, run_end_local)
                    or (
                        tail_duration_value <= max(0.12, 4.0 * sample_dt)
                        and tail_mean_table <= 0.18
                        and tail_mean_ball <= 0.32
                    )
                    or (
                        tail_duration_value <= max(0.45, 14.0 * sample_dt)
                        and tail_mean_table <= 0.12
                        and tail_mean_eff <= 0.18
                        and tail_mean_ball <= 0.45
                    )
                ):
                    pseudo_tail_only = False
                    break

            if saw_tail_run and pseudo_tail_only:
                refined_end = float(np.clip(body_start_t, safe_start + 0.01, baseline_end))
                endpoint_confidence = float(
                    np.clip(
                        0.48
                        + (0.10 * body_mean_term)
                        + (0.12 * body_mean_reset)
                        + (0.10 * min(1.0, body_duration_value / max(1.20, 32.0 * sample_dt)))
                        + (0.08 * prior_peak),
                        0.38,
                        0.95,
                    )
                )
                return refined_end, "post_dead_plateau_start", endpoint_confidence

    if competitive_runs and ball_only_false_tail_runs:
        for tail_start_local, tail_end_local in ball_only_false_tail_runs:
            tail_start_t = float(interval_times[tail_start_local])
            if tail_start_t < earliest_dead_time:
                continue
            if tail_start_local > future_guard_idx:
                break

            prior_peak = float(prior_exchange_peak[tail_start_local - 1]) if tail_start_local > 0 else 0.0
            if prior_peak < 0.30:
                continue

            next_competitive_run: tuple[int, int] | None = None
            for run_start_local, run_end_local in competitive_runs:
                if run_start_local <= tail_end_local:
                    continue
                next_competitive_run = (run_start_local, run_end_local)
                break

            if next_competitive_run is None:
                continue

            if not is_long_gap_pseudo_resume_fragment(
                next_competitive_run[0],
                next_competitive_run[1],
                anchor_t=tail_start_t,
            ):
                continue

            refined_end = float(np.clip(tail_start_t, safe_start + 0.01, baseline_end))
            endpoint_confidence = float(
                np.clip(
                    0.48
                    + (0.12 * float(terminal_reset_score[tail_start_local]))
                    + (0.10 * float(interval_terminal_body[tail_start_local]))
                    + (0.08 * float(interval_reset[tail_start_local])),
                    0.36,
                    0.92,
                )
            )
            return refined_end, "ball_only_false_tail_start", endpoint_confidence

    if competitive_runs:
        primary_start_local, primary_end_local = competitive_runs[0]
        primary_duration = run_duration_sec(primary_start_local, primary_end_local)
        if primary_duration >= max(0.75, 6.0 * sample_dt):
            next_dead_after_primary: tuple[int, int] | None = None
            for dead_start_local, dead_end_local in dead_runs:
                if dead_start_local > primary_end_local:
                    next_dead_after_primary = (dead_start_local, dead_end_local)
                    break

            if next_dead_after_primary is not None:
                primary_end_t = float(interval_times[primary_end_local])
                next_dead_start_local, _ = next_dead_after_primary
                next_dead_start_t = float(interval_times[next_dead_start_local])
                long_tail_after_primary = bool(
                    next_dead_start_t - primary_end_t >= max(1.80, 0.18 * interval_duration)
                )
                if long_tail_after_primary:
                    seed_run: tuple[int, int] | None = None
                    seed_search_lead_sec = max(0.35, 10.0 * sample_dt)
                    seed_start_floor_t = primary_end_t - max(0.25, 7.0 * sample_dt)
                    for body_start_local, body_end_local in terminal_body_runs:
                        body_start_t = float(interval_times[body_start_local])
                        if body_start_t < seed_start_floor_t:
                            continue
                        if body_start_t - primary_end_t > seed_search_lead_sec:
                            break
                        if run_duration_sec(body_start_local, body_end_local) < max(0.20, 4.0 * sample_dt):
                            continue
                        seed_run = (body_start_local, body_end_local)
                        break

                    if seed_run is not None:
                        weak_tail_only = True
                        for run_start_local, run_end_local in competitive_runs[1:]:
                            if run_start_local < seed_run[0]:
                                continue
                            if run_start_local >= next_dead_start_local:
                                break
                            if not is_weak_tail_fragment(run_start_local, run_end_local):
                                weak_tail_only = False
                                break
                        if weak_tail_only:
                            seed_start_t = float(interval_times[seed_run[0]])
                            refined_end = float(
                                np.clip(max(seed_start_t, seed_start_floor_t), safe_start + 0.01, baseline_end)
                            )
                            endpoint_confidence = float(
                                np.clip(
                                    0.48
                                    + (0.12 * float(run_mean(interval_terminal_body, seed_run[0], seed_run[1])))
                                    + (0.08 * float(run_mean(interval_reset, seed_run[0], seed_run[1])))
                                    + (0.08 * min(1.0, (next_dead_start_t - primary_end_t) / max(1.80, 0.18 * interval_duration))),
                                    0.36,
                                    0.94,
                                )
                            )
                            return refined_end, "terminal_body_seed_start", endpoint_confidence

                cluster_runs: list[tuple[int, int]] = []
                cluster_start_t: float | None = None
                cluster_end_t: float | None = None
                gap_limit_sec = max(0.40, 12.0 * sample_dt)
                weak_span_min_sec = max(1.00, 0.11 * interval_duration)
                weak_dead_gap_limit_sec = max(0.70, 16.0 * sample_dt)

                prev_end_local = primary_end_local
                for run_start_local, run_end_local in competitive_runs[1:]:
                    if run_start_local >= next_dead_start_local:
                        break
                    gap_before_run = float(interval_times[run_start_local] - interval_times[prev_end_local])
                    prev_end_local = run_end_local
                    if gap_before_run > gap_limit_sec:
                        cluster_runs = []
                        break
                    if not is_weak_tail_fragment(run_start_local, run_end_local):
                        cluster_runs = []
                        break
                    cluster_runs.append((run_start_local, run_end_local))
                    if cluster_start_t is None:
                        cluster_start_t = float(interval_times[run_start_local])
                    cluster_end_t = float(interval_times[run_end_local])

                if cluster_runs and cluster_start_t is not None and cluster_end_t is not None:
                    cluster_span = float(cluster_end_t - cluster_start_t + sample_dt)
                    dead_gap_after_cluster = float(next_dead_start_t - cluster_end_t)
                    if (
                        cluster_span >= weak_span_min_sec
                        and dead_gap_after_cluster <= weak_dead_gap_limit_sec
                        and cluster_start_t > float(interval_times[primary_end_local])
                    ):
                        refined_end = float(
                            np.clip(cluster_start_t, safe_start + 0.01, baseline_end)
                        )
                        endpoint_confidence = float(
                            np.clip(
                                0.46
                                + (0.10 * min(1.0, cluster_span / max(weak_span_min_sec, 1e-6)))
                                + (0.08 * float(1.0 - run_mean(interval_interaction, cluster_runs[0][0], cluster_runs[-1][1])))
                                + (0.08 * float(run_mean(interval_reset, cluster_runs[0][0], cluster_runs[-1][1]))),
                                0.34,
                                0.92,
                            )
                        )
                        return refined_end, "weak_tail_cluster_start", endpoint_confidence

    resume_min_sec = max(0.28, 3.0 * sample_dt)
    for dead_run_idx, (dead_start_local, dead_end_local) in enumerate(dead_runs):
        dead_start_t = float(interval_times[dead_start_local])
        dead_end_t = float(interval_times[dead_end_local])
        if dead_start_t < earliest_dead_time:
            continue
        if dead_start_t <= safe_start + 0.10:
            continue

        prior_peak = float(prior_exchange_peak[dead_start_local - 1]) if dead_start_local > 0 else 0.0
        if prior_peak < 0.30:
            continue

        min_dead_duration = max(0.09, 2.0 * sample_dt)
        if terminal_reset_score[dead_start_local] < 0.82:
            min_dead_duration = max(min_dead_duration, 0.15)
        dead_duration_value = run_duration_sec(dead_start_local, dead_end_local)
        if dead_duration_value < min_dead_duration:
            bridged_dead = False
            if (
                dead_run_idx + 1 < len(dead_runs)
                and dead_duration_value >= max(0.05, 1.5 * sample_dt)
                and terminal_reset_score[dead_start_local] >= 0.76
                and interval_ball[dead_start_local] <= 0.24
                and interval_interaction[dead_start_local] <= 0.16
            ):
                next_dead_start_local, _ = dead_runs[dead_run_idx + 1]
                next_dead_gap = float(interval_times[next_dead_start_local] - dead_end_t)
                if next_dead_gap <= max(0.75, 8.0 * sample_dt):
                    bridged_dead = True
            if not bridged_dead:
                continue

        strong_dead = bool(
            terminal_reset_score[dead_start_local] >= 0.80
            and interval_ball[dead_start_local] <= 0.24
            and interval_interaction[dead_start_local] <= 0.20
        )
        terminal_disengaged_dead = bool(
            interval_terminal_body[dead_start_local] >= 0.34
            and interval_reset[dead_start_local] >= 0.48
        )
        if strong_dead:
            resume_horizon_sec = min(1.15, max(0.80, 0.11 * interval_duration))
            resume_duration_sec = max(resume_min_sec, 0.38)
            resume_peak_threshold = 0.44
            resume_ball_peak_threshold = 0.24
            resume_live_peak_threshold = 0.26
            resume_interaction_peak_threshold = 0.20
            resume_mean_interaction_threshold = 0.22
            resume_mean_ball_threshold = 0.22
            resume_reset_mean_threshold = 0.52
        else:
            resume_horizon_sec = min(1.75, max(1.05, 0.18 * interval_duration))
            resume_duration_sec = resume_min_sec
            resume_peak_threshold = 0.38
            resume_ball_peak_threshold = 0.20
            resume_live_peak_threshold = 0.22
            resume_interaction_peak_threshold = 0.16
            resume_mean_interaction_threshold = 0.16
            resume_mean_ball_threshold = 0.14
            resume_reset_mean_threshold = 0.62

        embedded_exchange_tail_end_local: int | None = None
        overlap_horizon_sec = max(1.00, 24.0 * sample_dt)
        for run_start_local, run_end_local in exchange_runs:
            if run_end_local <= dead_start_local:
                continue
            run_start_t = float(interval_times[run_start_local])
            if run_start_t - dead_start_t > overlap_horizon_sec:
                break
            overlap_start_local = max(run_start_local, dead_start_local)
            overlap_duration_value = run_duration_sec(overlap_start_local, run_end_local)
            if (
                overlap_duration_value >= max(sample_dt, 0.03)
                and run_peak(interval_live, overlap_start_local, run_end_local) >= 0.20
                and (
                    run_peak(interval_ball, overlap_start_local, run_end_local) >= 0.18
                    or run_peak(interval_table, overlap_start_local, run_end_local) >= 0.12
                )
                and run_mean(interval_reset, overlap_start_local, run_end_local) <= 0.74
            ):
                embedded_exchange_tail_end_local = run_end_local

        resume_found = False
        for run_start_local, run_end_local in competitive_runs:
            if run_start_local <= dead_end_local:
                continue
            run_start_t = float(interval_times[run_start_local])
            if run_start_t - dead_end_t > resume_horizon_sec:
                break
            if run_start_local > future_guard_idx:
                break
            future_peak = float(np.max(combined_live[run_start_local : run_end_local + 1]))
            future_ball_peak = float(np.max(interval_ball[run_start_local : run_end_local + 1]))
            future_live_peak = float(np.max(interval_live[run_start_local : run_end_local + 1]))
            future_interaction_peak = float(np.max(interval_interaction[run_start_local : run_end_local + 1]))
            future_effective_interaction_peak = float(np.max(effective_interaction[run_start_local : run_end_local + 1]))
            future_ball_mean = float(np.mean(interval_ball[run_start_local : run_end_local + 1]))
            future_table_mean = float(np.mean(interval_table[run_start_local : run_end_local + 1]))
            future_interaction_mean = float(np.mean(interval_interaction[run_start_local : run_end_local + 1]))
            future_effective_interaction_mean = float(np.mean(effective_interaction[run_start_local : run_end_local + 1]))
            future_reset_mean = float(np.mean(interval_reset[run_start_local : run_end_local + 1]))
            future_one_sided_mean = float(np.mean(interval_one_sided[run_start_local : run_end_local + 1]))
            future_shared_mean = float(np.mean(interval_shared[run_start_local : run_end_local + 1]))
            run_duration_value = run_duration_sec(run_start_local, run_end_local)
            if (
                terminal_disengaged_dead
                and run_duration_value >= max(0.30, 4.0 * sample_dt)
                and future_ball_mean >= 0.34
                and future_table_mean <= 0.18
                and future_effective_interaction_mean <= 0.12
                and future_shared_mean <= 0.34
                and future_reset_mean >= 0.52
            ):
                continue
            if (
                run_duration_value <= max(0.16, 2.5 * sample_dt)
                and future_interaction_mean <= 0.20
                and future_ball_mean <= 0.32
                and future_reset_mean >= 0.60
                and future_one_sided_mean <= 0.22
            ):
                continue
            if (
                strong_dead
                and run_duration_value >= max(0.28, 4.0 * sample_dt)
                and future_ball_peak >= 0.30
                and future_table_mean >= 0.42
                and future_live_peak >= 0.50
                and future_effective_interaction_mean >= 0.18
                and future_reset_mean <= 0.58
            ):
                resume_found = True
                break
            if (
                run_duration_value >= resume_duration_sec
                and future_peak >= resume_peak_threshold
                and future_ball_peak >= resume_ball_peak_threshold
                and future_live_peak >= resume_live_peak_threshold
                and future_interaction_peak >= resume_interaction_peak_threshold
                and future_ball_mean >= resume_mean_ball_threshold
                and future_interaction_mean >= resume_mean_interaction_threshold
                and future_reset_mean <= resume_reset_mean_threshold
                and (
                    (not is_open_tail)
                    or dead_run_idx == 0
                    or (
                        future_effective_interaction_peak >= max(0.26, resume_interaction_peak_threshold)
                        and future_effective_interaction_mean >= max(0.22, resume_mean_interaction_threshold)
                    )
                )
            ):
                resume_found = True
                break

        if not resume_found:
            player_resume_horizon_sec = min(3.20, max(2.60, 0.30 * interval_duration))
            hint_horizon_sec = max(0.75, 10.0 * sample_dt)
            hint_count = 0
            hint_total_duration = 0.0
            for run_start_local, run_end_local in competitive_runs:
                if run_start_local <= dead_end_local:
                    continue
                run_start_t = float(interval_times[run_start_local])
                if run_start_t - dead_end_t > hint_horizon_sec:
                    break
                if run_start_local > future_guard_idx:
                    break
                future_ball_peak = float(np.max(interval_ball[run_start_local : run_end_local + 1]))
                future_live_peak = float(np.max(interval_live[run_start_local : run_end_local + 1]))
                future_interaction_peak = float(np.max(interval_interaction[run_start_local : run_end_local + 1]))
                future_reset_mean = float(np.mean(interval_reset[run_start_local : run_end_local + 1]))
                run_duration_value = run_duration_sec(run_start_local, run_end_local)
                if (
                    run_duration_value >= max(0.08, 2.0 * sample_dt)
                    and future_ball_peak >= 0.15
                    and future_live_peak >= 0.20
                    and future_interaction_peak >= 0.12
                    and future_reset_mean <= 0.78
                ):
                    hint_count += 1
                    hint_total_duration += run_duration_value

            if hint_count < 1 and hint_total_duration < max(0.10, 3.0 * sample_dt):
                hint_count = 0

            if hint_count >= 1 or hint_total_duration >= max(0.10, 3.0 * sample_dt):
                for run_start_local, run_end_local in competitive_runs:
                    if run_start_local <= dead_end_local:
                        continue
                    run_start_t = float(interval_times[run_start_local])
                    if run_start_t - dead_end_t > player_resume_horizon_sec:
                        break
                    if run_start_local > future_guard_idx:
                        break
                    future_ball_peak = float(np.max(interval_ball[run_start_local : run_end_local + 1]))
                    future_live_peak = float(np.max(interval_live[run_start_local : run_end_local + 1]))
                    future_interaction_peak = float(np.max(interval_interaction[run_start_local : run_end_local + 1]))
                    future_interaction_mean = float(np.mean(interval_interaction[run_start_local : run_end_local + 1]))
                    future_reset_mean = float(np.mean(interval_reset[run_start_local : run_end_local + 1]))
                    run_duration_value = run_duration_sec(run_start_local, run_end_local)
                    intervening_dead_count = 0
                    for next_dead_start_local, next_dead_end_local in dead_runs:
                        if next_dead_start_local <= dead_end_local:
                            continue
                        if next_dead_end_local >= run_start_local:
                            break
                        intervening_dead_count += 1
                    if (
                        run_duration_value >= max(0.45, 7.0 * sample_dt)
                        and future_ball_peak >= 0.15
                        and future_live_peak >= 0.62
                        and future_interaction_peak >= 0.40
                        and future_interaction_mean >= 0.32
                        and future_reset_mean <= 0.64
                        and intervening_dead_count >= 1
                    ):
                        resume_found = True
                        break
        if (not resume_found) and is_open_tail:
            tail_resume_duration_min = 0.90
            for run_start_local, run_end_local in competitive_runs:
                if run_start_local <= dead_end_local:
                    continue
                run_start_t = float(interval_times[run_start_local])
                future_ball_mean = float(np.mean(interval_ball[run_start_local : run_end_local + 1]))
                future_live_peak = float(np.max(interval_live[run_start_local : run_end_local + 1]))
                future_effective_interaction_mean = float(np.mean(effective_interaction[run_start_local : run_end_local + 1]))
                future_reset_mean = float(np.mean(interval_reset[run_start_local : run_end_local + 1]))
                run_duration_value = run_duration_sec(run_start_local, run_end_local)
                if (
                    run_start_t - dead_end_t >= max(0.55, 8.0 * sample_dt)
                    and run_duration_value >= tail_resume_duration_min
                    and future_ball_mean >= 0.45
                    and future_live_peak >= 0.55
                    and future_effective_interaction_mean >= 0.18
                    and future_reset_mean <= 0.58
                ):
                    resume_found = True
                    break
        if (not resume_found) and is_open_tail:
            cluster_horizon_sec = min(2.35, max(1.55, 0.26 * interval_duration))
            gap_limit_sec = max(0.62, 18.0 * sample_dt)
            cluster_runs: list[tuple[int, int]] = []
            prev_end_local = dead_end_local
            for run_start_local, run_end_local in competitive_runs:
                if run_start_local <= dead_end_local:
                    continue
                run_start_t = float(interval_times[run_start_local])
                if run_start_t - dead_end_t > cluster_horizon_sec:
                    break
                if run_start_local > future_guard_idx:
                    break
                gap_before_run = float(interval_times[run_start_local] - interval_times[prev_end_local])
                if gap_before_run > gap_limit_sec:
                    break
                cluster_runs.append((run_start_local, run_end_local))
                prev_end_local = run_end_local

            if cluster_runs:
                cluster_start_local = cluster_runs[0][0]
                cluster_end_local = cluster_runs[-1][1]
                cluster_total_duration = float(
                    sum(run_duration_sec(run_start_local, run_end_local) for run_start_local, run_end_local in cluster_runs)
                )
                cluster_span = float(interval_times[cluster_end_local] - interval_times[cluster_start_local] + sample_dt)
                cluster_ball_peak = run_peak(interval_ball, cluster_start_local, cluster_end_local)
                cluster_table_peak = run_peak(interval_table, cluster_start_local, cluster_end_local)
                cluster_live_peak = run_peak(interval_live, cluster_start_local, cluster_end_local)
                cluster_effective_interaction_peak = run_peak(effective_interaction, cluster_start_local, cluster_end_local)
                cluster_effective_interaction_mean = run_mean(effective_interaction, cluster_start_local, cluster_end_local)
                cluster_reset_mean = run_mean(interval_reset, cluster_start_local, cluster_end_local)
                if (
                    len(cluster_runs) >= 3
                    and cluster_total_duration >= max(0.34, 7.0 * sample_dt)
                    and cluster_span >= max(0.80, 18.0 * sample_dt)
                    and cluster_ball_peak >= 0.24
                    and cluster_table_peak >= 0.70
                    and cluster_live_peak >= 0.58
                    and cluster_effective_interaction_peak >= 0.42
                    and cluster_effective_interaction_mean >= 0.22
                    and cluster_reset_mean <= 0.60
                ):
                    resume_found = True
        if resume_found:
            continue

        # Some rallies produce an early "dead" blip, then fragmented exchange
        # activity, then a later stronger dead run. In those cases we should let
        # the loop evaluate the later dead run instead of locking onto the first
        # brief dip.
        current_dead_short = bool(dead_duration_value <= max(0.45, 12.0 * sample_dt))
        future_dead_scan_horizon_sec = min(2.85, max(1.25, 0.34 * interval_duration))
        for future_dead_start_local, future_dead_end_local in dead_runs[dead_run_idx + 1 :]:
            future_dead_start_t = float(interval_times[future_dead_start_local])
            future_dead_gap_sec = future_dead_start_t - dead_end_t
            if future_dead_gap_sec > future_dead_scan_horizon_sec:
                break
            if future_dead_start_t > baseline_end + sample_dt:
                break

            future_dead_duration_value = run_duration_sec(future_dead_start_local, future_dead_end_local)
            future_dead_stronger = bool(
                future_dead_duration_value >= max(0.60, dead_duration_value + max(0.20, 4.0 * sample_dt))
                and terminal_reset_score[future_dead_start_local] >= (terminal_reset_score[dead_start_local] - 0.05)
            )

            bridge_exchange_runs = slice_runs_between(
                exchange_runs,
                after_idx=dead_end_local,
                before_idx=future_dead_start_local,
            )
            bridge_total_duration = runs_total_duration_sec(bridge_exchange_runs)
            bridge_span = runs_span_sec(bridge_exchange_runs)
            if bridge_exchange_runs:
                bridge_slice_start = bridge_exchange_runs[0][0]
                bridge_slice_end = bridge_exchange_runs[-1][1]
                bridge_ball_peak = run_peak(interval_ball, bridge_slice_start, bridge_slice_end)
                bridge_table_peak = run_peak(interval_table, bridge_slice_start, bridge_slice_end)
                bridge_live_peak = run_peak(interval_live, bridge_slice_start, bridge_slice_end)
                bridge_interaction_peak = run_peak(interval_interaction, bridge_slice_start, bridge_slice_end)
                bridge_effective_interaction_peak = run_peak(effective_interaction, bridge_slice_start, bridge_slice_end)
            else:
                bridge_ball_peak = 0.0
                bridge_table_peak = 0.0
                bridge_live_peak = 0.0
                bridge_interaction_peak = 0.0
                bridge_effective_interaction_peak = 0.0

            fragmented_continuation = bool(
                bridge_total_duration >= max(0.40, 10.0 * sample_dt)
                and bridge_span >= max(0.60, 12.0 * sample_dt)
                and (
                    bridge_live_peak >= 0.52
                    or bridge_table_peak >= 0.55
                    or bridge_effective_interaction_peak >= 0.24
                    or bridge_interaction_peak >= 0.32
                )
            )
            stable_follow_on_dead = bool(
                future_dead_gap_sec <= max(0.70, 16.0 * sample_dt)
                and bridge_total_duration <= max(0.10, 3.0 * sample_dt)
                and future_dead_duration_value >= max(1.40, dead_duration_value + max(0.70, 18.0 * sample_dt))
            )
            short_blip_before_later_dead = bool(
                current_dead_short
                and future_dead_gap_sec <= max(1.25, 30.0 * sample_dt)
                and future_dead_stronger
            )

            if fragmented_continuation or stable_follow_on_dead or short_blip_before_later_dead:
                resume_found = True
                break

        if resume_found:
            continue

        if not resume_found:
            tail_exchange_runs = slice_runs_between(
                exchange_runs,
                after_idx=dead_end_local,
                before_idx=len(exchange_mask),
            )
            next_dead_exists = any(next_dead_start_local > dead_end_local for next_dead_start_local, _ in dead_runs[dead_run_idx + 1 :])
            if tail_exchange_runs and not next_dead_exists:
                tail_first_gap_sec = float(interval_times[tail_exchange_runs[0][0]] - dead_end_t)
                tail_total_duration = runs_total_duration_sec(tail_exchange_runs)
                tail_span = runs_span_sec(tail_exchange_runs)
                tail_slice_start = tail_exchange_runs[0][0]
                tail_slice_end = tail_exchange_runs[-1][1]
                tail_ball_peak = run_peak(interval_ball, tail_slice_start, tail_slice_end)
                tail_table_peak = run_peak(interval_table, tail_slice_start, tail_slice_end)
                tail_live_peak = run_peak(interval_live, tail_slice_start, tail_slice_end)
                tail_interaction_peak = run_peak(interval_interaction, tail_slice_start, tail_slice_end)
                tail_effective_interaction_peak = run_peak(effective_interaction, tail_slice_start, tail_slice_end)
                tail_reset_mean = run_mean(interval_reset, tail_slice_start, tail_slice_end)
                if (
                    dead_duration_value >= max(2.00, 42.0 * sample_dt)
                    and tail_first_gap_sec <= max(1.05, 30.0 * sample_dt)
                    and tail_total_duration >= max(0.45, 12.0 * sample_dt)
                    and tail_span >= max(0.55, 14.0 * sample_dt)
                    and tail_ball_peak >= 0.60
                    and tail_table_peak >= 0.55
                    and tail_live_peak >= 0.45
                    and (
                        tail_effective_interaction_peak >= 0.20
                        or tail_interaction_peak >= 0.32
                    )
                    and tail_reset_mean <= 0.68
                ):
                    resume_found = True

        if resume_found:
            continue

        refined_dead_start_t = dead_start_t
        if (
            not strong_dead
            and dead_duration_value >= max(0.80, 24.0 * sample_dt)
            and interval_ball[dead_start_local] >= 0.18
            and terminal_reset_score[dead_start_local] < 0.82
        ):
            max_shift_samples = max(1, int(round(0.40 / max(sample_dt, 1e-6))))
            buffered_dead_start_local = dead_start_local
            for probe_local in range(dead_start_local, min(dead_end_local, dead_start_local + max_shift_samples) + 1):
                if (
                    terminal_reset_score[probe_local] >= 0.84
                    and interval_ball[probe_local] <= 0.18
                    and interval_interaction[probe_local] <= 0.05
                    and interval_table[probe_local] <= 0.08
                ):
                    buffered_dead_start_local = probe_local
                    break
            refined_dead_start_t = float(interval_times[buffered_dead_start_local])

        if embedded_exchange_tail_end_local is not None and embedded_exchange_tail_end_local < dead_end_local:
            delayed_dead_start_local = min(dead_end_local, embedded_exchange_tail_end_local + 1)
            refined_dead_start_t = float(max(refined_dead_start_t, float(interval_times[delayed_dead_start_local])))

        overlapping_competitive_continuation = False
        overlap_comp_horizon_sec = max(1.00, 24.0 * sample_dt)
        for run_start_local, run_end_local in competitive_runs:
            if run_end_local <= dead_end_local:
                continue
            run_start_t = float(interval_times[run_start_local])
            if run_start_t - dead_start_t > overlap_comp_horizon_sec:
                break
            overlap_start_local = max(run_start_local, dead_start_local)
            overlap_duration_value = run_duration_sec(overlap_start_local, run_end_local)
            if (
                overlap_duration_value >= max(0.28, 4.0 * sample_dt)
                and run_peak(interval_ball, overlap_start_local, run_end_local) >= 0.30
                and run_peak(interval_table, overlap_start_local, run_end_local) >= 0.70
                and run_peak(interval_live, overlap_start_local, run_end_local) >= 0.50
                and run_mean(effective_interaction, overlap_start_local, run_end_local) >= 0.18
                and run_mean(interval_reset, overlap_start_local, run_end_local) <= 0.60
            ):
                overlapping_competitive_continuation = True
                break
        if overlapping_competitive_continuation:
            continue

        refined_end = float(np.clip(refined_dead_start_t, safe_start + 0.01, safe_upper))
        dead_len = dead_end_local - dead_start_local + 1
        endpoint_confidence = float(
            np.clip(
                0.40
                + (0.07 * dead_len)
                + (0.18 * float(interval_reset[dead_start_local]))
                + (0.12 * float(1.0 - interval_ball[dead_start_local]))
                + (0.10 * prior_peak)
                + (0.10 * float(terminal_reset_score[dead_start_local])),
                0.28,
                0.95,
            )
        )
        return refined_end, "dead_reset_run_start", endpoint_confidence

    fallback_runs = competitive_runs if competitive_runs else exchange_runs
    if fallback_runs:
        selected_fallback_runs = fallback_runs
        if is_open_tail and competitive_runs:
            strong_open_tail_runs: list[tuple[int, int]] = []
            for run_start_local, run_end_local in competitive_runs:
                run_duration_value = run_duration_sec(run_start_local, run_end_local)
                mean_effective_interaction = run_mean(effective_interaction, run_start_local, run_end_local)
                mean_reset = run_mean(interval_reset, run_start_local, run_end_local)
                peak_live = run_peak(interval_live, run_start_local, run_end_local)
                peak_ball = run_peak(interval_ball, run_start_local, run_end_local)
                peak_table = run_peak(interval_table, run_start_local, run_end_local)
                if (
                    run_duration_value >= max(0.10, 3.0 * sample_dt)
                    and (
                        (
                            mean_effective_interaction >= 0.28
                            and mean_reset <= 0.62
                            and peak_live >= 0.45
                        )
                        or (
                            run_duration_value >= max(0.90, 22.0 * sample_dt)
                            and mean_effective_interaction >= 0.20
                            and mean_reset <= 0.50
                            and peak_live >= 0.75
                            and peak_ball >= 0.80
                            and peak_table >= 0.70
                        )
                    )
                ):
                    strong_open_tail_runs.append((run_start_local, run_end_local))
            if strong_open_tail_runs:
                selected_fallback_runs = strong_open_tail_runs

        _, last_live_local = selected_fallback_runs[-1]
        if competitive_runs and terminal_body_runs:
            last_run_start_local, last_run_end_local = selected_fallback_runs[-1]
            last_run_table_peak = run_peak(interval_table, last_run_start_local, last_run_end_local)
            last_run_interaction_peak = run_peak(interval_interaction, last_run_start_local, last_run_end_local)
            for body_start_local, body_end_local in terminal_body_runs:
                if body_end_local <= last_live_local:
                    continue
                if body_start_local > last_live_local + max(1, int(round(0.12 / max(sample_dt, 1e-6)))):
                    break
                body_duration_value = run_duration_sec(body_start_local, body_end_local)
                body_mean_ball = run_mean(interval_ball, body_start_local, body_end_local)
                body_mean_reset = run_mean(interval_reset, body_start_local, body_end_local)
                body_mean_table = run_mean(interval_table, body_start_local, body_end_local)
                body_mean_eff = run_mean(effective_interaction, body_start_local, body_end_local)
                if (
                    body_duration_value >= max(1.00, 28.0 * sample_dt)
                    and body_mean_ball >= 0.45
                    and body_mean_reset >= 0.62
                    and body_mean_table <= 0.06
                    and body_mean_eff <= 0.08
                    and last_run_table_peak >= 0.85
                    and last_run_interaction_peak >= 0.70
                ):
                    refined_end = float(np.clip(float(interval_times[body_end_local]), safe_start + 0.01, safe_upper))
                    endpoint_confidence = float(
                        np.clip(
                            0.30
                            + (0.18 * body_mean_ball)
                            + (0.10 * body_mean_reset)
                            + (0.12 * last_run_table_peak)
                            + (0.10 * last_run_interaction_peak),
                            0.24,
                            0.86,
                        )
                    )
                    return refined_end, "last_exchange_body_tail_end", endpoint_confidence
        refined_end = float(np.clip(float(interval_times[last_live_local]), safe_start + 0.01, safe_upper))
        endpoint_confidence = float(np.clip(0.24 + (0.62 * combined_live[last_live_local]), 0.20, 0.88))
        return refined_end, "last_exchange_support", endpoint_confidence

    return baseline_end, "detector_end_clamped", 0.20


def _refine_points_with_endpoint_signals(
    points: List[RallyTimelinePoint],
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
            support_series["interaction_pair"],
            support_series["one_sided_motion"],
            support_series["reset_pair"],
            support_series["shared_activity"],
            support_series["terminal_body_pair"],
            t_start=float(point.t_start),
            detector_end=float(point.t_end),
            search_upper_bound=float(search_upper_bound),
            is_open_tail=(point.boundary_mode == "video_end_open_tail"),
        )
        point.t_end = float(refined_end)
        point.endpoint_mode = endpoint_mode
        point.endpoint_confidence = float(endpoint_confidence)


def build_rally_timeline(
    video_path: str,
    table_weights_path: str,
    *,
    pose_weights_path: str = PRODUCTION_RALLY_DEFAULTS.pose_weights_path,
    best_of: int = 5,
    stride: int = PRODUCTION_RALLY_DEFAULTS.stride,
    mode: str = PRODUCTION_RALLY_DEFAULTS.mode,
    player_margin_px: int = PRODUCTION_RALLY_DEFAULTS.player_margin_px,
    player_fuse_gain: float = PRODUCTION_RALLY_DEFAULTS.player_fuse_gain,
    player_signal_source: str = PRODUCTION_RALLY_DEFAULTS.player_signal_source,
    ball_fuse_gain: float = PRODUCTION_RALLY_DEFAULTS.ball_fuse_gain,
    ball_signal_source: str = PRODUCTION_RALLY_DEFAULTS.ball_signal_source,
    table_roi=None,
    log_fn=None,
) -> RallyTimeline:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for multi-stream rally timeline generation.")

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
        table_roi=table_roi,
        log_fn=log_fn,
    )
    segments = detect_multistream_rallies(signals, mode=mode)

    v_path = Path(video_path).resolve()
    video_end_sec = 0.0
    if signals.timestamps:
        video_end_sec = float(signals.timestamps[-1])
    if segments:
        video_end_sec = float(max(video_end_sec, max(float(seg.t_end) for seg in segments)))
    video_end_sec = float(max(video_end_sec, _probe_video_duration_sec(video_path)))

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

    return RallyTimeline(
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
    parser = argparse.ArgumentParser(description="Generate rally timeline JSON using the current multistream pipeline.")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--weights", default=PRODUCTION_RALLY_DEFAULTS.table_weights_path, help="Path to YOLO table weights")
    parser.add_argument("--pose-weights", default=PRODUCTION_RALLY_DEFAULTS.pose_weights_path, help="Path to YOLO pose weights")
    parser.add_argument("--out", required=True, help="Output rally timeline JSON path")
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument("--stride", type=int, default=PRODUCTION_RALLY_DEFAULTS.stride)
    parser.add_argument("--mode", choices=["table", "player", "ball", "fused", "table_refined", "table_ball_refined"], default=PRODUCTION_RALLY_DEFAULTS.mode)
    parser.add_argument("--player-margin-px", type=int, default=PRODUCTION_RALLY_DEFAULTS.player_margin_px)
    parser.add_argument("--player-fuse-gain", type=float, default=PRODUCTION_RALLY_DEFAULTS.player_fuse_gain)
    parser.add_argument("--player-signal-source", choices=["role_tracker", "nearest_two", "none"], default=PRODUCTION_RALLY_DEFAULTS.player_signal_source)
    parser.add_argument("--ball-fuse-gain", type=float, default=PRODUCTION_RALLY_DEFAULTS.ball_fuse_gain)
    parser.add_argument("--ball-signal-source", choices=["none", "classical"], default=PRODUCTION_RALLY_DEFAULTS.ball_signal_source)
    parser.add_argument("--table-roi", type=str, default=None,
                        help="Pre-detected table ROI as JSON: '{\"x\":N,\"y\":N,\"w\":N,\"h\":N}'. "
                             "Skips YOLOv8x-table detection if provided.")
    args = parser.parse_args()

    # Parse optional pre-detected table ROI
    pre_table_roi = None
    if args.table_roi:
        import json as _json
        from backend.ai_table_roi import TableROI
        _roi_d = _json.loads(args.table_roi)
        pre_table_roi = TableROI(
            x=int(_roi_d["x"]), y=int(_roi_d["y"]),
            w=int(_roi_d["w"]), h=int(_roi_d["h"]),
            confidence=float(_roi_d.get("confidence", 1.0)),
            method=str(_roi_d.get("method", "cli_override")),
        )

    timeline = build_rally_timeline(
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
        table_roi=pre_table_roi,
    )
    out_path = Path(args.out)
    save_rally_timeline(out_path, timeline)
    print(f"[OK] Saved {args.mode} rally timeline: {out_path} | total_rallies={len(timeline.points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
