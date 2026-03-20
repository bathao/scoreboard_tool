from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from backend.ai_ball_tracking import extract_ball_motion_energies
from backend.offline_player_tracker import OfflinePlayerTracker, TrackletObservation
from backend.ai_rally_segmentation import RallySegment, detect_rally_segments_advanced_gpu
from backend.ai_table_roi import TableROI
from backend.ai_table_roi_dl import DLConfig, detect_table_roi_dl
from backend.video_gpu_io import nvdec_bgr24_stream, probe_video_ffprobe


@dataclass(frozen=True)
class MultiStreamSignals:
    roi: TableROI
    timestamps: List[float]
    table_energies: List[float]
    ball_energies: List[float]
    player_a_energies: List[float]
    player_b_energies: List[float]
    player_energies: List[float]
    fused_energies: List[float]
    effective_fps: float
    player_signal_source: str
    ball_signal_source: str


def _calc_wrist_velocity(
    curr_kpts: np.ndarray,
    prev_kpts: np.ndarray | None,
    player_h: float,
    *,
    gap_frames: int = 1,
) -> float:
    if prev_kpts is None or player_h <= 0:
        return 0.0

    norm_gap = max(1.0, float(gap_frames))
    valid_vels: List[float] = []
    for wrist_idx in (9, 10):
        cx, cy = curr_kpts[wrist_idx]
        px, py = prev_kpts[wrist_idx]
        if cx <= 0 or cy <= 0 or px <= 0 or py <= 0:
            continue
        dist = float(np.linalg.norm(curr_kpts[wrist_idx] - prev_kpts[wrist_idx]))
        norm_v = dist / (float(player_h) * norm_gap)
        if norm_v < 0.75:
            valid_vels.append(norm_v)
    return max(valid_vels) if valid_vels else 0.0


def _calc_box_center_velocity(
    curr_box: Tuple[int, int, int, int],
    prev_box: Tuple[int, int, int, int],
    player_h: float,
    *,
    gap_frames: int = 1,
) -> float:
    if player_h <= 0:
        return 0.0

    curr_center = np.array([(curr_box[0] + curr_box[2]) / 2.0, (curr_box[1] + curr_box[3]) / 2.0], dtype=np.float32)
    prev_center = np.array([(prev_box[0] + prev_box[2]) / 2.0, (prev_box[1] + prev_box[3]) / 2.0], dtype=np.float32)
    norm_gap = max(1.0, float(gap_frames))
    return float(np.linalg.norm(curr_center - prev_center) / (float(player_h) * norm_gap))


def _is_player_near_table(box_xyxy: np.ndarray, roi: TableROI, margin_px: int) -> bool:
    x1, _y1, x2, _y2 = [float(v) for v in box_xyxy]
    tx, _ty, tw, _th = roi.as_tuple()
    return (x1 < tx + tw + margin_px) and (x2 > tx - margin_px)


def _smooth_and_normalize(values: List[float]) -> np.ndarray:
    if not values:
        return np.array([], dtype=np.float32)

    arr = np.asarray(values, dtype=np.float32)
    if arr.size < 3:
        p10, p95 = float(arr.min()), float(arr.max())
        return np.clip((arr - p10) / (p95 - p10 + 1e-6), 0.0, 1.0)

    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    smooth = np.convolve(arr, kernel, mode="same")
    p10, p95 = np.percentile(smooth, 10), np.percentile(smooth, 95)
    return np.clip((smooth - p10) / (p95 - p10 + 1e-6), 0.0, 1.0)


def _calc_role_motion_energy(
    curr_obs: TrackletObservation,
    prev_obs: Optional[TrackletObservation],
    *,
    gap_frames: int = 1,
) -> float:
    if prev_obs is None:
        return 0.0

    player_h = max(1.0, float(curr_obs.box[3] - curr_obs.box[1]))
    wrist_v = _calc_wrist_velocity(
        curr_obs.keypoints,
        prev_obs.keypoints,
        player_h,
        gap_frames=max(1, gap_frames),
    )
    center_v = _calc_box_center_velocity(
        curr_obs.box,
        prev_obs.box,
        player_h,
        gap_frames=max(1, gap_frames),
    )
    return max(wrist_v, min(center_v, 1.0))


def _build_role_energy_series(
    frame_indices: List[int],
    role_frames: Dict[int, Dict[str, TrackletObservation]],
    role_state_frames: Dict[int, Dict[str, str]],
    *,
    role: str,
    occluded_hold_samples: int = 3,
    occluded_decay: float = 0.55,
) -> List[float]:
    energies: List[float] = []
    prev_obs: Optional[TrackletObservation] = None
    prev_visible_frame_idx: Optional[int] = None
    prev_energy = 0.0

    for frame_idx in frame_indices:
        obs = role_frames.get(frame_idx, {}).get(role)
        state = role_state_frames.get(frame_idx, {}).get(role)

        if obs is not None:
            gap_frames = 1 if prev_visible_frame_idx is None else max(1, frame_idx - prev_visible_frame_idx)
            energy = _calc_role_motion_energy(obs, prev_obs, gap_frames=gap_frames)
            prev_obs = obs
            prev_visible_frame_idx = frame_idx
            prev_energy = energy
        elif state == "occluded" and prev_visible_frame_idx is not None:
            sample_gap = max(1, frame_idx - prev_visible_frame_idx)
            if sample_gap <= occluded_hold_samples:
                energy = prev_energy * float(occluded_decay ** max(0, sample_gap - 1))
            else:
                energy = 0.0
        else:
            energy = 0.0
            if state != "occluded":
                prev_energy = 0.0

        energies.append(float(energy))

    return energies


def _collect_role_tracker_energies(
    cap: cv2.VideoCapture,
    person_model: YOLO,
    *,
    roi: TableROI,
    fps: float,
    stride: int,
    device: str,
    player_margin_px: int,
    frame_w: int,
    frame_h: int,
) -> Tuple[List[int], List[float], List[float], List[float]]:
    tx, ty, tw, th = roi.as_tuple()
    tracker = OfflinePlayerTracker(roi, frame_w=frame_w, frame_h=frame_h)

    timestamps: List[float] = []
    frame_indices: List[int] = []
    table_energies: List[float] = []
    prev_table_gray: Optional[np.ndarray] = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        timestamps.append(float(frame_idx / fps))
        frame_indices.append(int(frame_idx))

        table_crop = frame[ty : ty + th, tx : tx + tw]
        table_gray = cv2.cvtColor(table_crop, cv2.COLOR_BGR2GRAY)
        if prev_table_gray is None:
            table_energies.append(0.0)
        else:
            table_energies.append(float(np.mean(cv2.absdiff(table_gray, prev_table_gray))))
        prev_table_gray = table_gray

        result = person_model.predict(
            frame,
            classes=[0],
            device=device,
            verbose=False,
        )[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
            keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
            if boxes.size > 0:
                near_mask = [_is_player_near_table(box, roi, player_margin_px) for box in boxes]
                boxes = boxes[np.asarray(near_mask, dtype=bool)]
                if confs is not None:
                    confs = confs[np.asarray(near_mask, dtype=bool)]
                if keypoints is not None:
                    keypoints = keypoints[np.asarray(near_mask, dtype=bool)]
            if boxes.size > 0:
                detections = tracker.build_detections(
                    frame,
                    frame_idx=frame_idx,
                    boxes_xyxy=boxes,
                    keypoints_xy=keypoints,
                    confidences=confs,
                )
                tracker.add_frame_detections(detections)

        frame_idx += 1

    tracking_result = tracker.finish()
    role_hold_samples = max(2, int(round(6 / max(1, stride))))
    player_a_energies = _build_role_energy_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="A",
        occluded_hold_samples=role_hold_samples,
    )
    player_b_energies = _build_role_energy_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="B",
        occluded_hold_samples=role_hold_samples,
    )
    return timestamps, table_energies, player_a_energies, player_b_energies


def _collect_nearest_two_energies(
    cap: cv2.VideoCapture,
    person_model: YOLO,
    *,
    roi: TableROI,
    fps: float,
    stride: int,
    device: str,
    player_margin_px: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    tx, ty, tw, th = roi.as_tuple()
    timestamps: List[float] = []
    table_energies: List[float] = []
    player_energies: List[float] = []
    prev_table_gray: Optional[np.ndarray] = None
    prev_player_kpts: Dict[int, np.ndarray] = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        timestamp = float(frame_idx / fps)
        timestamps.append(timestamp)

        table_crop = frame[ty : ty + th, tx : tx + tw]
        table_gray = cv2.cvtColor(table_crop, cv2.COLOR_BGR2GRAY)
        if prev_table_gray is None:
            table_energies.append(0.0)
        else:
            table_energies.append(float(np.mean(cv2.absdiff(table_gray, prev_table_gray))))
        prev_table_gray = table_gray

        results = person_model.track(
            frame,
            persist=True,
            classes=[0],
            device=device,
            verbose=False,
        )

        frame_player_vels: List[float] = []
        if results and results[0].boxes.id is not None and results[0].keypoints is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            kpts = results[0].keypoints.xy.cpu().numpy()
            candidates: List[Tuple[float, int]] = []
            table_center = np.array([tx + tw / 2.0, ty + th / 2.0], dtype=np.float32)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
                dist = float(np.linalg.norm(center - table_center))
                candidates.append((dist, i))
            candidates.sort(key=lambda x: x[0])

            for _, idx in candidates[:2]:
                player_id = int(ids[idx])
                box = boxes[idx]
                if not _is_player_near_table(box, roi, player_margin_px):
                    continue
                player_h = max(1.0, float(box[3] - box[1]))
                vel = _calc_wrist_velocity(kpts[idx], prev_player_kpts.get(player_id), player_h)
                frame_player_vels.append(vel)
                prev_player_kpts[player_id] = kpts[idx]

        player_energies.append(max(frame_player_vels) if frame_player_vels else 0.0)
        frame_idx += 1

    return timestamps, table_energies, player_energies, player_energies


def _extract_production_table_energies(
    video_path: str,
    *,
    roi: TableROI,
    width: int,
    height: int,
    fps: float,
    stride: int,
    device: str,
) -> Tuple[List[float], List[float]]:
    tx, ty, tw, th = roi.as_tuple()
    energies: List[float] = []
    timestamps: List[float] = []
    prev_frame_gpu = None

    frame_iter = nvdec_bgr24_stream(
        str(video_path),
        int(width),
        int(height),
        crop_roi=(tx, ty, tw, th),
    )

    for idx, frame_np in enumerate(frame_iter):
        if idx % stride != 0:
            continue

        curr_gpu = torch.from_numpy(frame_np).to(device).float()
        if prev_frame_gpu is not None:
            diff = torch.abs(curr_gpu - prev_frame_gpu)
            diff_max = F.max_pool2d(
                diff.permute(2, 0, 1).unsqueeze(0),
                kernel_size=3,
                stride=1,
                padding=1,
            )
            energies.append(float(diff_max.mean().item()))
            timestamps.append(float(idx / fps))
        prev_frame_gpu = curr_gpu

    return timestamps, energies


def _refine_table_segments_with_role_support(
    segments: List[RallySegment],
    *,
    timestamps: List[float],
    table_energies: List[float],
    player_a_energies: List[float],
    player_b_energies: List[float],
    min_segment_sec: float = 8.0,
    quiet_table_thresh: float = 0.28,
    quiet_role_thresh: float = 0.16,
    quiet_run_sec: float = 0.75,
    boundary_guard_sec: float = 0.9,
    min_piece_sec: float = 1.0,
) -> List[RallySegment]:
    if not segments or not timestamps:
        return list(segments)

    ts = np.asarray(timestamps, dtype=np.float32)
    table_norm = _smooth_and_normalize(table_energies)
    player_a_norm = _smooth_and_normalize(player_a_energies)
    player_b_norm = _smooth_and_normalize(player_b_energies)
    role_norm = np.maximum(player_a_norm, player_b_norm)

    refined: List[RallySegment] = []
    for seg in segments:
        seg_dur = float(seg.t_end - seg.t_start)
        if seg_dur < min_segment_sec:
            refined.append(seg)
            continue

        start_idx = int(np.searchsorted(ts, float(seg.t_start), side="left"))
        end_idx = int(np.searchsorted(ts, float(seg.t_end), side="right")) - 1
        start_idx = max(0, min(start_idx, len(ts) - 1))
        end_idx = max(start_idx, min(end_idx, len(ts) - 1))
        if end_idx <= start_idx:
            refined.append(seg)
            continue

        inner_start_t = float(seg.t_start + boundary_guard_sec)
        inner_end_t = float(seg.t_end - boundary_guard_sec)
        inner_start_idx = int(np.searchsorted(ts, inner_start_t, side="left"))
        inner_end_idx = int(np.searchsorted(ts, inner_end_t, side="right")) - 1
        inner_start_idx = max(start_idx + 1, inner_start_idx)
        inner_end_idx = min(end_idx - 1, inner_end_idx)
        if inner_end_idx <= inner_start_idx:
            refined.append(seg)
            continue

        quiet_cuts: List[int] = []
        run_start: Optional[int] = None
        for idx in range(inner_start_idx, inner_end_idx + 1):
            is_quiet = bool(table_norm[idx] < quiet_table_thresh and role_norm[idx] < quiet_role_thresh)
            if is_quiet:
                if run_start is None:
                    run_start = idx
                continue

            if run_start is not None:
                run_end = idx - 1
                if float(ts[run_end] - ts[run_start]) >= quiet_run_sec:
                    quiet_cuts.append(int((run_start + run_end) // 2))
                run_start = None

        if run_start is not None:
            run_end = inner_end_idx
            if float(ts[run_end] - ts[run_start]) >= quiet_run_sec:
                quiet_cuts.append(int((run_start + run_end) // 2))

        if not quiet_cuts:
            refined.append(seg)
            continue

        cut_indices = sorted(set(quiet_cuts))
        current_start_idx = start_idx
        split_segments: List[RallySegment] = []
        for cut_idx in cut_indices:
            left_dur = float(ts[cut_idx] - ts[current_start_idx])
            right_dur = float(ts[end_idx] - ts[cut_idx])
            if left_dur < min_piece_sec or right_dur < min_piece_sec:
                continue

            piece_conf = float(np.clip(np.median(table_norm[current_start_idx:cut_idx + 1]), 0.5, 1.0))
            split_segments.append(
                RallySegment(
                    t_start=float(ts[current_start_idx]),
                    t_end=float(ts[cut_idx]),
                    confidence=piece_conf,
                    flags=sorted(set(list(seg.flags) + ["role_quiet_gap_split"])),
                )
            )
            current_start_idx = cut_idx

        final_dur = float(ts[end_idx] - ts[current_start_idx])
        if not split_segments or final_dur < min_piece_sec:
            refined.append(seg)
            continue

        piece_conf = float(np.clip(np.median(table_norm[current_start_idx:end_idx + 1]), 0.5, 1.0))
        split_segments.append(
            RallySegment(
                t_start=float(ts[current_start_idx]),
                t_end=float(ts[end_idx]),
                confidence=piece_conf,
                flags=sorted(set(list(seg.flags) + ["role_quiet_gap_split"])),
            )
        )
        refined.extend(split_segments)

    return refined


def _merge_segments_with_ball_support(
    segments: List[RallySegment],
    *,
    timestamps: List[float],
    ball_energies: List[float],
    merge_max_gap_sec: float = 1.2,
    max_merged_duration_sec: float = 8.5,
    boundary_window_sec: float = 0.65,
    active_peak_thresh: float = 0.34,
    active_mean_thresh: float = 0.18,
) -> List[RallySegment]:
    if len(segments) <= 1 or not timestamps or not ball_energies:
        return list(segments)

    ts = np.asarray(timestamps, dtype=np.float32)
    ball_norm = _smooth_and_normalize(ball_energies)
    merged: List[RallySegment] = []
    current = segments[0]

    for nxt in segments[1:]:
        gap = float(nxt.t_start - current.t_end)
        should_merge = False
        candidate_duration = float(nxt.t_end - current.t_start)
        if gap <= merge_max_gap_sec and candidate_duration <= max_merged_duration_sec and (
            gap <= 0.08 or "split_long" in current.flags or "split_long" in nxt.flags
        ):
            win_start = max(float(current.t_start), float(current.t_end - boundary_window_sec))
            win_end = min(float(nxt.t_end), float(nxt.t_start + boundary_window_sec))
            start_idx = int(np.searchsorted(ts, win_start, side="left"))
            end_idx = int(np.searchsorted(ts, win_end, side="right")) - 1
            start_idx = max(0, min(start_idx, len(ts) - 1))
            end_idx = max(start_idx, min(end_idx, len(ts) - 1))
            if end_idx >= start_idx:
                window = ball_norm[start_idx : end_idx + 1]
                peak = float(window.max())
                mean = float(window.mean())
                if peak >= active_peak_thresh and mean >= active_mean_thresh:
                    should_merge = True

        if should_merge:
            merged_flags = sorted(set(list(current.flags) + list(nxt.flags) + ["ball_gap_merge"]))
            merged_conf = float(np.clip(np.median([current.confidence, nxt.confidence]), 0.5, 1.0))
            current = RallySegment(
                t_start=float(current.t_start),
                t_end=float(nxt.t_end),
                confidence=merged_conf,
                flags=merged_flags,
            )
            continue

        merged.append(current)
        current = nxt

    merged.append(current)
    return merged


def _merge_ball_split_pair_artifacts(
    segments: List[RallySegment],
    *,
    contiguous_eps_sec: float = 0.05,
    short_piece_sec: float = 3.8,
    max_pair_sec: float = 10.0,
) -> List[RallySegment]:
    if len(segments) <= 1:
        return list(segments)

    merged: List[RallySegment] = []
    i = 0
    while i < len(segments):
        if i + 1 < len(segments):
            current = segments[i]
            nxt = segments[i + 1]
            gap = float(nxt.t_start - current.t_end)
            current_dur = float(current.t_end - current.t_start)
            next_dur = float(nxt.t_end - nxt.t_start)
            if (
                gap <= contiguous_eps_sec
                and "split_long" in current.flags
                and "split_long" in nxt.flags
                and min(current_dur, next_dur) < short_piece_sec
                and (current_dur + next_dur) <= max_pair_sec
            ):
                merged.append(
                    RallySegment(
                        t_start=float(current.t_start),
                        t_end=float(nxt.t_end),
                        confidence=float(max(current.confidence, nxt.confidence)),
                        flags=sorted(set(list(current.flags) + list(nxt.flags) + ["ball_pair_merge"])),
                    )
                )
                i += 2
                continue

        merged.append(segments[i])
        i += 1

    return merged


def extract_multistream_signals(
    video_path: str,
    table_weights_path: str,
    *,
    pose_weights_path: str = "weights/yolov8x-pose.pt",
    stride: int = 2,
    player_margin_px: int = 220,
    player_fuse_gain: float = 1.0,
    player_signal_source: str = "role_tracker",
    ball_fuse_gain: float = 1.15,
    ball_signal_source: str = "none",
    ball_tracking_profile: str = "support",
    device: str = "cuda",
) -> MultiStreamSignals:
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA GPU is required for multi-stream extraction.")

    v_path = Path(video_path).resolve()
    table_w_path = Path(table_weights_path).resolve()
    pose_w_path = Path(pose_weights_path).resolve()
    if not v_path.exists():
        raise FileNotFoundError(f"Video not found: {v_path}")
    if not table_w_path.exists():
        raise FileNotFoundError(f"Table weights not found: {table_w_path}")
    if player_signal_source not in {"role_tracker", "nearest_two", "none"}:
        raise ValueError(f"Invalid player_signal_source: {player_signal_source}")
    if ball_signal_source not in {"none", "classical"}:
        raise ValueError(f"Invalid ball_signal_source: {ball_signal_source}")
    if player_signal_source != "none" and not pose_w_path.exists():
        raise FileNotFoundError(f"Pose weights not found: {pose_w_path}")

    info = probe_video_ffprobe(str(v_path))
    roi = detect_table_roi_dl(
        str(v_path),
        cfg=DLConfig(weights_path=str(table_w_path), device=device),
    )
    if player_signal_source == "none":
        timestamps, table_energies = _extract_production_table_energies(
            str(v_path),
            roi=roi,
            width=int(info.width),
            height=int(info.height),
            fps=float(info.fps),
            stride=max(1, int(stride)),
            device=device,
        )
        player_a_energies = [0.0 for _ in timestamps]
        player_b_energies = [0.0 for _ in timestamps]
    else:
        person_model = YOLO(str(pose_w_path))
        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {v_path}")

        if player_signal_source == "role_tracker":
            timestamps, table_energies, player_a_energies, player_b_energies = _collect_role_tracker_energies(
                cap,
                person_model,
                roi=roi,
                fps=info.fps,
                stride=max(1, int(stride)),
                device=device,
                player_margin_px=int(player_margin_px),
                frame_w=int(info.width),
                frame_h=int(info.height),
            )
        else:
            timestamps, table_energies, player_a_energies, player_b_energies = _collect_nearest_two_energies(
                cap,
                person_model,
                roi=roi,
                fps=info.fps,
                stride=max(1, int(stride)),
                device=device,
                player_margin_px=int(player_margin_px),
            )

    if player_signal_source != "none":
        cap.release()
        prod_timestamps, prod_table_energies = _extract_production_table_energies(
            str(v_path),
            roi=roi,
            width=int(info.width),
            height=int(info.height),
            fps=float(info.fps),
            stride=max(1, int(stride)),
            device=device,
        )

        if prod_timestamps and prod_table_energies:
            aligned_len = min(len(prod_timestamps), max(0, len(player_a_energies) - 1), max(0, len(player_b_energies) - 1))
            timestamps = prod_timestamps[:aligned_len]
            table_energies = prod_table_energies[:aligned_len]
            player_a_energies = player_a_energies[1 : 1 + aligned_len]
            player_b_energies = player_b_energies[1 : 1 + aligned_len]
        elif timestamps:
            timestamps = timestamps[1:]
            table_energies = table_energies[1:]
            player_a_energies = player_a_energies[1:]
            player_b_energies = player_b_energies[1:]

    if ball_signal_source == "classical":
        ball_signals = extract_ball_motion_energies(
            str(v_path),
            roi=roi,
            frame_w=int(info.width),
            frame_h=int(info.height),
            fps=float(info.fps),
            stride=max(1, int(stride)),
            profile=ball_tracking_profile,
        )
        aligned_len = min(len(timestamps), len(ball_signals.timestamps), len(ball_signals.energies))
        timestamps = timestamps[:aligned_len]
        table_energies = table_energies[:aligned_len]
        player_a_energies = player_a_energies[:aligned_len]
        player_b_energies = player_b_energies[:aligned_len]
        ball_energies = ball_signals.energies[:aligned_len]
    else:
        ball_energies = [0.0 for _ in timestamps]

    player_energies = [
        max(float(a), float(b))
        for a, b in zip(player_a_energies, player_b_energies)
    ]

    table_norm = _smooth_and_normalize(table_energies)
    ball_norm = _smooth_and_normalize(ball_energies)
    player_a_norm = _smooth_and_normalize(player_a_energies)
    player_b_norm = _smooth_and_normalize(player_b_energies)
    player_norm = np.maximum(player_a_norm, player_b_norm)
    fused_norm = np.maximum(
        np.maximum(table_norm, player_norm * float(player_fuse_gain)),
        ball_norm * float(ball_fuse_gain),
    )

    return MultiStreamSignals(
        roi=roi,
        timestamps=timestamps,
        table_energies=table_energies,
        ball_energies=ball_energies,
        player_a_energies=player_a_energies,
        player_b_energies=player_b_energies,
        player_energies=player_energies,
        fused_energies=fused_norm.tolist(),
        effective_fps=float(info.fps / max(1, stride)),
        player_signal_source=player_signal_source,
        ball_signal_source=ball_signal_source,
    )


def detect_multistream_rallies(
    signals: MultiStreamSignals,
    *,
    mode: str = "fused",
) -> List[RallySegment]:
    if mode not in {"table", "ball", "fused", "table_refined", "table_ball_refined"}:
        raise ValueError(f"Invalid mode: {mode}")
    if mode == "ball" and signals.ball_signal_source == "none":
        raise ValueError("Ball-only mode requires a real ball signal source.")

    if mode in {"table", "table_refined", "table_ball_refined"}:
        energies = signals.table_energies
    elif mode == "ball":
        energies = signals.ball_energies
    else:
        energies = signals.fused_energies
    detect_kwargs = {}
    if mode == "ball":
        detect_kwargs = {
            "high_thresh": 0.28,
            "low_thresh": 0.08,
            "max_gap_sec": 1.15,
            "long_segment_sec": 10.0,
            "split_gap_sec": 0.40,
            "min_split_dur_sec": 1.5,
            "artifact_min_dur_sec": 1.2,
        }
    segments = detect_rally_segments_advanced_gpu(
        list(energies),
        list(signals.timestamps),
        effective_fps=signals.effective_fps,
        **detect_kwargs,
    )
    if mode in {"table", "ball"}:
        if mode == "ball":
            return _merge_ball_split_pair_artifacts(segments)
        return segments
    if mode == "table_refined":
        return _refine_table_segments_with_role_support(
            segments,
            timestamps=signals.timestamps,
            table_energies=signals.table_energies,
            player_a_energies=signals.player_a_energies,
            player_b_energies=signals.player_b_energies,
        )
    if mode == "table_ball_refined":
        return _merge_segments_with_ball_support(
            segments,
            timestamps=signals.timestamps,
            ball_energies=signals.ball_energies,
        )
    return segments
