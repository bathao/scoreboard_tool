from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_multistream_rally import (
    MultiStreamSignals,
    PlayerStateMachineDiagnostics,
    _build_role_energy_series,
    _build_role_feature_series,
    _calc_role_crouch_raw,
    _calc_role_footwork_raw,
    _calc_role_net_approach_raw,
    _calc_role_reach_raw,
    _calc_role_serve_raw,
    _calc_role_upper_body_raw,
    _compute_player_state_machine_diagnostics,
    _is_player_near_table,
)
from backend.ai_table_roi_dl import DLConfig, detect_table_roi_dl
from backend.offline_player_tracker import OfflinePlayerTracker, TrackletObservation
from backend.video_gpu_io import probe_video_ffprobe


def _run_pose_detection(
    person_model: YOLO,
    frame: np.ndarray,
    *,
    device: str,
    imgsz: int,
):
    return person_model.predict(frame, classes=[0], device=device, verbose=False, imgsz=imgsz)[0]


def _build_role_tracker_debug_payload(
    video_path: Path,
    *,
    table_weights: str,
    pose_weights: str,
    stride: int,
    player_margin_px: int,
    device: str,
    start_seconds: float,
    max_seconds: float,
    imgsz: int,
) -> Tuple[
    Dict[int, Dict[str, TrackletObservation]],
    Dict[int, Dict[str, str]],
    List[int],
    PlayerStateMachineDiagnostics,
    Tuple[int, int, int, int],
    float,
    int,
    int,
]:
    info = probe_video_ffprobe(str(video_path))
    table_roi = detect_table_roi_dl(
        str(video_path),
        cfg=DLConfig(weights_path=table_weights, device=device),
    )
    tx, ty, tw, th = table_roi.as_tuple()

    fps = float(info.fps)
    start_frame = int(max(0.0, start_seconds) * fps)
    if max_seconds > 0:
        end_frame = int((start_seconds + max_seconds) * fps)
    else:
        cap_probe = cv2.VideoCapture(str(video_path))
        total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap_probe.release()
        end_frame = total_frames - 1 if total_frames > 0 else start_frame

    person_model = YOLO(str(Path(pose_weights).resolve()))
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    tracker = OfflinePlayerTracker(table_roi, frame_w=int(info.width), frame_h=int(info.height))
    frame_indices: List[int] = []
    timestamps: List[float] = []
    frame_idx = start_frame

    print("Pass 1/2: offline role tracking and sampled feature collection...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx > end_frame:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        timestamps.append(float(frame_idx / fps))
        frame_indices.append(int(frame_idx))

        result = _run_pose_detection(person_model, frame, device=device, imgsz=imgsz)
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
            keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
            if boxes.size > 0:
                near_mask = [_is_player_near_table(box, table_roi, player_margin_px) for box in boxes]
                keep = np.asarray(near_mask, dtype=bool)
                boxes = boxes[keep]
                if confs is not None:
                    confs = confs[keep]
                if keypoints is not None:
                    keypoints = keypoints[keep]
            if boxes.size > 0:
                detections = tracker.build_detections(
                    frame,
                    frame_idx=frame_idx,
                    boxes_xyxy=boxes,
                    keypoints_xy=keypoints,
                    confidences=confs,
                )
                tracker.add_frame_detections(detections)

        if len(frame_indices) % 200 == 0 and frame_indices:
            print(f"  analyzed sampled frame {frame_indices[-1]}")
        frame_idx += 1

    cap.release()
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
    player_a_crouch = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="A",
        feature_fn=lambda obs, _prev, gap_frames=1: _calc_role_crouch_raw(obs),
        occluded_hold_samples=role_hold_samples,
        occluded_decay=0.72,
    )
    player_b_crouch = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="B",
        feature_fn=lambda obs, _prev, gap_frames=1: _calc_role_crouch_raw(obs),
        occluded_hold_samples=role_hold_samples,
        occluded_decay=0.72,
    )
    player_a_serve = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="A",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_serve_raw(obs, prev, gap_frames=gap_frames),
        occluded_hold_samples=max(1, role_hold_samples - 1),
        occluded_decay=0.55,
    )
    player_b_serve = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="B",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_serve_raw(obs, prev, gap_frames=gap_frames),
        occluded_hold_samples=max(1, role_hold_samples - 1),
        occluded_decay=0.55,
    )
    player_a_upper = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="A",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_upper_body_raw(obs, prev, gap_frames=gap_frames),
        occluded_hold_samples=max(1, role_hold_samples - 1),
        occluded_decay=0.58,
    )
    player_b_upper = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="B",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_upper_body_raw(obs, prev, gap_frames=gap_frames),
        occluded_hold_samples=max(1, role_hold_samples - 1),
        occluded_decay=0.58,
    )
    player_a_footwork = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="A",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_footwork_raw(obs, prev, gap_frames=gap_frames),
        occluded_hold_samples=role_hold_samples,
        occluded_decay=0.62,
    )
    player_b_footwork = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="B",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_footwork_raw(obs, prev, gap_frames=gap_frames),
        occluded_hold_samples=role_hold_samples,
        occluded_decay=0.62,
    )
    player_a_reach = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="A",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_reach_raw(obs, prev, role="A", gap_frames=gap_frames),
        occluded_hold_samples=max(1, role_hold_samples - 1),
        occluded_decay=0.55,
    )
    player_b_reach = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="B",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_reach_raw(obs, prev, role="B", gap_frames=gap_frames),
        occluded_hold_samples=max(1, role_hold_samples - 1),
        occluded_decay=0.55,
    )
    player_a_approach = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="A",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_net_approach_raw(obs, prev, roi=table_roi, gap_frames=gap_frames),
        occluded_hold_samples=role_hold_samples,
        occluded_decay=0.68,
    )
    player_b_approach = _build_role_feature_series(
        frame_indices,
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        role="B",
        feature_fn=lambda obs, prev, gap_frames=1: _calc_role_net_approach_raw(obs, prev, roi=table_roi, gap_frames=gap_frames),
        occluded_hold_samples=role_hold_samples,
        occluded_decay=0.68,
    )

    if len(timestamps) <= 1:
        raise RuntimeError("Not enough sampled frames to build player-state diagnostics.")

    aligned_timestamps = timestamps[1:]
    aligned_frame_indices = frame_indices[1:]
    player_a_energies = player_a_energies[1:]
    player_b_energies = player_b_energies[1:]
    player_a_crouch = player_a_crouch[1:]
    player_b_crouch = player_b_crouch[1:]
    player_a_serve = player_a_serve[1:]
    player_b_serve = player_b_serve[1:]
    player_a_upper = player_a_upper[1:]
    player_b_upper = player_b_upper[1:]
    player_a_footwork = player_a_footwork[1:]
    player_b_footwork = player_b_footwork[1:]
    player_a_reach = player_a_reach[1:]
    player_b_reach = player_b_reach[1:]
    player_a_approach = player_a_approach[1:]
    player_b_approach = player_b_approach[1:]

    player_energies = [max(float(a), float(b)) for a, b in zip(player_a_energies, player_b_energies)]
    signals = MultiStreamSignals(
        roi=table_roi,
        timestamps=aligned_timestamps,
        table_energies=[0.0 for _ in aligned_timestamps],
        ball_energies=[0.0 for _ in aligned_timestamps],
        player_a_energies=player_a_energies,
        player_b_energies=player_b_energies,
        player_energies=player_energies,
        fused_energies=player_energies,
        effective_fps=float(fps / max(1, stride)),
        player_signal_source="role_tracker",
        ball_signal_source="none",
        player_a_crouch_scores=player_a_crouch,
        player_b_crouch_scores=player_b_crouch,
        player_a_serve_scores=player_a_serve,
        player_b_serve_scores=player_b_serve,
        player_a_upper_body_scores=player_a_upper,
        player_b_upper_body_scores=player_b_upper,
        player_a_footwork_scores=player_a_footwork,
        player_b_footwork_scores=player_b_footwork,
        player_a_reach_scores=player_a_reach,
        player_b_reach_scores=player_b_reach,
        player_a_net_approach_scores=player_a_approach,
        player_b_net_approach_scores=player_b_approach,
    )
    diagnostics = _compute_player_state_machine_diagnostics(signals)
    return (
        tracking_result.role_frames,
        tracking_result.role_state_frames,
        aligned_frame_indices,
        diagnostics,
        (tx, ty, tw, th),
        fps,
        int(info.width),
        int(info.height),
    )


def _segment_at_time(segments, timestamp: float):
    for idx, seg in enumerate(segments, start=1):
        if float(seg.t_start) <= timestamp <= float(seg.t_end):
            return idx, seg
    return None, None


def _draw_timeline(
    frame: np.ndarray,
    *,
    diagnostics: PlayerStateMachineDiagnostics,
    current_t: float,
    clip_start: float,
    clip_end: float,
) -> None:
    h, w = frame.shape[:2]
    bar_x1 = 60
    bar_x2 = w - 60
    bar_y1 = h - 70
    bar_y2 = h - 38
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (35, 35, 35), -1)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (180, 180, 180), 2)
    total = max(1e-6, clip_end - clip_start)

    for seg in diagnostics.segments:
        sx = int(bar_x1 + ((float(seg.t_start) - clip_start) / total) * (bar_x2 - bar_x1))
        ex = int(bar_x1 + ((float(seg.t_end) - clip_start) / total) * (bar_x2 - bar_x1))
        ex = max(ex, sx + 2)
        color = (0, 180, 255) if "rally_label_let" in seg.flags else (0, 180, 0)
        cv2.rectangle(frame, (sx, bar_y1 + 4), (ex, bar_y2 - 4), color, -1)

    cx = int(bar_x1 + ((current_t - clip_start) / total) * (bar_x2 - bar_x1))
    cx = max(bar_x1, min(bar_x2, cx))
    cv2.line(frame, (cx, bar_y1 - 8), (cx, bar_y2 + 8), (255, 255, 255), 3)
    cv2.putText(frame, f"{clip_start:.1f}s", (bar_x1, bar_y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
    cv2.putText(frame, f"{clip_end:.1f}s", (bar_x2 - 70, bar_y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)


def _write_debug_csv(out_csv: Path, frame_indices: List[int], diagnostics: PlayerStateMachineDiagnostics) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_idx",
        "frame_idx",
        "timestamp",
        "phase",
        "server_role",
        "ready_recent",
        "live_now",
        "dead_now",
        "catch_proxy",
        "quiet_after_catch",
        "ready_pair",
        "live_pair",
        "casual_pair",
        "stand_pair",
        "motion_a",
        "motion_b",
        "crouch_a",
        "crouch_b",
        "serve_a",
        "serve_b",
        "upper_a",
        "upper_b",
        "foot_a",
        "foot_b",
        "reach_a",
        "reach_b",
        "approach_a",
        "approach_b",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, frame_idx in enumerate(frame_indices):
            writer.writerow(
                {
                    "sample_idx": i,
                    "frame_idx": frame_idx,
                    "timestamp": diagnostics.timestamps[i],
                    "phase": diagnostics.phase_by_frame[i],
                    "server_role": diagnostics.server_role_by_frame[i],
                    "ready_recent": diagnostics.ready_recent_flags[i],
                    "live_now": diagnostics.live_now_flags[i],
                    "dead_now": diagnostics.dead_now_flags[i],
                    "catch_proxy": diagnostics.catch_proxy_scores[i],
                    "quiet_after_catch": diagnostics.quiet_after_catch_scores[i],
                    "ready_pair": diagnostics.ready_pair[i],
                    "live_pair": diagnostics.live_pair[i],
                    "casual_pair": diagnostics.casual_pair[i],
                    "stand_pair": diagnostics.stand_pair[i],
                    "motion_a": diagnostics.motion_a[i],
                    "motion_b": diagnostics.motion_b[i],
                    "crouch_a": diagnostics.crouch_a[i],
                    "crouch_b": diagnostics.crouch_b[i],
                    "serve_a": diagnostics.serve_a[i],
                    "serve_b": diagnostics.serve_b[i],
                    "upper_a": diagnostics.upper_a[i],
                    "upper_b": diagnostics.upper_b[i],
                    "foot_a": diagnostics.foot_a[i],
                    "foot_b": diagnostics.foot_b[i],
                    "reach_a": diagnostics.reach_a[i],
                    "reach_b": diagnostics.reach_b[i],
                    "approach_a": diagnostics.approach_a[i],
                    "approach_b": diagnostics.approach_b[i],
                }
            )


def export_debug_video(
    video_path_str: str,
    table_weights: str,
    *,
    pose_weights: str,
    out_video_str: str,
    out_csv_str: str,
    stride: int = 2,
    player_margin_px: int = 220,
    start_seconds: float = 0.0,
    max_seconds: float = 80.0,
    imgsz: int = 1280,
) -> None:
    video_path = Path(video_path_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    role_frames, role_state_frames, aligned_frame_indices, diagnostics, roi, fps, width, height = _build_role_tracker_debug_payload(
        video_path,
        table_weights=table_weights,
        pose_weights=pose_weights,
        stride=max(1, int(stride)),
        player_margin_px=int(player_margin_px),
        device=device,
        start_seconds=float(start_seconds),
        max_seconds=float(max_seconds),
        imgsz=int(imgsz),
    )
    tx, ty, tw, th = roi
    render_lookup = {frame_idx: i for i, frame_idx in enumerate(aligned_frame_indices)}

    out_video = Path(out_video_str)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    out_csv = Path(out_csv_str)
    _write_debug_csv(out_csv, aligned_frame_indices, diagnostics)

    clip_start = diagnostics.timestamps[0]
    clip_end = diagnostics.timestamps[-1]
    out_fps = float(fps / max(1, stride))
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height),
    )

    print("Pass 2/2: rendering annotated debug video...")
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, aligned_frame_indices[0])
    frame_idx = aligned_frame_indices[0]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx > aligned_frame_indices[-1]:
            break
        sample_idx = render_lookup.get(frame_idx)
        if sample_idx is None:
            frame_idx += 1
            continue

        current_t = diagnostics.timestamps[sample_idx]
        phase = diagnostics.phase_by_frame[sample_idx]
        server_role = diagnostics.server_role_by_frame[sample_idx] or "-"
        seg_idx, seg = _segment_at_time(diagnostics.segments, current_t)
        seg_label = "NONE"
        if seg is not None:
            seg_label = "LET" if "rally_label_let" in seg.flags else "POINT"

        cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (255, 0, 0), 4)
        cv2.putText(frame, "TABLE ROI", (tx, max(40, ty - 14)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

        role_colors = {"A": (0, 0, 255), "B": (0, 255, 255)}
        current_roles = role_frames.get(frame_idx, {})
        current_role_states = role_state_frames.get(frame_idx, {})
        for role in ("A", "B"):
            if role in current_roles:
                obs = current_roles[role]
                color = role_colors[role]
                x1, y1, x2, y2 = obs.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                cv2.putText(frame, f"{role}", (x1, max(30, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
                for kp in obs.keypoints:
                    kx, ky = int(kp[0]), int(kp[1])
                    if kx > 0 and ky > 0:
                        cv2.circle(frame, (kx, ky), 4, color, -1)
            elif current_role_states.get(role) == "occluded":
                cv2.putText(frame, f"{role}: occluded", (70, 470 if role == "A" else 510), cv2.FONT_HERSHEY_SIMPLEX, 0.9, role_colors[role], 2)
            else:
                cv2.putText(frame, f"{role}: missing", (70, 470 if role == "A" else 510), cv2.FONT_HERSHEY_SIMPLEX, 0.9, role_colors[role], 2)

        cv2.rectangle(frame, (38, 34), (1175, 380), (0, 0, 0), -1)
        cv2.putText(frame, f"t={current_t:7.3f}s  frame={frame_idx}  sample={sample_idx}", (60, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        cv2.putText(frame, f"phase={phase}  server={server_role}  seg={seg_idx or '-'}:{seg_label}", (60, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"ready={diagnostics.ready_pair[sample_idx]:.3f}  live={diagnostics.live_pair[sample_idx]:.3f}  casual={diagnostics.casual_pair[sample_idx]:.3f}  stand={diagnostics.stand_pair[sample_idx]:.3f}",
            (60, 144),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.83,
            (180, 255, 180),
            2,
        )
        cv2.putText(
            frame,
            f"flags  ready_recent={int(diagnostics.ready_recent_flags[sample_idx])}  live_now={int(diagnostics.live_now_flags[sample_idx])}  dead_now={int(diagnostics.dead_now_flags[sample_idx])}",
            (60, 178),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.83,
            (200, 220, 255),
            2,
        )
        cv2.putText(
            frame,
            f"let proxy={diagnostics.catch_proxy_scores[sample_idx]:.3f}  quiet_after_catch={diagnostics.quiet_after_catch_scores[sample_idx]:.3f}",
            (60, 212),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.83,
            (255, 210, 140),
            2,
        )
        cv2.putText(
            frame,
            f"A  motion={diagnostics.motion_a[sample_idx]:.3f} crouch={diagnostics.crouch_a[sample_idx]:.3f} serve={diagnostics.serve_a[sample_idx]:.3f} upper={diagnostics.upper_a[sample_idx]:.3f}",
            (60, 258),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            role_colors["A"],
            2,
        )
        cv2.putText(
            frame,
            f"A  foot={diagnostics.foot_a[sample_idx]:.3f} reach={diagnostics.reach_a[sample_idx]:.3f} approach={diagnostics.approach_a[sample_idx]:.3f}",
            (60, 292),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            role_colors["A"],
            2,
        )
        cv2.putText(
            frame,
            f"B  motion={diagnostics.motion_b[sample_idx]:.3f} crouch={diagnostics.crouch_b[sample_idx]:.3f} serve={diagnostics.serve_b[sample_idx]:.3f} upper={diagnostics.upper_b[sample_idx]:.3f}",
            (60, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            role_colors["B"],
            2,
        )
        cv2.putText(
            frame,
            f"B  foot={diagnostics.foot_b[sample_idx]:.3f} reach={diagnostics.reach_b[sample_idx]:.3f} approach={diagnostics.approach_b[sample_idx]:.3f}",
            (60, 364),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            role_colors["B"],
            2,
        )

        _draw_timeline(
            frame,
            diagnostics=diagnostics,
            current_t=current_t,
            clip_start=clip_start,
            clip_end=clip_end,
        )
        writer.write(frame)

        if sample_idx % 200 == 0:
            print(f"  rendered sample {sample_idx}/{len(aligned_frame_indices)}")
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[DONE] Debug video: {out_video}")
    print(f"[DONE] Debug csv:   {out_csv}")
    print(f"[DONE] Segments found: {len(diagnostics.segments)}")
    for i, seg in enumerate(diagnostics.segments, start=1):
        label = "LET" if "rally_label_let" in seg.flags else "POINT"
        print(f"  #{i:02d} {seg.t_start:.3f}->{seg.t_end:.3f} {label} conf={seg.confidence:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export annotated player state machine debug video and CSV.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--weights", default="weights/yolov8x_table.pt", help="Table ROI weights path")
    parser.add_argument("--pose-weights", default="weights/yolov8x-pose.pt", help="Pose weights path")
    parser.add_argument("--out-video", required=True, help="Output MP4 path")
    parser.add_argument("--out-csv", required=True, help="Output CSV path")
    parser.add_argument("--stride", type=int, default=2, help="Sampling stride")
    parser.add_argument("--player-margin-px", type=int, default=220, help="Player vicinity margin around ROI")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Debug clip start time")
    parser.add_argument("--max-seconds", type=float, default=80.0, help="Debug clip duration")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size")
    args = parser.parse_args()

    export_debug_video(
        args.video,
        args.weights,
        pose_weights=args.pose_weights,
        out_video_str=args.out_video,
        out_csv_str=args.out_csv,
        stride=args.stride,
        player_margin_px=args.player_margin_px,
        start_seconds=args.start_seconds,
        max_seconds=args.max_seconds,
        imgsz=args.imgsz,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
