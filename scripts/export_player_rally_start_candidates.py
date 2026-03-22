from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import torch

sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_multistream_rally import (
    _compute_player_rally_start_candidates,
    _compute_player_state_machine_diagnostics,
    _select_player_sandwich_start_candidates,
    extract_multistream_signals,
)
from scripts.export_player_state_machine_debug import _build_role_tracker_debug_payload


def _read_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return frame


def _annotate_candidate_frame(
    frame,
    *,
    roi,
    frame_idx: int,
    candidate_idx: int,
    timestamp: float,
    role: str,
    score: float,
    prep_score: float,
    launch_score: float,
    opponent_ready_score: float,
    dominance_ratio: float,
    episode_start_t: float,
    episode_peak_t: float,
    ready_pair_score: float,
    pre_ready_mean: float,
    pre_live_peak: float,
    server_action_score: float,
    server_peak_score: float,
    server_growth_score: float,
    server_peak_delay_sec: float,
    receiver_peak_score: float,
    live_peak_score: float,
    active_roles,
):
    tx, ty, tw, th = roi
    cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (255, 0, 0), 4)
    cv2.putText(frame, "TABLE ROI", (tx, max(30, ty - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

    role_colors = {"A": (0, 0, 255), "B": (0, 255, 255)}
    candidate_color = role_colors.get(role, (0, 255, 0))
    for role_name in ("A", "B"):
        obs = active_roles.get(role_name)
        if obs is None:
            continue
        color = role_colors[role_name]
        thickness = 6 if role_name == role else 3
        x1, y1, x2, y2 = obs.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, role_name, (x1, max(35, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        for kp in obs.keypoints:
            kx, ky = int(kp[0]), int(kp[1])
            if kx > 0 and ky > 0:
                cv2.circle(frame, (kx, ky), 4, color, -1)

    cv2.rectangle(frame, (40, 40), (1460, 370), (0, 0, 0), -1)
    cv2.putText(frame, f"RALLY START CANDIDATE #{candidate_idx}", (60, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(frame, f"t={timestamp:.3f}s  frame={frame_idx}  role={role}", (60, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.95, candidate_color, 3)
    cv2.putText(frame, f"score={score:.3f}  prep={prep_score:.3f}  launch={launch_score:.3f}", (60, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"opp_ready={opponent_ready_score:.3f}  dominance={dominance_ratio:.2f}", (60, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"episode_start={episode_start_t:.3f}s  episode_peak={episode_peak_t:.3f}s", (60, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 255, 180), 2)
    cv2.putText(frame, f"ready_pair={ready_pair_score:.3f}  pre_ready={pre_ready_mean:.3f}  pre_live={pre_live_peak:.3f}", (60, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.84, (255, 240, 180), 2)
    cv2.putText(frame, f"server_now={server_action_score:.3f}  peak={server_peak_score:.3f}  growth={server_growth_score:.3f}  delay={server_peak_delay_sec:.3f}s", (60, 308), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (180, 255, 255), 2)
    cv2.putText(frame, f"receiver_peak={receiver_peak_score:.3f}  live_peak={live_peak_score:.3f}", (60, 344), cv2.FONT_HERSHEY_SIMPLEX, 0.84, (180, 255, 180), 2)
    return frame


def export_rally_start_candidates(
    video_path_str: str,
    table_weights: str,
    *,
    pose_weights: str,
    out_dir_str: str,
    selection_mode: str = "raw",
    stride: int = 2,
    player_margin_px: int = 220,
    start_seconds: float = 0.0,
    max_seconds: float = 0.0,
    imgsz: int = 1280,
) -> None:
    video_path = Path(video_path_str)
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    role_frames, _role_state_frames, aligned_frame_indices, diagnostics, roi, _fps, _width, _height = _build_role_tracker_debug_payload(
        video_path,
        table_weights=table_weights,
        pose_weights=pose_weights,
        stride=max(1, int(stride)),
        player_margin_px=int(player_margin_px),
        device="cuda" if torch.cuda.is_available() else "cpu",
        start_seconds=float(start_seconds),
        max_seconds=float(max_seconds),
        imgsz=int(imgsz),
    )

    production_diagnostics = diagnostics
    if float(start_seconds) <= 0.0 and float(max_seconds) <= 0.0:
        signals = extract_multistream_signals(
            str(video_path),
            table_weights,
            pose_weights_path=pose_weights,
            stride=max(1, int(stride)),
            player_margin_px=int(player_margin_px),
            player_signal_source="role_tracker",
            ball_signal_source="none",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        production_diagnostics = _compute_player_state_machine_diagnostics(signals)

    if selection_mode == "sandwich":
        candidates = _select_player_sandwich_start_candidates(production_diagnostics)
    else:
        candidates = _compute_player_rally_start_candidates(production_diagnostics)

    debug_timestamps = diagnostics.timestamps

    def resolve_debug_sample_idx(timestamp: float) -> int:
        if not debug_timestamps:
            return 0
        return min(range(len(debug_timestamps)), key=lambda idx: abs(float(debug_timestamps[idx]) - float(timestamp)))

    csv_path = out_dir / "rally_start_candidates.csv"
    rows = []
    print(f"Exporting {len(candidates)} rally-start candidate frames (selection_mode={selection_mode})...")
    for i, candidate in enumerate(candidates, start=1):
        debug_sample_idx = resolve_debug_sample_idx(candidate.timestamp)
        frame_idx = aligned_frame_indices[debug_sample_idx]
        frame = _read_frame(video_path, frame_idx)
        active_roles = role_frames.get(frame_idx, {})
        annotated = _annotate_candidate_frame(
            frame,
            roi=roi,
            frame_idx=frame_idx,
            candidate_idx=i,
            timestamp=candidate.timestamp,
            role=candidate.role,
            score=candidate.score,
            prep_score=candidate.prep_score,
            launch_score=candidate.launch_score,
            opponent_ready_score=candidate.opponent_ready_score,
            dominance_ratio=candidate.dominance_ratio,
            episode_start_t=production_diagnostics.timestamps[candidate.episode_start_sample_idx],
            episode_peak_t=production_diagnostics.timestamps[candidate.episode_peak_sample_idx],
            ready_pair_score=candidate.ready_pair_score,
            pre_ready_mean=candidate.pre_ready_mean,
            pre_live_peak=candidate.pre_live_peak,
            server_action_score=candidate.server_action_score,
            server_peak_score=candidate.server_peak_score,
            server_growth_score=candidate.server_growth_score,
            server_peak_delay_sec=candidate.server_peak_delay_sec,
            receiver_peak_score=candidate.receiver_peak_score,
            live_peak_score=candidate.live_peak_score,
            active_roles=active_roles,
        )
        image_name = f"candidate_{i:04d}_{candidate.timestamp:08.3f}s_role{candidate.role}.jpg"
        image_path = out_dir / image_name
        cv2.imwrite(str(image_path), annotated)
        rows.append(
            {
                "candidate_id": i,
                "image_file": image_name,
                "timestamp": f"{candidate.timestamp:.6f}",
                "frame_idx": frame_idx,
                "role": candidate.role,
                "selection_mode": selection_mode,
                "debug_sample_idx": debug_sample_idx,
                "production_sample_idx": candidate.sample_idx,
                "score": f"{candidate.score:.6f}",
                "prep_score": f"{candidate.prep_score:.6f}",
                "launch_score": f"{candidate.launch_score:.6f}",
                "opponent_ready_score": f"{candidate.opponent_ready_score:.6f}",
                "dominance_ratio": f"{candidate.dominance_ratio:.6f}",
                "episode_start_timestamp": f"{production_diagnostics.timestamps[candidate.episode_start_sample_idx]:.6f}",
                "episode_peak_timestamp": f"{production_diagnostics.timestamps[candidate.episode_peak_sample_idx]:.6f}",
                "episode_end_timestamp": f"{production_diagnostics.timestamps[candidate.episode_end_sample_idx]:.6f}",
                "crouch_score": f"{candidate.crouch_score:.6f}",
                "reach_score": f"{candidate.reach_score:.6f}",
                "serve_score": f"{candidate.serve_score:.6f}",
                "upper_body_score": f"{candidate.upper_body_score:.6f}",
                "footwork_score": f"{candidate.footwork_score:.6f}",
                "ready_pair_score": f"{candidate.ready_pair_score:.6f}",
                "pre_ready_mean": f"{candidate.pre_ready_mean:.6f}",
                "pre_live_peak": f"{candidate.pre_live_peak:.6f}",
                "server_action_score": f"{candidate.server_action_score:.6f}",
                "server_peak_score": f"{candidate.server_peak_score:.6f}",
                "server_growth_score": f"{candidate.server_growth_score:.6f}",
                "server_peak_delay_sec": f"{candidate.server_peak_delay_sec:.6f}",
                "receiver_peak_score": f"{candidate.receiver_peak_score:.6f}",
                "live_peak_score": f"{candidate.live_peak_score:.6f}",
            }
        )
        print(f"  #{i:02d} t={candidate.timestamp:.3f}s role={candidate.role} -> {image_path.name}")

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "candidate_id",
                "image_file",
                "timestamp",
                "frame_idx",
                "role",
                "selection_mode",
                "debug_sample_idx",
                "production_sample_idx",
                "score",
                "prep_score",
                "launch_score",
                "opponent_ready_score",
                "dominance_ratio",
                "episode_start_timestamp",
                "episode_peak_timestamp",
                "episode_end_timestamp",
                "crouch_score",
                "reach_score",
                "serve_score",
                "upper_body_score",
                "footwork_score",
                "ready_pair_score",
                "pre_ready_mean",
                "pre_live_peak",
                "server_action_score",
                "server_peak_score",
                "server_growth_score",
                "server_peak_delay_sec",
                "receiver_peak_score",
                "live_peak_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Images: {out_dir}")
    print(f"[DONE] CSV:    {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export rally-start candidate frames from the new player-only serve-start detector.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--weights", default="weights/yolov8x_table.pt", help="Table ROI weights")
    parser.add_argument("--pose-weights", default="weights/yolov8x-pose.pt", help="Pose weights")
    parser.add_argument("--out-dir", required=True, help="Output directory for JPG files and CSV")
    parser.add_argument("--selection-mode", choices=["raw", "sandwich"], default="sandwich", help="Which start list to export")
    parser.add_argument("--stride", type=int, default=2, help="Sampling stride")
    parser.add_argument("--player-margin-px", type=int, default=220, help="Player vicinity margin around ROI")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Clip start time")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Clip duration; <=0 means full clip")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size")
    args = parser.parse_args()

    export_rally_start_candidates(
        args.video,
        args.weights,
        pose_weights=args.pose_weights,
        out_dir_str=args.out_dir,
        selection_mode=args.selection_mode,
        stride=args.stride,
        player_margin_px=args.player_margin_px,
        start_seconds=args.start_seconds,
        max_seconds=args.max_seconds,
        imgsz=args.imgsz,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
