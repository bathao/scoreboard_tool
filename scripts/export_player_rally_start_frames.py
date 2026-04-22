from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import torch

sys.path.append(str(Path(__file__).parent.parent))

from scripts.export_player_state_machine_debug import _build_role_tracker_debug_payload


def _read_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return frame


def _annotate_start_frame(
    frame,
    *,
    roi,
    frame_idx: int,
    event_idx: int,
    trigger_t: float,
    segment_start_t: float,
    server_role: str,
    reason: str,
    score: float,
    serve_driver_a: float,
    serve_driver_b: float,
    react_to_a: float,
    react_to_b: float,
    active_roles,
):
    tx, ty, tw, th = roi
    cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (255, 0, 0), 4)
    cv2.putText(frame, "TABLE ROI", (tx, max(30, ty - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

    role_colors = {"A": (0, 0, 255), "B": (0, 255, 255)}
    for role in ("A", "B"):
        obs = active_roles.get(role)
        if obs is None:
            continue
        color = role_colors[role]
        x1, y1, x2, y2 = obs.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cv2.putText(frame, f"{role}", (x1, max(35, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        for kp in obs.keypoints:
            kx, ky = int(kp[0]), int(kp[1])
            if kx > 0 and ky > 0:
                cv2.circle(frame, (kx, ky), 4, color, -1)

    cv2.rectangle(frame, (40, 40), (1320, 250), (0, 0, 0), -1)
    cv2.putText(frame, f"RALLY START #{event_idx}", (60, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(frame, f"trigger_t={trigger_t:.3f}s  segment_start_t={segment_start_t:.3f}s  frame={frame_idx}", (60, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"server={server_role or '-'}  reason={reason}  score={score:.3f}", (60, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 255, 255), 2)
    cv2.putText(frame, f"serve_driver_a={serve_driver_a:.3f}  react_to_a={react_to_a:.3f}", (60, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 170, 255), 2)
    cv2.putText(frame, f"serve_driver_b={serve_driver_b:.3f}  react_to_b={react_to_b:.3f}", (60, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 170, 255), 2)
    return frame


def export_rally_start_frames(
    video_path_str: str,
    table_weights: str,
    *,
    pose_weights: str,
    out_dir_str: str,
    stride: int = 2,
    player_margin_px: int = 220,
    start_seconds: float = 0.0,
    max_seconds: float = 0.0,
    imgsz: int = 1280,
) -> None:
    video_path = Path(video_path_str)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required: torch.cuda.is_available() returned False for rally-start frame export.")
    device = "cuda"
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    role_frames, _role_state_frames, aligned_frame_indices, diagnostics, roi, _fps, _width, _height = _build_role_tracker_debug_payload(
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

    csv_path = out_dir / "rally_start_frames.csv"
    rows = []
    print(f"Exporting {len(diagnostics.start_events)} rally-start frames...")
    for i, event in enumerate(diagnostics.start_events, start=1):
        trigger_frame_idx = aligned_frame_indices[event.trigger_sample_idx]
        frame = _read_frame(video_path, trigger_frame_idx)
        active_roles = role_frames.get(trigger_frame_idx, {})
        annotated = _annotate_start_frame(
            frame,
            roi=roi,
            frame_idx=trigger_frame_idx,
            event_idx=i,
            trigger_t=event.trigger_timestamp,
            segment_start_t=event.segment_start_timestamp,
            server_role=event.server_role,
            reason=event.reason,
            score=event.score,
            serve_driver_a=event.serve_driver_a,
            serve_driver_b=event.serve_driver_b,
            react_to_a=event.react_to_a,
            react_to_b=event.react_to_b,
            active_roles=active_roles,
        )
        image_name = f"start_{i:04d}_{event.trigger_timestamp:08.3f}s.jpg"
        image_path = out_dir / image_name
        cv2.imwrite(str(image_path), annotated)
        rows.append(
            {
                "event_id": i,
                "image_file": image_name,
                "trigger_timestamp": f"{event.trigger_timestamp:.6f}",
                "segment_start_timestamp": f"{event.segment_start_timestamp:.6f}",
                "trigger_frame_idx": trigger_frame_idx,
                "segment_start_frame_idx": aligned_frame_indices[event.segment_start_sample_idx],
                "server_role": event.server_role,
                "reason": event.reason,
                "score": f"{event.score:.6f}",
                "serve_driver_a": f"{event.serve_driver_a:.6f}",
                "serve_driver_b": f"{event.serve_driver_b:.6f}",
                "react_to_a": f"{event.react_to_a:.6f}",
                "react_to_b": f"{event.react_to_b:.6f}",
            }
        )
        print(f"  #{i:02d} trigger={event.trigger_timestamp:.3f}s -> {image_path.name}")

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "event_id",
                "image_file",
                "trigger_timestamp",
                "segment_start_timestamp",
                "trigger_frame_idx",
                "segment_start_frame_idx",
                "server_role",
                "reason",
                "score",
                "serve_driver_a",
                "serve_driver_b",
                "react_to_a",
                "react_to_b",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Images: {out_dir}")
    print(f"[DONE] CSV:    {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export rally-start trigger frames from the current player-only state machine.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--weights", default="weights/yolov8x_table.pt", help="Table ROI weights")
    parser.add_argument("--pose-weights", default="weights/yolov8x-pose.pt", help="Pose weights")
    parser.add_argument("--out-dir", required=True, help="Output directory for JPG files and CSV")
    parser.add_argument("--stride", type=int, default=2, help="Sampling stride")
    parser.add_argument("--player-margin-px", type=int, default=220, help="Player vicinity margin around ROI")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Clip start time")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Clip duration; <=0 means full clip")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size")
    args = parser.parse_args()

    export_rally_start_frames(
        args.video,
        args.weights,
        pose_weights=args.pose_weights,
        out_dir_str=args.out_dir,
        stride=args.stride,
        player_margin_px=args.player_margin_px,
        start_seconds=args.start_seconds,
        max_seconds=args.max_seconds,
        imgsz=args.imgsz,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
