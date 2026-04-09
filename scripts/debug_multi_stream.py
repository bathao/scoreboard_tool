import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_table_roi_dl import DLConfig, detect_table_roi_dl
from backend.offline_player_tracker import OfflinePlayerTracker
from backend.video_gpu_io import probe_video_ffprobe


def _run_person_detection(
    person_model: YOLO,
    frame: np.ndarray,
    *,
    device: str,
    imgsz: int,
):
    return person_model.predict(frame, classes=[0], device=device, verbose=False, imgsz=imgsz)[0]


def run_multi_stream_debug(
    video_path_str: str,
    table_weights: str,
    *,
    person_weights: str,
    out_path_str: str,
    start_seconds: float = 0.0,
    max_seconds: float = 60.0,
    imgsz: int = 1600,
):
    video_path = Path(video_path_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("--- STARTING PRODUCTION-GRADE MULTI-STREAM DEBUG ---")
    print(f"Hardware: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print("Loading YOLO person detector...")
    person_model = YOLO(str(Path(person_weights).resolve()))

    info = probe_video_ffprobe(video_path)
    table_roi = detect_table_roi_dl(str(video_path), cfg=DLConfig(weights_path=table_weights, device=device))
    tx, ty, tw, th = table_roi.as_tuple()
    table_center = (tx + tw // 2, ty + th // 2)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    start_frame = int(max(0.0, start_seconds) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    max_frames = start_frame + int(max(1.0, max_seconds) * fps)

    output_path = Path(out_path_str)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (info.width, info.height))

    print(f"Tracking players based on proximity to table center: {table_center}")
    print(f"Output: {output_path}")
    print("Analyzing full clip offline for tracklets and global A/B assignment...")

    offline_tracker = OfflinePlayerTracker(table_roi, frame_w=info.width, frame_h=info.height)
    frame_idx = start_frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break
        det_res = _run_person_detection(person_model, frame, device=device, imgsz=imgsz)
        if det_res.boxes is not None and len(det_res.boxes) > 0:
            boxes = det_res.boxes.xyxy.cpu().numpy()
            confs = det_res.boxes.conf.cpu().numpy() if det_res.boxes.conf is not None else None
            detections = offline_tracker.build_detections(
                frame,
                frame_idx=frame_idx,
                boxes_xyxy=boxes,
                keypoints_xy=None,
                confidences=confs,
            )
            offline_tracker.add_frame_detections(detections)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Analysis frame {frame_idx}...")

    offline_result = offline_tracker.finish()
    offline_role_frames = offline_result.role_frames
    offline_role_state_frames = offline_result.role_state_frames
    print(f"Offline analysis complete. Tracklets: {len(offline_result.tracklets)}")

    cap.release()
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    role_states = {
        "A": type("RoleState", (), {"confidence": 1.0, "visible": False, "missing_frames": 0})(),
        "B": type("RoleState", (), {"confidence": 1.0, "visible": False, "missing_frames": 0})(),
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        active_players = offline_role_frames.get(frame_idx, {})
        active_states = offline_role_state_frames.get(frame_idx, {})
        for role in ("A", "B"):
            state = role_states[role]
            if role in active_players:
                state.visible = True
                state.missing_frames = 0
            elif active_states.get(role) == "occluded":
                state.visible = False
                state.missing_frames = 0
            else:
                state.visible = False
                state.missing_frames += 1

        cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (255, 0, 0), 4)
        cv2.putText(frame, "STREAM 1: TABLE ANCHOR", (tx, ty - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        role_colors = {"A": (0, 0, 255), "B": (0, 255, 255)}
        for role in ("A", "B"):
            if role not in active_players:
                continue
            player = active_players[role]
            b = player.box
            color = role_colors[role]
            stream_name = "STREAM 2" if role == "A" else "STREAM 3"
            label = f"{stream_name}: PLAYER {role} (conf={player.confidence:.2f})"
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 4)
            cv2.putText(frame, label, (b[0], b[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            for kp in player.keypoints:
                kx, ky = int(kp[0]), int(kp[1])
                if kx > 0 and ky > 0:
                    cv2.circle(frame, (kx, ky), 5, (0, 255, 0), -1)

        cv2.putText(frame, f"GLOBAL: {len(active_players)} people in view", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        tracked_roles = ", ".join(
            (
                f"{role} visible"
                if role_states[role].visible
                else (f"{role} occluded" if active_states.get(role) == "occluded" else f"{role} missing({role_states[role].missing_frames})")
            )
            for role in ("A", "B")
        )
        cv2.putText(frame, f"STATUS: {tracked_roles}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx} analyzed...")

    cap.release()
    out.release()
    print(f"\n[DONE] High-precision tracking saved to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a debug video with table ROI and offline player tracking.")
    parser.add_argument("--video", default="inputs/debug_sets/match_vinh_001/set_01.mp4", help="Input video path")
    parser.add_argument("--weights", default="weights/yolov8x_table.pt", help="Table YOLO weights path")
    parser.add_argument("--person-weights", default="weights/yolov8s.pt", help="Person YOLO weights path")
    parser.add_argument("--out", default="debug_report/multi_stream_tracking_v2.mp4", help="Output video path")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Start analysis/render from this second")
    parser.add_argument("--max-seconds", type=float, default=60.0, help="Maximum output duration in seconds")
    parser.add_argument("--imgsz", type=int, default=1600, help="YOLO inference image size")
    args = parser.parse_args()

    run_multi_stream_debug(
        args.video,
        args.weights,
        person_weights=args.person_weights,
        out_path_str=args.out,
        start_seconds=args.start_seconds,
        max_seconds=args.max_seconds,
        imgsz=args.imgsz,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

