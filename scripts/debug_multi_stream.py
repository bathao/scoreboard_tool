# scripts/debug_multi_stream.py
import torch
import cv2
import numpy as np
import sys
import os
from pathlib import Path
from ultralytics import YOLO

# Ensure we can import from the backend folder in the root
sys.path.append(str(Path(__file__).parent.parent))

from backend.video_gpu_io import probe_video_ffprobe
from backend.ai_table_roi_dl import detect_table_roi_dl, DLConfig

def run_multi_stream_debug(video_path_str: str, table_weights: str):
    """
    Advanced Debug Tool for 3-Stream Architecture:
    1. Table Anchor (Fixed)
    2. Player A & B Tracking (Dynamic via YOLOv8x-Pose)
    3. Global Context (Spectator Filtering)
    """
    video_path = Path(video_path_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"--- STARTING PRODUCTION-GRADE MULTI-STREAM DEBUG ---")
    print(f"Hardware: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    # 1. Load the most accurate Pose Model (X-Large)
    # This model provides high-precision keypoints for swing detection
    print("Loading YOLOv8x-Pose model...")
    person_model = YOLO('yolov8x-pose.pt') 

    # 2. Identify Table ROI (The Fixed Anchor)
    info = probe_video_ffprobe(video_path)
    table_roi = detect_table_roi_dl(str(video_path), cfg=DLConfig(weights_path=table_weights, device=device))
    tx, ty, tw, th = table_roi.as_tuple()
    table_center = (tx + tw // 2, ty + th // 2)

    # 3. Setup Video Capture and Output
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    
    output_dir = Path("debug_report")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "multi_stream_tracking_v2.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (info.width, info.height))

    print(f"Tracking players based on proximity to table center: {table_center}")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx > 900: # Debug first 15 seconds for rapid testing
            break

        # A. Detect and Track people with Persistence (ID tracking)
        # Using track() instead of predict() to maintain Player A/B consistency
        results = person_model.track(frame, persist=True, classes=[0], device=device, verbose=False)
        
        candidates = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            keypoints = results[0].keypoints.xy.cpu().numpy() # [N, 17, 2]

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Calculate distance to table center to filter out spectators
                dist_to_table = np.sqrt((cx - table_center[0])**2 + (cy - table_center[1])**2)
                
                candidates.append({
                    'id': ids[i],
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'dist': dist_to_table,
                    'kpts': keypoints[i]
                })

        # B. Selection Logic: Pick the 2 closest people to the table surface
        candidates.sort(key=lambda x: x['dist'])
        active_players = candidates[:2]

        # C. Visualization - Drawing the 3 Streams
        
        # Stream 1: Table (Blue)
        cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (255, 0, 0), 4)
        cv2.putText(frame, "STREAM 1: TABLE ANCHOR", (tx, ty - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        # Stream 2: Dynamic Player Tracking
        for i, player in enumerate(active_players):
            b = player['box']
            p_id = player['id']
            # Color: P1 = Red, P2 = Yellow
            color = (0, 0, 255) if i == 0 else (0, 255, 255)
            label = f"STREAM 2: PLAYER {chr(65+i)} (ID:{p_id})"
            
            # Draw Bounding Box
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 4)
            cv2.putText(frame, label, (b[0], b[1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Draw Keypoints (Pose)
            for kp in player['kpts']:
                kx, ky = int(kp[0]), int(kp[1])
                if kx > 0 and ky > 0:
                    cv2.circle(frame, (kx, ky), 5, (0, 255, 0), -1)
            
            # Highlight Wrists (Index 9 and 10 in COCO pose)
            for wrist_idx in [9, 10]:
                wx, wy = int(player['kpts'][wrist_idx][0]), int(player['kpts'][wrist_idx][1])
                if wx > 0 and wy > 0:
                    cv2.circle(frame, (wx, wy), 10, color, -1) # Highlighting potential swing source

        # Stream 3: Global Info
        cv2.putText(frame, f"GLOBAL: {len(candidates)} people in view", (50, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        cv2.putText(frame, f"STATUS: Tracking 2 Active Players", (50, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx} analyzed...")

    cap.release()
    out.release()
    print(f"\n[DONE] High-precision tracking saved to: {output_path}")
    print(f"Check if IDs are stable and spectators are ignored.")

if __name__ == "__main__":
    # Update these paths to match your local setup
    VIDEO_FILE = "Vinh_set1.mp4"
    YOLO_TABLE_WEIGHTS = "weights/yolov8x_table.pt"
    
    run_multi_stream_debug(VIDEO_FILE, YOLO_TABLE_WEIGHTS)