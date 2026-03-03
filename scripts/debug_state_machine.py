# scripts/debug_state_machine.py
import torch
import cv2
import numpy as np
import sys
import os
from pathlib import Path
from ultralytics import YOLO

# Project root sync
sys.path.append(str(Path(__file__).parent.parent))

from backend.video_gpu_io import probe_video_ffprobe
from backend.ai_table_roi_dl import detect_table_roi_dl, DLConfig
from backend.ai_multi_stream_engine import MultiStreamStateMachine, FrameSignal, MatchState

def _calculate_wrist_velocity(curr_kpts, prev_kpts, player_h):
    """Calculates resolution-independent wrist velocity."""
    if prev_kpts is None or player_h <= 0: return 0.0
    c_wrists, p_wrists = curr_kpts[9:11], prev_kpts[9:11]
    valid_vels = []
    for i in range(2):
        if c_wrists[i][0] > 0 and p_wrists[i][0] > 0:
            dist = np.linalg.norm(c_wrists[i] - p_wrists[i])
            norm_v = dist / player_h
            if norm_v < 0.5: valid_vels.append(norm_v)
    return max(valid_vels) if valid_vels else 0.0

def _get_compression_ratio(kpts, player_h):
    """Calculates vertical ratio between nose and hips for stance detection."""
    if player_h <= 0 or kpts[0][1] == 0: return 1.0
    nose_y = kpts[0][1]
    hip_y = (kpts[11][1] + kpts[12][1]) / 2.0
    if hip_y <= nose_y: return 1.0
    return (hip_y - nose_y) / player_h

def run_debug_v5(video_path_str: str, weights_path_str: str):
    """
    Main Debug Loop with Strict Hardware Validation and Detailed UI.
    """
    print("--- INITIALIZING POSE-AWARE DEBUG SYSTEM (V5) ---")
    
    # 1. STRICT HARDWARE CHECK
    if not torch.cuda.is_available():
        print("CRITICAL ERROR: CUDA GPU NOT FOUND. System must run on GPU.")
        sys.exit(1)
        
    v_path, w_path = Path(video_path_str), Path(weights_path_str)
    if not v_path.exists() or not w_path.exists():
        print(f"CRITICAL ERROR: Input files missing at {v_path}")
        sys.exit(1)

    device = "cuda"
    print(f"GPU Verified: {torch.cuda.get_device_name(0)}")
    
    # 2. LOAD MODELS
    print("Loading YOLOv8x-pose model...")
    person_model = YOLO('yolov8x-pose.pt')
    sm = MultiStreamStateMachine()

    # 3. IO SETUP
    info = probe_video_ffprobe(v_path)
    table_roi = detect_table_roi_dl(str(v_path), cfg=DLConfig(weights_path=str(w_path), device=device))
    tx, ty, tw, th = table_roi.as_tuple()
    
    cap = cv2.VideoCapture(str(v_path))
    output_dir = Path("debug_report")
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / "state_machine_v5_ratios.mp4"
    out = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*'mp4v'), info.fps, (info.width, info.height))

    prev_player_data = {}
    prev_table_gray = None
    frame_idx = 0
    
    print(f"Processing Video... Trace saved to {out_file.absolute()}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx > 2400: # Process 40 seconds
            break

        # A. Track Players
        results = person_model.track(frame, persist=True, classes=[0], device=device, verbose=False)
        
        # B. Table Energy
        t_crop = frame[ty:ty+th, tx:tx+tw]
        t_gray = cv2.cvtColor(t_crop, cv2.COLOR_BGR2GRAY)
        t_energy = 0.0
        if prev_table_gray is not None:
            t_energy = np.mean(cv2.absdiff(t_gray, prev_table_gray))
        prev_table_gray = t_gray

        # C. Player Feature Extraction
        player_signals = []
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            kpts = results[0].keypoints.xy.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            for i, p_id in enumerate(ids):
                bx1, by1, bx2, by2 = boxes[i]
                p_h = max(1, by2 - by1)
                is_near = (bx1 < tx + tw + 250) and (bx2 > tx - 250)
                
                # Ratio Calculation (Small = Crouch, Large = Stand)
                ratio = _get_compression_ratio(kpts[i], p_h)
                # THRESHOLD CALIBRATION: Crouching usually < 0.39
                is_low = ratio < 0.39 
                
                v_norm = _calculate_wrist_velocity(kpts[i], prev_player_data.get(p_id, {}).get('kpts'), p_h)
                
                prev_player_data[p_id] = {'kpts': kpts[i]}
                player_signals.append({
                    'id': p_id, 'vel': v_norm, 'near': is_near, 'low': is_low, 'ratio': ratio
                })

        # D. State Machine Update
        player_signals.sort(key=lambda x: x['id'])
        p1 = player_signals[0] if len(player_signals) > 0 else {'vel': 0, 'near': False, 'low': False, 'ratio': 1.0}
        p2 = player_signals[1] if len(player_signals) > 1 else {'vel': 0, 'near': False, 'low': False, 'ratio': 1.0}
        
        sig = FrameSignal(
            timestamp=frame_idx / info.fps,
            table_energy=t_energy,
            p1_wrist_vel=p1['vel'], p2_wrist_vel=p2['vel'],
            p1_near_table=p1['near'], p2_near_table=p2['near'],
            p1_is_low=p1['low'], p2_is_low=p2['low'],
            p1_ratio=p1['ratio'], p2_ratio=p2['ratio']
        )
        sm.update(sig)

        # E. VISUAL DEBUG UI
        cv2.rectangle(frame, (40, 40), (850, 420), (0,0,0), -1)
        st_color = (0,0,255) if sm.state == MatchState.RALLY else (0,255,0)
        cv2.putText(frame, f"STATE: {sm.state.name}", (60, 110), 0, 1.8, st_color, 5)
        
        # Activity Bar
        act, tgt = sm.last_fused_activity, sm.RALLY_START_THRESHOLD
        fill_w = int(min(1.0, act / (tgt * 1.5)) * 600)
        cv2.rectangle(frame, (60, 140), (660, 170), (100,100,100), 2)
        cv2.rectangle(frame, (60, 140), (60 + fill_w, 170), (255,255,0), -1)
        
        # Ratio & Stance Metrics
        p1_color = (0,255,0) if p1['low'] else (0,0,255)
        p2_color = (0,255,0) if p2['low'] else (0,0,255)
        cv2.putText(frame, f"P1 Ratio: {p1['ratio']:.3f} [{'LOW' if p1['low'] else 'HIGH'}]", (60, 240), 0, 0.8, p1_color, 2)
        cv2.putText(frame, f"P1 Norm Vel: {sm.last_p1_smoothed_vel:.3f}", (550, 240), 0, 0.7, (200,200,200), 2)
        cv2.putText(frame, f"P2 Ratio: {p2['ratio']:.3f} [{'LOW' if p2['low'] else 'HIGH'}]", (60, 300), 0, 0.8, p2_color, 2)
        cv2.putText(frame, f"P2 Norm Vel: {sm.last_p2_smoothed_vel:.3f}", (550, 300), 0, 0.7, (200,200,200), 2)
        cv2.putText(frame, f"Activity: {act:.4f}", (60, 360), 0, 0.9, (255,255,255), 2)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print(f"\n--- DEBUG FINISHED ---")
    print(f"Check: {out_file.absolute()}")

if __name__ == "__main__":
    run_debug_v5("Vinh_set1.mp4", "weights/yolov8x_table.pt")