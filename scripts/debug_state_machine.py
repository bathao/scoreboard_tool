# scripts/debug_state_machine.py
import torch
import cv2
import numpy as np
import sys
import os
from pathlib import Path
from ultralytics import YOLO

# Ensure backend visibility
sys.path.append(str(Path(__file__).parent.parent))

from backend.video_gpu_io import probe_video_ffprobe
from backend.ai_table_roi_dl import detect_table_roi_dl, DLConfig
from backend.ai_multi_stream_engine import MultiStreamStateMachine, FrameSignal, MatchState

def _calculate_wrist_velocity(curr_kpts, prev_kpts, player_h):
    """
    Calculates velocity normalized by body height (resolution independent).
    Filters out (0,0) jumps and tracking artifacts.
    """
    if prev_kpts is None or player_h <= 0:
        return 0.0
    
    # Indices 9 (Left Wrist), 10 (Right Wrist)
    c_wrists = curr_kpts[9:11]
    p_wrists = prev_kpts[9:11]
    
    valid_velocities = []
    for i in range(2):
        if c_wrists[i][0] > 0 and p_wrists[i][0] > 0:
            dist = np.linalg.norm(c_wrists[i] - p_wrists[i])
            norm_v = dist / player_h
            # Filter teleportation errors (> 50% body height jump in 1/60s)
            if norm_v < 0.5:
                valid_velocities.append(norm_v)
                
    return max(valid_velocities) if valid_velocities else 0.0

def _check_is_low_stance(kpts, player_h):
    """
    Identifies crouching stance (Ready/Active) vs Standing (Idle).
    Uses the vertical ratio between Nose(0) and average Hip(11, 12).
    """
    if player_h <= 0 or kpts[0][1] == 0:
        return False
        
    nose_y = kpts[0][1]
    hip_y = (kpts[11][1] + kpts[12][1]) / 2.0
    
    if hip_y == 0:
        return False
    
    # Calculate upper body compression ratio
    # Standing usually > 0.42, Crouching usually < 0.38
    compression = (hip_y - nose_y) / player_h
    return compression < 0.39

def run_debug_system(video_path_str: str, table_weights_str: str):
    """
    Main Debug Loop with Pose-Aware State Machine.
    """
    print("--- SYSTEM INITIALIZATION (V4 STANCE-AWARE) ---")
    
    # 1. HARDWARE VALIDATION
    if not torch.cuda.is_available():
        print("CRITICAL: CUDA GPU required. CPU processing is disabled.")
        sys.exit(1)
        
    v_path = Path(video_path_str)
    w_path = Path(table_weights_str)

    if not v_path.exists() or not w_path.exists():
        print(f"CRITICAL: Input files not found at {v_path}")
        sys.exit(1)

    device = "cuda"
    print(f"Verified GPU: {torch.cuda.get_device_name(0)}")
    
    # 2. LOAD MODELS
    print("Loading YOLOv8x-pose (Production High-Precision)...")
    person_model = YOLO('yolov8x-pose.pt')
    state_machine = MultiStreamStateMachine()

    # 3. ANCHORS & IO
    info = probe_video_ffprobe(v_path)
    table_roi = detect_table_roi_dl(str(v_path), cfg=DLConfig(weights_path=str(w_path), device=device))
    tx, ty, tw, th = table_roi.as_tuple()
    
    cap = cv2.VideoCapture(str(v_path))
    output_dir = Path("debug_report")
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / "state_machine_v4_stance.mp4"
    
    out = cv2.VideoWriter(str(out_file), 
                         cv2.VideoWriter_fourcc(*'mp4v'), info.fps, (info.width, info.height))

    prev_player_data = {} # {id: {'kpts': ..., 'box': ...}}
    prev_table_gray = None
    frame_idx = 0
    
    print(f"Processing... Output: {out_file.absolute()}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx > 1800: # Limit to 30s
            break

        # A. Neural Inference (Tracking Players)
        results = person_model.track(frame, persist=True, classes=[0], device=device, verbose=False)
        
        # B. Table Activity Analysis
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
                p_height = max(1, by2 - by1)
                
                # Spatial Check: Within table lane + 250px buffer
                is_near = (bx1 < tx + tw + 250) and (bx2 > tx - 250)
                
                # Crouch/Stance Check
                is_low = _check_is_low_stance(kpts[i], p_height)
                
                # Robust Velocity Check
                v_norm = _calculate_wrist_velocity(kpts[i], prev_player_data.get(p_id, {}).get('kpts'), p_height)
                
                prev_player_data[p_id] = {'kpts': kpts[i]}
                player_signals.append({
                    'id': p_id, 'vel': v_norm, 'near': is_near, 'low': is_low
                })

        # D. State Machine Update
        # Consistency: Lock P1/P2 by their unique Tracking IDs
        player_signals.sort(key=lambda x: x['id'])
        p1 = player_signals[0] if len(player_signals) > 0 else {'vel': 0.0, 'near': False, 'low': False}
        p2 = player_signals[1] if len(player_signals) > 1 else {'vel': 0.0, 'near': False, 'low': False}
        
        frame_signal = FrameSignal(
            timestamp=frame_idx / info.fps,
            table_energy=t_energy,
            p1_wrist_vel=p1['vel'], p2_wrist_vel=p2['vel'],
            p1_near_table=p1['near'], p2_near_table=p2['near'],
            p1_is_low=p1['low'], p2_is_low=p2['low']
        )
        
        state_machine.update(frame_signal)

        # E. VISUAL DEBUG OVERLAY (Optimized for 4K)
        # Background Container
        cv2.rectangle(frame, (40, 40), (820, 420), (0,0,0), -1)
        
        # State Monitor
        st_color = (0,0,255) if state_machine.state == MatchState.RALLY else (0,255,0)
        cv2.putText(frame, f"STATE: {state_machine.state.name}", (60, 110), 0, 1.8, st_color, 5)
        
        # Fused Activity Bar
        act = state_machine.last_fused_activity
        target = state_machine.RALLY_START_THRESHOLD
        bar_max = target * 1.5
        bar_w = 650
        fill_w = int(min(1.0, act / bar_max) * bar_w)
        
        cv2.rectangle(frame, (60, 140), (60 + bar_w, 180), (100,100,100), 2)
        cv2.rectangle(frame, (60, 140), (60 + fill_w, 180), (255, 255, 0), -1)
        
        # Threshold Marker
        tx_mark = int((target / bar_max) * bar_w)
        cv2.line(frame, (60 + tx_mark, 130), (60 + tx_mark, 190), (0,0,255), 4)
        
        # Numeric Stats
        cv2.putText(frame, f"Activity: {act:.4f} (Target: {target})", (60, 230), 0, 0.9, (255,255,255), 2)
        
        # Stance & Velocity Stats
        p1_color = (0,255,0) if p1['low'] else (0,0,255)
        p2_color = (0,255,0) if p2['low'] else (0,0,255)
        
        cv2.putText(frame, f"P1 Stance: {'LOW (Ready)' if p1['low'] else 'HIGH (Stand)'}", (60, 290), 0, 0.8, p1_color, 2)
        cv2.putText(frame, f"P1 Norm Vel: {state_machine.last_p1_smoothed_vel:.4f}", (450, 290), 0, 0.7, (200,200,200), 2)
        
        cv2.putText(frame, f"P2 Stance: {'LOW (Ready)' if p2['low'] else 'HIGH (Stand)'}", (60, 350), 0, 0.8, p2_color, 2)
        cv2.putText(frame, f"P2 Norm Vel: {state_machine.last_p2_smoothed_vel:.4f}", (450, 350), 0, 0.7, (200,200,200), 2)

        # F. Draw keypoints for stance verification on players
        for i, player in enumerate(player_signals):
            # Nose and Hips for Crouch Logic check
            k = prev_player_data[player['id']]['kpts']
            cv2.circle(frame, (int(k[0][0]), int(k[0][1])), 8, (0, 255, 255), -1) # Nose
            cv2.circle(frame, (int(k[11][0]), int(k[11][1])), 8, (255, 0, 255), -1) # L-Hip
            cv2.circle(frame, (int(k[12][0]), int(k[12][1])), 8, (255, 0, 255), -1) # R-Hip

        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx} processed...")

    cap.release()
    out.release()
    print(f"\n--- DEBUG FINISHED ---")
    print(f"Output saved to: {out_file.absolute()}")

if __name__ == "__main__":
    # Config
    VIDEO = "Vinh_set1.mp4"
    WEIGHTS = "weights/yolov8x_table.pt"
    
    run_debug_system(VIDEO, WEIGHTS)