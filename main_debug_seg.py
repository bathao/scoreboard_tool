# main_debug_seg.py
import torch
import sys
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Core Backend Imports
from backend.video_gpu_io import probe_video_ffprobe, nvdec_bgr24_stream
from backend.ai_table_roi_dl import detect_table_roi_dl, DLConfig
from backend.ai_rally_segmentation import detect_rally_segments_gpu
from backend.ai_table_roi import TableROI

def run_unified_debug():
    # --- CONFIGURATION ---
    VIDEO_INPUT = "Vinh_set1.mp4"
    YOLO_WEIGHTS = "weights/yolov8x_table.pt"
    
    # PARAMETERS TO TWEAK (Update these to stabilize segmentation)
    ACTIVE_THRESHOLD = 0.35
    MIN_SEG_SEC = 1.0
    MERGE_GAP_SEC = 0.8

    output_dir = Path("debug_report")
    output_dir.mkdir(exist_ok=True)

    print(f"====================================================")
    print(f"   UNIFIED SEGMENTATION DEBUGGER")
    print(f"   Video: {VIDEO_INPUT}")
    print(f"====================================================")

    video_path = Path(VIDEO_INPUT)
    if not video_path.exists():
        sys.exit(f"CRITICAL: File not found: {VIDEO_INPUT}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = datetime.now()

    # --- STEP 1: ROI & UPZ DETECTION ---
    print(f"[1/3] Detecting ROI and calculating UPZ...")
    info = probe_video_ffprobe(VIDEO_INPUT)
    table_roi = detect_table_roi_dl(VIDEO_INPUT, cfg=DLConfig(weights_path=YOLO_WEIGHTS, device=device))
    upz_roi = table_roi.get_unified_play_zone(info.width, info.height)
    ux, uy, uw, uh = upz_roi
    
    # Save Visual ROI Verification
    cap = cv2.VideoCapture(VIDEO_INPUT)
    ret, frame = cap.read()
    if ret:
        # Draw UPZ (RED) - The area for motion analysis
        cv2.rectangle(frame, (ux, uy), (ux + uw, uy + uh), (0, 0, 255), 8)
        cv2.putText(frame, "UPZ (Motion Analysis Zone)", (ux, uy - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        
        # Draw Table Surface (BLUE)
        tx, ty, tw, th = table_roi.as_tuple()
        cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (255, 0, 0), 6)
        
        roi_check_path = output_dir / "01_roi_upz_check.jpg"
        cv2.imwrite(str(roi_check_path), frame)
        print(f"      Visual check saved: {roi_check_path}")
    cap.release()

    # --- STEP 2: MOTION ANALYSIS ---
    print(f"[2/3] Extracting motion energy from UPZ via GPU...")
    energies, timestamps = [], []
    prev_frame_gpu = None
    stride = 2 
    
    frame_gen = nvdec_bgr24_stream(VIDEO_INPUT, info.width, info.height, crop_roi=upz_roi)
    
    for idx, frame_np in enumerate(frame_gen):
        if idx % stride != 0: continue
        
        curr_gpu = torch.from_numpy(frame_np).to(device).float()
        if prev_frame_gpu is not None:
            # Mean absolute difference is the 'Energy' signal
            diff = torch.abs(curr_gpu - prev_frame_gpu).mean().item()
            energies.append(diff)
            timestamps.append(idx / info.fps)
            
        prev_frame_gpu = curr_gpu
        if idx % 1000 == 0:
            print(f"    > Progress: Frame {idx} processed...", end="\r")

    # Run the segmentation algorithm with current parameters
    segments = detect_rally_segments_gpu(
        energies, 
        timestamps, 
        effective_fps=info.fps/stride,
        active_threshold=ACTIVE_THRESHOLD,
        min_segment_sec=MIN_SEG_SEC,
        merge_gap_sec=MERGE_GAP_SEC
    )
    
    print(f"\n\nRESULT: Detected {len(segments)} rallies.")

    # --- STEP 3: VISUALIZATION ---
    print(f"[3/3] Generating energy analysis plot...")
    plt.figure(figsize=(18, 7))
    
    # Internal normalization matches backend logic
    e_arr = np.array(energies)
    p10, p90 = np.percentile(e_arr, 10), np.percentile(e_arr, 90)
    e_norm = np.clip((e_arr - p10) / (p90 - p10 + 1e-6), 0, 1)
    
    # Plot Energy Signal
    plt.plot(timestamps, e_norm, label="Normalized Motion Energy", color='royalblue', alpha=0.5)
    
    # Plot Threshold Line
    plt.axhline(y=ACTIVE_THRESHOLD, color='red', linestyle='--', label=f"Active Threshold ({ACTIVE_THRESHOLD})")
    
    # Shade Rally Segments
    for i, s in enumerate(segments):
        plt.axvspan(s.t_start, s.t_end, color='limegreen', alpha=0.3)
        plt.text((s.t_start+s.t_end)/2, 0.92, f"#{i+1}", 
                 ha='center', color='darkgreen', fontweight='bold', fontsize=10)

    plt.title(f"Segmentation Analysis: {len(segments)} Rallies (Vinh_set1)")
    plt.xlabel("Seconds (Video Timeline)")
    plt.ylabel("Energy Intensity (0.0 - 1.0)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plot_path = output_dir / "02_energy_analysis.png"
    plt.savefig(str(plot_path), dpi=150)
    
    # Print timestamps for manual video comparison
    print("\nSummary of Detected Rallies:")
    for i, s in enumerate(segments):
        print(f"  Rally #{i+1:02d}: {s.t_start:7.2f}s -> {s.t_end:7.2f}s (Dur: {s.t_end-s.t_start:4.1f}s)")

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n====================================================")
    print(f"Debug Complete in {total_time:.2f}s")
    print(f"Check folder: {output_dir.absolute()}")
    print(f"====================================================")

if __name__ == "__main__":
    run_unified_debug()