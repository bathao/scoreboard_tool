# scripts/debug_segmentation.py
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from backend.video_gpu_io import probe_video_ffprobe, nvdec_bgr24_stream
from backend.ai_table_roi_dl import detect_table_roi_dl, DLConfig
from backend.ai_rally_segmentation import detect_rally_segments_gpu

def run_segmentation_debug(video_path_str: str, weights_path_str: str):
    video_path = Path(video_path_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"--- STARTING SEGMENTATION DEBUG: {video_path.name} ---")
    
    # 1. Video Info & ROI
    info = probe_video_ffprobe(video_path)
    # Using the same DL ROI detection as main.py
    roi = detect_table_roi_dl(str(video_path), cfg=DLConfig(weights_path=weights_path_str, device=device))
    print(f"Target ROI: {roi.as_tuple()}")

    # 2. Extract Motion Energy (Same logic as main.py)
    energies = []
    timestamps = []
    prev_frame_gpu = None
    stride = 2 
    
    # Stream frames through NVDEC
    frame_gen = nvdec_bgr24_stream(str(video_path), info.width, info.height, crop_roi=roi.as_tuple())
    
    print("Calculating motion energy from GPU stream...")
    for idx, frame_np in enumerate(frame_gen):
        if idx % stride != 0: continue
        
        curr_gpu = torch.from_numpy(frame_np).to(device).float()
        if prev_frame_gpu is not None:
            # Mean absolute difference as motion energy
            diff = torch.abs(curr_gpu - prev_frame_gpu).mean().item()
            energies.append(diff)
            timestamps.append(idx / info.fps)
        
        prev_frame_gpu = curr_gpu
        if idx % 1000 == 0:
            print(f"  Processed {idx} frames...")

    # 3. Run Segmentation Algorithm
    # Adjust these parameters here to see how they affect the result
    active_threshold = 0.22
    min_segment_sec = 1.0
    merge_gap_sec = 0.8
    
    segments = detect_rally_segments_gpu(
        energies, 
        timestamps, 
        effective_fps=info.fps/stride,
        active_threshold=active_threshold,
        min_segment_sec=min_segment_sec,
        merge_gap_sec=merge_gap_sec
    )

    # 4. Visualization
    print("Generating analysis plot...")
    plt.figure(figsize=(16, 6))
    
    # Normalize energies for plotting (Matches logic in backend)
    e = np.array(energies)
    p10, p90 = np.percentile(e, 10), np.percentile(e, 90)
    e_norm = np.clip((e - p10) / (p90 - p10 + 1e-6), 0, 1)
    
    # Plot raw energy
    plt.plot(timestamps, e_norm, label="Normalized Motion Energy", color='royalblue', alpha=0.6)
    
    # Plot threshold line
    plt.axhline(y=active_threshold, color='red', linestyle='--', label=f"Threshold ({active_threshold})")
    
    # Shade the detected rally areas
    for i, seg in enumerate(segments):
        plt.axvspan(seg.t_start, seg.t_end, color='limegreen', alpha=0.3)
        plt.text((seg.t_start + seg.t_end)/2, 0.95, f"Rally {i+1}", 
                 ha='center', fontsize=9, color='green', fontweight='bold')

    plt.title(f"Segmentation Analysis: {video_path.name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Energy (0-1)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Save the report
    report_path = f"debug_report/seg_analysis_{video_path.stem}.png"
    plt.savefig(report_path, dpi=150)
    print(f"[DONE] Plot saved to: {report_path}")
    
    # Print results for manual timing check
    print("\nDetected Rally Timestamps:")
    for i, seg in enumerate(segments):
        duration = seg.t_end - seg.t_start
        print(f"  #{i+1:02d} | {seg.t_start:7.2f}s -> {seg.t_end:7.2f}s | Duration: {duration:4.2f}s")

if __name__ == "__main__":
    # Configure your paths here
    VIDEO_FILE = "Vinh_set1.mp4"
    YOLO_MODEL = "weights/yolov8x_table.pt"
    
    run_segmentation_debug(VIDEO_FILE, YOLO_MODEL)