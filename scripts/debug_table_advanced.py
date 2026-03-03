# scripts/debug_table_advanced.py
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Ensure backend visibility
sys.path.append(str(Path(__file__).parent.parent))

from backend.video_gpu_io import probe_video_ffprobe, nvdec_bgr24_stream
from backend.ai_table_roi_dl import detect_table_roi_dl, DLConfig

def format_time(seconds: float) -> str:
    """Converts seconds to MM:SS format for video cross-referencing."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def run_advanced_table_pipeline(video_path_str: str, table_weights_str: str):
    """
    Advanced Table-Only Rally Detection.
    Optimized for RTX 5060 Ti using GPU-based signal processing.
    Uses Hysteresis thresholding for robust segmentation.
    """
    print("--- STARTING ADVANCED TABLE-ONLY PIPELINE ---")

    # 1. STRICT HARDWARE VALIDATION
    if not torch.cuda.is_available():
        print("CRITICAL ERROR: CUDA GPU is required. High-resolution analysis disabled on CPU.")
        sys.exit(1)

    v_path = Path(video_path_str)
    w_path = Path(table_weights_str)
    if not v_path.exists():
        print(f"CRITICAL ERROR: Video file not found: {v_path}")
        sys.exit(1)

    device = "cuda"
    print(f"Hardware Verified: {torch.cuda.get_device_name(0)}")

    # 2. ANCHORING: Precise Table Surface
    info = probe_video_ffprobe(v_path)
    table_roi = detect_table_roi_dl(str(v_path), cfg=DLConfig(weights_path=str(w_path), device=device))
    tx, ty, tw, th = table_roi.as_tuple()
    print(f"Table Surface Anchor: {tw}x{th} at ({tx}, {ty})")

    # 3. GPU SIGNAL EXTRACTION
    print("\nPhase 1: GPU Motion Energy Extraction...")
    energies = []
    timestamps = []
    prev_frame_gpu = None
    stride = 2 # Process at 30Hz for stability
    
    frame_gen = nvdec_bgr24_stream(str(v_path), info.width, info.height, crop_roi=(tx, ty, tw, th))
    
    for idx, frame_np in enumerate(frame_gen):
        if idx % stride != 0: continue
            
        curr_gpu = torch.from_numpy(frame_np).to(device).float()
        
        if prev_frame_gpu is not None:
            # A. Temporal Difference
            diff = torch.abs(curr_gpu - prev_frame_gpu)
            
            # B. Morphological Dilation on GPU (MaxPool)
            # This amplifies the small ball signal across pixels
            diff_max = F.max_pool2d(
                diff.permute(2, 0, 1).unsqueeze(0), kernel_size=3, stride=1, padding=1
            )
            
            # C. Energy Mean
            energy = diff_max.mean().item()
            energies.append(energy)
            timestamps.append(idx / info.fps)
            
        prev_frame_gpu = curr_gpu
        if idx % 500 == 0:
            print(f"  > GPU Analysis: {idx} frames processed...", end="\r")

    if not energies:
        sys.exit("ERROR: No signal captured.")

    # 4. GPU SIGNAL REFINEMENT (Advanced Smoothing)
    print("\n\nPhase 2: Signal Smoothing & Hysteresis Segmentation...")
    
    # Move raw energies back to GPU for fast convolution
    signal_tensor = torch.tensor(energies, device=device).view(1, 1, -1)
    
    # Gaussian Kernel for smoothing (Sigma 3.0)
    k_size, sigma = 11, 3.0
    gx = torch.arange(k_size).to(device) - (k_size - 1) / 2
    kernel = (torch.exp(-gx.pow(2) / (2 * sigma**2))).view(1, 1, -1)
    kernel /= kernel.sum()
    
    # GPU-based 1D Smoothing
    smoothed_signal = F.conv1d(signal_tensor, kernel, padding=k_size//2).squeeze()
    
    # Normalization (Based on the robust Advanced logic)
    e_np = smoothed_signal.cpu().numpy()
    p10, p95 = np.percentile(e_np, 10), np.percentile(e_np, 95)
    e_norm = np.clip((e_np - p10) / (p95 - p10 + 1e-6), 0.0, 1.0)
    
    t_final = timestamps[:len(e_norm)]

    # 5. HYSTERESIS SEGMENTATION (The Core Logic)
    HIGH_T, LOW_T = 0.25, 0.12 # Triggers and Sustain levels
    MAX_GAP = 2.0              # Max seconds ball stays in air
    MIN_DUR = 1.0              # Min rally duration
    
    rallies = []
    active = False
    s_time, l_time = 0.0, 0.0
    
    for i, val in enumerate(e_norm):
        curr_t = t_final[i]
        if not active:
            if val > HIGH_T:
                active, s_time, l_time = True, curr_t, curr_t
        else:
            if val > LOW_T:
                l_time = curr_t
            if curr_t - l_time > MAX_GAP:
                if l_time - s_time > MIN_DUR:
                    rallies.append((s_time, l_time))
                active = False

    if active and (l_time - s_time > MIN_DUR):
        rallies.append((s_time, l_time))

    # 6. RESULTS
    print("\n" + "═"*70)
    print(f" ADVANCED PIPELINE RESULTS: {len(rallies)} RALLIES")
    print("═"*70)
    print(f"{'No.':<4} │ {'Start (MM:SS)':<15} │ {'End (MM:SS)':<15} │ {'Dur':<6}")
    print("─"*70)
    for i, (s, e) in enumerate(rallies):
        print(f" #{i+1:02d} │ {format_time(s)} ({s:6.2f}s)   │ {format_time(e)} ({e:6.2f}s)   │ {e-s:4.1f}s")
    print("═"*70)

if __name__ == "__main__":
    TARGET_VIDEO = "Vinh_set1.mp4"
    YOLO_WEIGHTS = "weights/yolov8x_table.pt"
    run_advanced_table_pipeline(TARGET_VIDEO, YOLO_WEIGHTS)