# scripts/debug_table_ultra.py
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import sys
from pathlib import Path

# Project root sync
sys.path.append(str(Path(__file__).parent.parent))

from backend.video_gpu_io import probe_video_ffprobe, nvdec_bgr24_stream
from backend.ai_table_roi_dl import detect_table_roi_dl, DLConfig

def format_time(seconds: float) -> str:
    """Converts seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def run_ultra_logic_v5(video_path_str: str, table_weights_str: str):
    """
    Ultra-Optimized V5: Adaptive SNR (Signal-to-Noise Ratio) Detection.
    Uses rolling background estimation on GPU to catch weak rallies and prevent merging.
    """
    print("--- STARTING ULTRA-OPTIMIZED GPU DETECTOR V5 (ADAPTIVE SNR) ---")

    # STRICT HARDWARE CHECK
    if not torch.cuda.is_available():
        sys.exit("CRITICAL ERROR: CUDA GPU required.")

    device = "cuda"
    v_path, w_path = Path(video_path_str), Path(table_weights_str)
    
    # 1. ANCHORING
    info = probe_video_ffprobe(v_path)
    table_roi = detect_table_roi_dl(str(v_path), cfg=DLConfig(weights_path=str(w_path), device=device))
    tx, ty, tw, th = table_roi.as_tuple()
    print(f"Table Anchor: x={tx}, y={ty}, w={tw}, h={th}")

    # 2. GPU ENERGY EXTRACTION
    print("\nPhase 1: GPU Motion Extraction...")
    energies = []
    timestamps = []
    
    frame_gen = nvdec_bgr24_stream(str(v_path), info.width, info.height, crop_roi=(tx, ty, tw, th))
    
    batch_buffer = []
    batch_size = 64
    
    for idx, frame_np in enumerate(frame_gen):
        if idx % 2 != 0: continue # 30Hz analysis
        
        frame_gpu = torch.from_numpy(frame_np).to(device).half()
        batch_buffer.append(frame_gpu)
        timestamps.append(idx / info.fps)
        
        if len(batch_buffer) >= batch_size:
            batch = torch.stack(batch_buffer).permute(0, 3, 1, 2)
            diffs = torch.abs(torch.diff(batch, dim=0))
            # Amplify small ball motion using MaxPool
            diffs_max = F.max_pool2d(diffs, kernel_size=3, stride=1, padding=1)
            energy_batch = diffs_max.mean(dim=(1, 2, 3))
            energies.extend(energy_batch.tolist())
            
            batch_buffer = [batch_buffer[-1]]
            if idx % 1000 == 0:
                print(f"  > GPU Processing: {idx} frames...", end="\r")

    if not energies:
        sys.exit("ERROR: No signal captured.")

    # 3. ADAPTIVE SNR SIGNAL PROCESSING (GPU)
    print("\n\nPhase 2: Adaptive SNR Analysis & Pulse Detection...")
    
    # Move signal to GPU
    S = torch.tensor(energies, device=device).view(1, 1, -1)
    
    # A. Smooth Signal (Short-term)
    k_short = 7
    kernel_s = torch.ones(1, 1, k_short).to(device) / k_short
    short_term = F.conv1d(S, kernel_s, padding=k_size//2 if (k_size:=k_short) else 0)
    
    # B. Background Noise Estimation (Long-term rolling average)
    k_long = 151 # ~5 seconds window at 30Hz
    kernel_l = torch.ones(1, 1, k_long).to(device) / k_long
    background = F.conv1d(S, kernel_l, padding=k_long//2)
    
    # C. Calculate SNR (Signal-to-Noise Ratio)
    # Ratio of current motion vs last 5 seconds of background
    snr_signal = short_term / (background + 1e-6)
    
    # Normalize SNR for segmentation
    # Values > 3.0 usually mean a rally hit
    e_final = snr_signal.squeeze().cpu().numpy()
    t_final = timestamps[:len(e_final)]

    # 4. HYSTERESIS SEGMENTATION
    # START_T: Rally triggers if motion is 2.5x higher than background
    # END_T: Rally ends if motion drops close to background (1.2x)
    START_T, END_T = 2.5, 1.2 
    MAX_GAP = 1.5 # Strict gap for table tennis
    
    rallies = []
    active, s_time, l_time = False, 0.0, 0.0
    
    for i, val in enumerate(e_final):
        curr_t = t_final[i]
        if not active:
            if val > START_T:
                active, s_time, l_time = True, curr_t, curr_t
        else:
            if val > END_T:
                l_time = curr_t
            if curr_t - l_time > MAX_GAP:
                duration = l_time - s_time
                if duration > 1.0:
                    rallies.append((s_time, l_time))
                active = False

    # 5. FINAL REPORT
    print("\n" + "═"*75)
    print(f" ULTRA-PIPELINE V5 (SNR-LOGIC) RESULTS: {len(rallies)} RALLIES")
    print("═"*75)
    print(f"{'No.':<4} │ {'Start (MM:SS)':<15} │ {'End (MM:SS)':<15} │ {'Dur':<6}")
    print("─"*75)
    for i, (s, e) in enumerate(rallies):
        print(f" #{i+1:02d} │ {format_time(s)} ({s:6.2f}s)   │ {format_time(e)} ({e:6.2f}s)   │ {e-s:4.1f}s")
    print("═"*75)

if __name__ == "__main__":
    run_ultra_logic_v5("Vinh_set1.mp4", "weights/yolov8x_table.pt")