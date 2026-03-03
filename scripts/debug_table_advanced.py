# scripts/debug_table_advanced.py
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure backend visibility
sys.path.append(str(Path(__file__).parent.parent))

from backend.video_gpu_io import probe_video_ffprobe, nvdec_bgr24_stream
from backend.ai_table_roi_dl import detect_table_roi_dl, DLConfig

def format_time(seconds: float) -> str:
    """MM:SS format."""
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

def run_advanced_debug_visual(video_path_str: str, table_weights_str: str):
    # --- 1. STRICT FILE VERIFICATION ---
    v_path = Path(video_path_str).absolute()
    w_path = Path(table_weights_str).absolute()
    
    print(f"\n" + "="*80)
    print(f" LOGIC: ADVANCED TABLE-ONLY (VISUAL DEBUG)")
    print(f" PROCESSING FILE: {v_path}") # Kiểm tra đường dẫn tuyệt đối
    print(f" WEIGHTS USED:    {w_path.name}")
    print("="*80)

    if not v_path.exists():
        sys.exit(f"CRITICAL ERROR: File not found: {v_path}")
    if not torch.cuda.is_available():
        sys.exit("CRITICAL ERROR: CUDA GPU Required.")

    device = "cuda"
    
    # 2. GET VIDEO INFO (To see total duration)
    info = probe_video_ffprobe(str(v_path))
    cap_temp = cv2.VideoCapture(str(v_path))
    total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_temp.release()
    
    duration_est = total_frames / info.fps
    print(f"  > Video Properties: {info.width}x{info.height} | {info.fps}fps")
    print(f"  > Total Frames:     {total_frames} (Approx {format_time(duration_est)})")

    # 3. TABLE ANCHORING
    print(f"  > Locating Table ROI for this specific file...")
    table_roi = detect_table_roi_dl(str(v_path), cfg=DLConfig(weights_path=str(w_path), device=device))
    tx, ty, tw, th = table_roi.as_tuple()
    print(f"  > Table Anchor: {tw}x{th} at ({tx}, {ty})")

    # 4. EXTRACTION (NO BREAK/NO LIMIT)
    energies, timestamps = [], []
    prev_frame_gpu = None
    stride = 2 
    
    frame_gen = nvdec_bgr24_stream(str(v_path), info.width, info.height, crop_roi=(tx, ty, tw, th))
    
    print("\nPhase 1: GPU Motion Extraction (Processing Full Stream)...")
    # LƯU Ý: Không có lệnh 'if idx > 9000' ở đây!
    for idx, frame_np in enumerate(frame_gen):
        if idx % stride != 0: continue
        
        curr_gpu = torch.from_numpy(frame_np).to(device).float()
        if prev_frame_gpu is not None:
            diff = torch.abs(curr_gpu - prev_frame_gpu)
            diff_max = F.max_pool2d(diff.permute(2, 0, 1).unsqueeze(0), kernel_size=3, stride=1, padding=1)
            energies.append(diff_max.mean().item())
            timestamps.append(idx / info.fps)
            
        prev_frame_gpu = curr_gpu
        if idx % 500 == 0:
            sys.stdout.write(f"\r    > Progress: Frame {idx}/{total_frames} analyzed...")
            sys.stdout.flush()

    if not energies:
        sys.exit("\nERROR: No motion data captured. Check FFmpeg/NVDEC stream.")

    # 5. SIGNAL REFINEMENT
    print("\n\nPhase 2: Signal Smoothing & Normalization...")
    sig_tensor = torch.tensor(energies, device=device, dtype=torch.float32).view(1, 1, -1)
    k_size, sigma = 11, 3.0
    gx = torch.arange(k_size, device=device, dtype=torch.float32) - (k_size - 1) / 2
    kernel = (torch.exp(-gx.pow(2) / (2 * sigma**2))).view(1, 1, -1)
    kernel /= kernel.sum()
    smoothed = F.conv1d(sig_tensor, kernel, padding=k_size//2).squeeze().cpu().numpy()
    
    # Advanced Normalization
    p10, p95 = np.percentile(smoothed, 10), np.percentile(smoothed, 95)
    e_norm = np.clip((smoothed - p10) / (p95 - p10 + 1e-6), 0.0, 1.0)
    ts_np = np.array(timestamps)

    # 6. HYSTERESIS SEGMENTATION
    HIGH_T, LOW_T = 0.25, 0.12
    MAX_GAP, MIN_DUR = 2.0, 1.0
    rallies, active, s_t, l_t = [], False, 0.0, 0.0
    
    for i, val in enumerate(e_norm):
        t = ts_np[i]
        if not active:
            if val > HIGH_T:
                active, s_t, l_t = True, t, t
        else:
            if val > LOW_T: l_t = t
            if t - l_t > MAX_GAP:
                if l_t - s_t > MIN_DUR: rallies.append((s_t, l_t))
                active = False
    if active and (l_t - s_t > MIN_DUR): rallies.append((s_t, l_t))

    # 7. VISUALIZATION
    print("Phase 3: Saving Energy Chart...")
    plt.figure(figsize=(20, 7))
    plt.plot(ts_np, e_norm, label="Motion Energy", color='blue', alpha=0.5)
    plt.axhline(y=HIGH_T, color='red', linestyle='--', label="Start Threshold")
    
    for i, (s, e) in enumerate(rallies):
        plt.axvspan(s, e, color='green', alpha=0.3)
        plt.text((s+e)/2, 0.95, str(i+1), ha='center', fontsize=8, color='green', fontweight='bold')

    plt.title(f"Analysis: {v_path.name} ({len(rallies)} Rallies)")
    plt.xlabel("Seconds")
    plt.grid(True, alpha=0.2)
    
    plot_path = Path("debug_report") / f"energy_{v_path.stem}.png"
    plt.savefig(str(plot_path), dpi=150)

    # 8. RESULTS
    print("\n" + "═"*80)
    print(f" FINAL REPORT FOR: {v_path.name}")
    print(f" TOTAL RALLIES FOUND: {len(rallies)}")
    print("═"*80)
    for i, (s, e) in enumerate(rallies):
        print(f" #{i+1:02d} │ {format_time(s)} ({s:6.2f}s) ➔ {format_time(e)} ({e:6.2f}s) │ Dur: {e-s:4.1f}s")
    print("═"*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--weights", default="weights/yolov8x_table.pt")
    args = parser.parse_args()
    run_advanced_debug_visual(args.video, args.weights)