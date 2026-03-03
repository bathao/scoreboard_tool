# main.py
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import sys
import os
import argparse
from pathlib import Path

# Ensure backend visibility
sys.path.append(str(Path(__file__).parent.parent))

from backend.video_gpu_io import probe_video_ffprobe, nvdec_bgr24_stream
from backend.ai_table_roi_dl import detect_table_roi_dl, DLConfig
from backend.ai_rally_segmentation import detect_rally_segments_advanced_gpu

def format_time(seconds: float) -> str:
    """Converts seconds to MM:SS format for video cross-referencing."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def run_production_pipeline(video_path_str: str, table_weights_str: str):
    """
    Main Production Pipeline with Visual ROI Verification.
    Optimized for RTX 5060 Ti.
    """
    v_path = Path(video_path_str).absolute()
    w_path = Path(table_weights_str).absolute()

    print(f"\n" + "="*80)
    print(f" TARGET VIDEO: {v_path.name}")
    print(f" STATUS: STARTING PRODUCTION PIPELINE")
    print("="*80)

    # 1. STRICT HARDWARE VALIDATION
    if not torch.cuda.is_available():
        print("CRITICAL ERROR: CUDA GPU is required. Analysis disabled on CPU.")
        sys.exit(1)

    if not v_path.exists():
        print(f"CRITICAL ERROR: Video file not found: {v_path}")
        sys.exit(1)

    device = "cuda"
    print(f"Hardware Verified: {torch.cuda.get_device_name(0)}")

    # 2. ANCHORING: Precise Table Surface
    info = probe_video_ffprobe(str(v_path))
    print(f"Step 1: Locating Table Anchor...")
    table_roi = detect_table_roi_dl(str(v_path), cfg=DLConfig(weights_path=str(w_path), device=device))
    tx, ty, tw, th = table_roi.as_tuple()
    print(f"      Table Anchor: {tw}x{th} at ({tx}, {ty})")

    # --- NEW: ROI VERIFICATION IMAGE GENERATION ---
    debug_dir = Path("debug_report")
    debug_dir.mkdir(exist_ok=True)
    
    cap_verify = cv2.VideoCapture(str(v_path))
    ret, frame = cap_verify.read()
    if ret:
        # Draw the ROI box in RED
        cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 0, 255), 4)
        # Label the coordinates
        cv2.putText(frame, f"TABLE ROI: {tx},{ty} {tw}x{th}", (tx, ty - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        roi_img_path = debug_dir / f"01_roi_verification_{v_path.stem}.jpg"
        cv2.imwrite(str(roi_img_path), frame)
        print(f"  [SUCCESS] ROI verification image saved to: {roi_img_path}")
    cap_verify.release()

    # 3. GPU SIGNAL EXTRACTION (Table Only)
    print("\nStep 2: GPU Motion Energy Extraction...")
    energies = []
    timestamps = []
    prev_frame_gpu = None
    stride = 2 
    
    # Stream frames through NVDEC
    frame_gen = nvdec_bgr24_stream(str(v_path), info.width, info.height, crop_roi=(tx, ty, tw, th))
    
    for idx, frame_np in enumerate(frame_gen):
        if idx % stride != 0: continue
            
        curr_gpu = torch.from_numpy(frame_np).to(device).float()
        
        if prev_frame_gpu is not None:
            # Temporal Difference
            diff = torch.abs(curr_gpu - prev_frame_gpu)
            
            # Morphological Dilation on GPU (MaxPool)
            diff_max = F.max_pool2d(
                diff.permute(2, 0, 1).unsqueeze(0), kernel_size=3, stride=1, padding=1
            )
            
            energies.append(diff_max.mean().item())
            timestamps.append(idx / info.fps)
            
        prev_frame_gpu = curr_gpu
        if idx % 1000 == 0:
            sys.stdout.write(f"\r    > GPU Processing: {idx} frames analyzed...")
            sys.stdout.flush()

    if not energies:
        sys.exit("\nERROR: No signal captured.")

    # 4. SEGMENTATION (Using sync logic with backend)
    print("\n\nStep 3: Signal Refinement & Segmentation...")
    segments = detect_rally_segments_advanced_gpu(
        energies, timestamps, effective_fps=info.fps/stride
    )

    # 5. RESULTS REPORT
    print("\n" + "═"*80)
    print(f" PIPELINE RESULTS FOR: {v_path.name}")
    print(f" TOTAL RALLIES FOUND: {len(segments)}")
    print("═"*80)
    print(f"{'No.':<4} │ {'Start (MM:SS)':<15} │ {'End (MM:SS)':<15} │ {'Dur':<6}")
    print("─"*70)
    for i, s in enumerate(segments):
        print(f" #{i+1:02d} │ {format_time(s.t_start)} ({s.t_start:6.2f}s)   │ {format_time(s.t_end)} ({s.t_end:6.2f}s)   │ {s.t_end-s.t_start:4.1f}s")
    print("═"*80)
    print(f" Done: {v_path.name}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table Tennis Rally Detection Pipeline")
    parser.add_argument("input", help="Path to video file")
    parser.add_argument("--weights", default="weights/yolov8x_table.pt", help="YOLO weights path")
    
    args = parser.parse_args()
    run_production_pipeline(args.input, args.weights)