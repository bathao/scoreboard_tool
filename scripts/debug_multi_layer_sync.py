# scripts/debug_multi_layer_sync.py
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from pathlib import Path
from ultralytics import YOLO

# Project root sync
sys.path.append(str(Path(__file__).parent.parent))

from backend.video_gpu_io import probe_video_ffprobe, nvdec_bgr24_stream
from backend.ai_multi_stream_engine import segment_hys_logic, RallyEvent
from backend.ai_table_roi_dl import detect_table_roi_dl, DLConfig

def format_time(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

def run_hyper_turbo_sync(video_path_str: str, table_weights_str: str):
    print("--- STARTING HYPER-TURBO PIPELINE (PCIe OPTIMIZED) ---")
    
    if not torch.cuda.is_available():
        sys.exit("CRITICAL ERROR: CUDA GPU Required.")

    device = "cuda"
    v_path, w_path = Path(video_path_str), Path(table_weights_str)
    
    # 1. PRE-FLIGHT: GET RESOLUTION & SCALE
    info = probe_video_ffprobe(v_path)
    
    # Target 1080p for processing to save 75% bandwidth
    TARGET_H = 1080
    scale_ratio = TARGET_H / info.height
    TARGET_W = int(info.width * scale_ratio)
    
    print(f"Original: {info.width}x{info.height} | Processing: {TARGET_W}x{TARGET_H}")

    # 2. LOAD MODELS & ANCHORS (Detect in 4K once, then scale)
    person_model = YOLO('yolov8x-pose.pt')
    table_roi = detect_table_roi_dl(str(v_path), cfg=DLConfig(weights_path=str(w_path), device=device))
    
    # Scale Table ROI coordinates to 1080p
    tx, ty, tw, th = [int(v * scale_ratio) for v in table_roi.as_tuple()]

    # YOLO Internal Batch size & Res (optimized for stride 32)
    YOLO_SIZE = 640
    BATCH_SIZE = 12 # Optimized for 1080p + 5060 Ti 16GB
    
    # 3. EXTRACTION
    raw_table_energies, raw_player_energies, timestamps = [], [], []
    frame_buffer, ts_buffer = [], []
    prev_table_gpu = None
    prev_kpts = {}
    stride = 2 # 30fps analysis
    
    # Start NVDEC Stream with DOWNSIZING (This makes it fast!)
    frame_gen = nvdec_bgr24_stream(str(v_path), TARGET_W, TARGET_H, crop_roi=None)
    
    print(f"Step 2: Dual-Stream Batch Inference...")
    try:
        for idx, frame_np in enumerate(frame_gen):
            if idx % stride != 0: continue
            
            # Fast GPU upload (FP16)
            f_gpu = torch.from_numpy(frame_np).to(device).half()
            frame_buffer.append(f_gpu)
            ts_buffer.append(idx / info.fps)

            if len(frame_buffer) == BATCH_SIZE:
                # Stack to [B, C, H, W]
                batch_tensor = torch.stack(frame_buffer).permute(0, 3, 1, 2).float()
                
                # --- PHASE A: PLAYER TRACKING (BATCH FP16) ---
                yolo_in = F.interpolate(batch_tensor, size=(YOLO_SIZE, YOLO_SIZE), mode='bilinear') / 255.0
                results = person_model.track(yolo_in, persist=True, classes=[0], device=device, verbose=False)
                
                # Coordinates in 'results' are relative to YOLO_SIZE, scale them to 1080p
                yolo_to_proc_scale = TARGET_H / YOLO_SIZE
                
                for b_idx in range(BATCH_SIZE):
                    p_e = 0.0
                    res = results[b_idx]
                    if res.boxes.id is not None:
                        ids = res.boxes.id.cpu().numpy().astype(int)
                        kpts = res.keypoints.xy.cpu().numpy()
                        boxes = res.boxes.xyxy.cpu().numpy()
                        p_v = []
                        for i, p_id in enumerate(ids):
                            # Scale box to 1080p to match scaled Table ROI
                            bx1, by1, bx2, by2 = boxes[i] * yolo_to_proc_scale
                            ph = max(1, by2 - by1)
                            # Near table check (in 1080p units)
                            if (bx1 < tx + tw + 100) and (bx2 > tx - 100):
                                cur_k = kpts[i] * yolo_to_proc_scale
                                if p_id in prev_kpts:
                                    d = np.linalg.norm(cur_k[9:11] - prev_kpts[p_id][9:11])
                                    p_v.append((d / ph) * 1.5)
                                prev_kpts[p_id] = cur_k
                        if p_v: p_e = max(p_v)
                    raw_player_energies.append(p_e)

                # --- PHASE B: TABLE PROCESSING (BATCH ON 1080P) ---
                table_batch = batch_tensor[:, :, ty:ty+th, tx:tx+tw]
                if prev_table_gpu is None: prev_table_gpu = table_batch[0:1]
                table_ext = torch.cat([prev_table_gpu, table_batch], dim=0)
                diffs = torch.abs(torch.diff(table_ext, dim=0))
                diff_m = F.max_pool2d(diffs, kernel_size=3, stride=1, padding=1)
                raw_table_energies.extend(diff_m.mean(dim=(1, 2, 3)).tolist())
                
                prev_table_gpu = table_batch[-1:]
                timestamps.extend(ts_buffer)
                frame_buffer, ts_buffer = [], []
                
                sys.stdout.write(f"\r    > Processing: {idx} frames | Speed: HYPER-TURBO | VRAM: {torch.cuda.memory_allocated()//1024**2}MB")
                sys.stdout.flush()

    except Exception as e:
        print(f"\nERROR: {e}")
    finally:
        print("\nPhase 2 Complete.")

    # 4. REFINEMENT
    def refine_strict(data):
        if not data: return np.array([])
        sig = torch.tensor(data, device=device, dtype=torch.float32).view(1, 1, -1)
        k_size, sigma = 11, 3.0
        gx = torch.arange(k_size, device=device, dtype=torch.float32) - (k_size - 1) / 2
        kernel = (torch.exp(-gx.pow(2) / (2 * sigma**2))).view(1, 1, -1)
        kernel /= kernel.sum()
        smoothed = F.conv1d(sig, kernel, padding=k_size//2).squeeze().cpu().numpy()
        p10, p95 = np.percentile(smoothed, 10), np.percentile(smoothed, 95)
        return np.clip((smoothed - p10) / (p95 - p10 + 1e-6), 0.0, 1.0)

    t_norm = refine_strict(raw_table_energies)
    p_norm = refine_strict(raw_player_energies)
    
    if len(t_norm) == 0: return

    fused_norm = np.maximum(t_norm, p_norm)
    ts_np = np.array(timestamps[:len(t_norm)])

    # 5. SEGMENTATION (0.25, 0.12, 2.0, 1.0)
    t1_res = segment_hys_logic(t_norm, ts_np, 0.25, 0.12, 2.0, 1.0)
    merged_res = segment_hys_logic(fused_norm, ts_np, 0.25, 0.12, 2.0, 1.0)

    # 6. PRINT RESULTS
    def print_res(title, data):
        print(f"\n{'='*55}\n {title}: {len(data)} FOUND\n{'='*55}")
        for i, r in enumerate(data):
            print(f" #{i+1:02d} │ {format_time(r.start)} ({r.start:6.2f}s) ➔ {format_time(r.end)} │ {r.duration:4.1f}s")

    print_res("LAYER 1: TABLE ONLY (Scaled Port)", t1_res)
    print_res("LAYER 1 + 2: MERGED (Table + Players)", merged_res)

if __name__ == "__main__":
    run_hyper_turbo_sync("Vinh_set1.mp4", "weights/yolov8x_table.pt")