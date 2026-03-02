# main.py
import torch
import sys
import os
from pathlib import Path
from datetime import datetime

# Core Backend Imports
from backend.video_gpu_io import probe_video_ffprobe, nvdec_bgr24_stream
from backend.ai_table_roi_dl import detect_table_roi_dl, DLConfig
from backend.ai_rally_segmentation import detect_rally_segments_gpu
from backend.ai_contract import (
    DraftMatch, DraftPointEvent, save_draft_match, 
    to_core_rally_events
)
from backend.ai_ollama_client import OllamaVisionClient
from backend.timeline import build_match_timeline

# UI & Rendering Imports
import cv2
import subprocess

def main():
    # --- CONFIGURATION ---
    VIDEO_INPUT = "Vinh_set1.mp4"
    YOLO_WEIGHTS = "weights/yolov8x_table.pt"
    OLLAMA_MODEL = "llama3.2-vision"

    print(f"====================================================")
    print(f"   AUTO PIPELINE [UNIFIED PLAY ZONE MODE]: {VIDEO_INPUT}")
    print(f"====================================================")

    # --- PHASE 0: PRE-FLIGHT CHECKS ---
    video_path = Path(VIDEO_INPUT)
    weights_path = Path(YOLO_WEIGHTS)

    if not video_path.exists() or not weights_path.exists():
        raise FileNotFoundError("CRITICAL: Video or Weights missing.")
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL: CUDA GPU not found.")

    DEVICE = "cuda"
    print(f"Verified Hardware: {torch.cuda.get_device_name(0)}")

    start_time = datetime.now()

    # --- STEP 1: ROI DETECTION ---
    print(f"\n[1/4] Detecting Table ROI...")
    info = probe_video_ffprobe(VIDEO_INPUT)
    table_roi = detect_table_roi_dl(VIDEO_INPUT, cfg=DLConfig(weights_path=str(weights_path), device=DEVICE))
    
    if table_roi.w <= 0 or table_roi.h <= 0:
        raise RuntimeError("CRITICAL: Table ROI detection failed.")
    
    # NEW: Calculate Unified Play Zone for stable motion analysis
    upz_roi = table_roi.get_unified_play_zone(info.width, info.height)
    print(f"      TABLE ROI: {table_roi.as_tuple()}")
    print(f"      UNIFIED PLAY ZONE: {upz_roi}")

    # --- STEP 2: RALLY SEGMENTATION (UPZ MODE) ---
    print(f"\n[2/4] Analyzing motion in UPZ on GPU...")
    energies, timestamps = [], []
    prev_frame_gpu = None
    STRIDE = 2 # Process every 2nd frame for speed
    
    # Use the expanded UPZ ROI here
    frame_gen = nvdec_bgr24_stream(VIDEO_INPUT, info.width, info.height, crop_roi=upz_roi)
    
    for idx, frame_np in enumerate(frame_gen):
        if idx % STRIDE != 0: continue
        
        # Transfer to GPU and calculate mean difference
        curr_gpu = torch.from_numpy(frame_np).to(DEVICE).float()
        if prev_frame_gpu is not None:
            # Captures both ball movement and player swings
            diff = torch.abs(curr_gpu - prev_frame_gpu).mean().item()
            energies.append(diff)
            timestamps.append(idx / info.fps)
            
        prev_frame_gpu = curr_gpu
        if idx % 500 == 0:
            print(f"    > Progress: Frame {idx} processed...", end="\r")

    # Algorithm to group energy spikes into rally segments
    segments = detect_rally_segments_gpu(
        energies, 
        timestamps, 
        effective_fps=info.fps/STRIDE,
        active_threshold=0.35 # Higher threshold due to human presence in UPZ
    )
    
    if not segments:
        sys.exit("Pipeline stopped: Zero rallies found. Check motion thresholds.")
    
    # Initialize Draft JSON with detected segments
    draft_match = DraftMatch(
        video_path=str(video_path.absolute()),
        video_fps=info.fps,
        roi=table_roi.to_dict(), # Keep the precise table ROI for rendering
        points=[DraftPointEvent(id=f"r_{i+1:03d}", t_start=s.t_start, t_end=s.t_end) for i, s in enumerate(segments)]
    )
    save_draft_match(Path(f"matches/{video_path.stem}_draft.json"), draft_match)
    print(f"\n      SUCCESS: {len(segments)} rallies identified.")

    # --- STEP 3: WINNER PREDICTION (OLLAMA) ---
    print(f"\n[3/4] Running Ollama Vision ({OLLAMA_MODEL}) to predict winners...")
    ollama_client = OllamaVisionClient(model_name=OLLAMA_MODEL)
    for p in draft_match.points:
        # Extract the 'moment of death' frame
        img_path = extract_last_frame(VIDEO_INPUT, p.t_end)
        if img_path:
            p.winner = ollama_client.predict_winner(img_path)
            print(f"    > {p.id}: Winner -> {p.winner}")
    
    save_draft_match(Path(f"matches/{video_path.stem}_refined.json"), draft_match)

    # --- STEP 4: SCOREBOARD RENDERING ---
    print(f"\n[4/4] Rendering 1080p video with Scoreboard...")
    final_video = f"{video_path.stem}_final_1080p.mp4"
    temp_render = "temp_render.mp4"
    
    core_events = to_core_rally_events(draft_match)
    timeline = build_match_timeline(best_of=draft_match.best_of, events=core_events)
    
    # Custom render function to handle 1080p resize and scoreboard overlay
    render_and_resize_1080p(VIDEO_INPUT, temp_render, timeline)
    
    # Final step: Merge original audio with rendered video
    print(f"\n[FINAL] Muxing audio...")
    mux_audio(temp_render, VIDEO_INPUT, final_video)

    # Cleanup temp files
    for tmp in [temp_render, "temp_last_frame.jpg"]:
        if os.path.exists(tmp): os.remove(tmp)

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"====================================================")
    print(f"DONE! Final Video: {final_video}")
    print(f"Total time: {total_time:.2f}s")
    print(f"====================================================")

# --- HELPER FUNCTIONS ---

def extract_last_frame(video_path, t_end):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t_end * 1000)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite("temp_last_frame.jpg", frame)
        return "temp_last_frame.jpg"
    return None

def render_and_resize_1080p(input_v, output_v, timeline):
    from render.renderer import ScoreboardRenderer
    renderer = ScoreboardRenderer(input_path=input_v, output_path=output_v, timeline=timeline)
    cap = cv2.VideoCapture(input_v)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_w, target_h = 1920, 1080
    out = cv2.VideoWriter(output_v, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w, target_h))

    f_count, s_idx = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Resize to standard 1080p
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        cur_t = f_count / fps
        
        # Sync timeline state
        while (s_idx + 1 < len(timeline) and cur_t >= timeline[s_idx + 1].timestamp):
            s_idx += 1
            
        renderer._draw_scoreboard(frame, timeline[s_idx], target_w, target_h)
        out.write(frame)
        f_count += 1
        if f_count % 100 == 0:
            print(f"    > Rendering: {f_count}/{total_f} frames...", end="\r")
    cap.release(); out.release()

def mux_audio(v_no_a, a_src, final_out):
    cmd = ['ffmpeg', '-y', '-i', v_no_a, '-i', a_src, '-map', '0:v:0', '-map', '1:a:0', 
           '-c:v', 'copy', '-c:a', 'aac', '-shortest', final_out]
    subprocess.run(cmd, check=True, capture_output=True)

if __name__ == "__main__":
    main()