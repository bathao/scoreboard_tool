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
    to_core_rally_events, load_draft_match
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
    print(f"   AUTO PIPELINE [STRICT MODE]: {VIDEO_INPUT}")
    print(f"====================================================")

    # --- PHASE 0: PRE-FLIGHT CHECKS ---
    video_path = Path(VIDEO_INPUT)
    weights_path = Path(YOLO_WEIGHTS)

    if not video_path.exists():
        raise FileNotFoundError(f"CRITICAL: Video not found: {VIDEO_INPUT}")
    if not weights_path.exists():
        raise FileNotFoundError(f"CRITICAL: YOLO weights not found: {YOLO_WEIGHTS}")
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL: CUDA GPU not found. High-performance pipeline requires GPU.")

    DEVICE = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Verified: GPU={gpu_name}, Weights={YOLO_WEIGHTS}")

    start_time = datetime.now()

    # --- STEP 1: ROI DETECTION (CHECKPOINT) ---
    print(f"\n[1/4] Detecting Table ROI...")
    info = probe_video_ffprobe(VIDEO_INPUT)
    roi = detect_table_roi_dl(VIDEO_INPUT, cfg=DLConfig(weights_path=str(weights_path), device=DEVICE))
    
    # STRICT CHECK: ROI
    if roi is None or roi.w <= 0 or roi.h <= 0:
        raise RuntimeError("CRITICAL: ROI detection failed or returned empty area. STOPPING.")
    print(f"      ROI SUCCESS: {roi.as_tuple()}")

    # --- STEP 2: RALLY SEGMENTATION (CHECKPOINT) ---
    print(f"\n[2/4] Analyzing motion in ROI on GPU...")
    energies, timestamps = [], []
    prev_frame_gpu = None
    STRIDE = 2
    
    frame_gen = nvdec_bgr24_stream(VIDEO_INPUT, info.width, info.height, crop_roi=roi.as_tuple())
    for idx, frame_np in enumerate(frame_gen):
        if idx % STRIDE != 0: continue
        curr_gpu = torch.from_numpy(frame_np).to(DEVICE).float()
        if prev_frame_gpu is not None:
            energies.append(torch.abs(curr_gpu - prev_frame_gpu).mean().item())
            timestamps.append(idx / info.fps)
        prev_frame_gpu = curr_gpu
        if idx % 500 == 0:
            print(f"    > Step 2 Progress: Frame {idx} analyzed...", end="\r")

    if not energies:
        raise RuntimeError("CRITICAL: No motion data processed. Check video stream/NVDEC.")

    segments = detect_rally_segments_gpu(energies, timestamps, effective_fps=info.fps/STRIDE)
    
    # CHECKPOINT: Segments
    if not segments:
        print("WARNING: No rallies detected. Adjust motion threshold in code.")
        sys.exit("Pipeline stopped: Zero rallies found.")
    
    # Save Initial JSON
    draft_match = DraftMatch(
        video_path=str(video_path.absolute()),
        video_fps=info.fps,
        roi=roi.to_dict(),
        points=[DraftPointEvent(id=f"r_{i+1:03d}", t_start=s.t_start, t_end=s.t_end) for i, s in enumerate(segments)]
    )
    draft_json = Path(f"matches/{video_path.stem}_draft.json")
    save_draft_match(draft_json, draft_match)
    print(f"\n      SEGMENTATION SUCCESS: {len(segments)} rallies found.")

    # --- STEP 3: OLLAMA WINNER PREDICTION ---
    print(f"\n[3/4] Running Ollama Vision ({OLLAMA_MODEL}) to predict winners...")
    ollama_client = OllamaVisionClient(model_name=OLLAMA_MODEL)
    for p in draft_match.points:
        img_path = extract_last_frame(VIDEO_INPUT, p.t_end)
        if img_path:
            p.winner = ollama_client.predict_winner(img_path)
            print(f"    > Rally {p.id}: AI Winner -> {p.winner}")
    
    refined_json = Path(f"matches/{video_path.stem}_refined.json")
    save_draft_match(refined_json, draft_match)

    # --- STEP 4: FINAL RENDERING ---
    print(f"\n[4/4] Rendering final 1080p video with scoreboard...")
    final_video = f"{video_path.stem}_final_1080p.mp4"
    temp_no_audio = "temp_render.mp4"
    
    core_events = to_core_rally_events(draft_match)
    timeline = build_match_timeline(best_of=draft_match.best_of, events=core_events)
    
    # Using the rendering helper with progress bar
    render_and_resize_1080p(VIDEO_INPUT, temp_no_audio, timeline)
    
    # Mux Audio
    print(f"\n[FINAL] Finalizing audio and muxing...")
    mux_audio(temp_no_audio, VIDEO_INPUT, final_video)

    # Cleanup
    if os.path.exists(temp_no_audio): os.remove(temp_no_audio)
    if os.path.exists("temp_last_frame.jpg"): os.remove("temp_last_frame.jpg")

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"====================================================")
    print(f"SUCCESS! Output Video: {final_video}")
    print(f"Total Workflow Time: {total_time:.2f}s")
    print(f"====================================================")

# --- UTILITIES ---

def extract_last_frame(video_path, t_end):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t_end * 1000)
    ret, frame = cap.read()
    cap.release()
    return "temp_last_frame.jpg" if (ret and cv2.imwrite("temp_last_frame.jpg", frame)) else None

def render_and_resize_1080p(input_v, output_v, timeline):
    from render.renderer import ScoreboardRenderer
    renderer = ScoreboardRenderer(input_path=input_v, output_path=output_v, timeline=timeline)
    cap = cv2.VideoCapture(input_v)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_w, target_h = 1920, 1080
    out = cv2.VideoWriter(output_v, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w, target_h))

    f_count, s_idx = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        cur_t = f_count / fps
        while (s_idx + 1 < len(timeline) and cur_t >= timeline[s_idx + 1].timestamp):
            s_idx += 1
        renderer._draw_scoreboard(frame, timeline[s_idx], target_w, target_h)
        out.write(frame)
        f_count += 1
        if f_count % 100 == 0:
            print(f"    > Step 4 Rendering: {f_count}/{total_f} frames...", end="\r")
    cap.release(); out.release()

def mux_audio(v_no_a, a_src, final_out):
    cmd = ['ffmpeg', '-y', '-i', v_no_a, '-i', a_src, '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-shortest', final_out]
    subprocess.run(cmd, check=True, capture_output=True)

if __name__ == "__main__":
    main()