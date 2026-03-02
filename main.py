import torch
import sys
import os
from pathlib import Path
from datetime import datetime

# Import backend components
from backend.video_gpu_io import probe_video_ffprobe, nvdec_bgr24_stream
from backend.ai_table_roi_dl import detect_table_roi_dl, DLConfig
from backend.ai_rally_segmentation import detect_rally_segments_gpu
from backend.ai_contract import DraftMatch, DraftPointEvent, save_draft_match, load_draft_match, to_core_rally_events
from backend.ai_ollama_client import OllamaVisionClient
from backend.timeline import build_match_timeline

# Import rendering logic
import cv2
import subprocess

def main():
    # ==========================================================
    # --- CONFIGURATION (Upgraded to YOLOv8x for RTX 5060 Ti) ---
    VIDEO_INPUT = "Vinh_set1_2events.mp4"
    YOLO_WEIGHTS = "weights/yolov8x_table.pt" # Upgraded to Extra Large
    OLLAMA_MODEL = "llama3.2-vision"
    # ==========================================================

    print(f"====================================================")
    print(f"   AUTO PIPELINE [X-LARGE MODE]: {VIDEO_INPUT}")
    print(f"====================================================")

    # --- STRICT PRE-FLIGHT CHECKS ---
    video_path = Path(VIDEO_INPUT)
    weights_path = Path(YOLO_WEIGHTS)

    if not video_path.exists():
        print(f"CRITICAL ERROR: Video NOT FOUND: {VIDEO_INPUT}")
        sys.exit(1)

    if not weights_path.exists():
        print(f"CRITICAL ERROR: YOLOv8x weights NOT FOUND at: {YOLO_WEIGHTS}")
        print("ACTION: Run the download command to get the Extra Large model.")
        sys.exit(1)
    else:
        print(f"SUCCESS: YOLOv8x (Extra Large) verified.")

    if not torch.cuda.is_available():
        print("CRITICAL ERROR: CUDA GPU NOT ACCESSIBLE.")
        sys.exit(1)
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"SUCCESS: Utilizing High-End GPU: {gpu_name}")

    # --- INITIALIZATION ---
    DEVICE = "cuda"
    STRIDE = 2 # 5060 Ti can handle STRIDE=1, but 2 is optimal for 60fps
    output_name = video_path.stem
    draft_json_path = Path(f"matches/{output_name}_draft.json")
    refined_json_path = Path(f"matches/{output_name}_refined.json")
    final_video_path = f"{output_name}_final_1080p.mp4"
    temp_no_audio = "temp_render.mp4"

    start_time = datetime.now()

    # --- STEP 1: VIDEO PROBE & ROI (High Precision) ---
    info = probe_video_ffprobe(VIDEO_INPUT)
    # Using v8x for maximum ROI precision
    roi = detect_table_roi_dl(VIDEO_INPUT, cfg=DLConfig(weights_path=str(weights_path), device=DEVICE))
    print(f"[1/4] Precision ROI Detected: {roi.as_tuple()} using {gpu_name}")

    # --- STEP 2: RALLY SEGMENTATION (GPU Acceleration) ---
    print(f"[2/4] Analyzing motion on GPU (Blackwell Optimized)...")
    energies, timestamps = [], []
    prev_frame_gpu = None
    frame_gen = nvdec_bgr24_stream(VIDEO_INPUT, info.width, info.height, crop_roi=roi.as_tuple())

    for idx, frame_np in enumerate(frame_gen):
        if idx % STRIDE != 0: continue
        
        current_frame_gpu = torch.from_numpy(frame_np).to(DEVICE).float()
        if prev_frame_gpu is not None:
            # Batch-like calculation on Tensor cores
            energies.append(torch.abs(current_frame_gpu - prev_frame_gpu).mean().item())
            timestamps.append(idx / info.fps)
        prev_frame_gpu = current_frame_gpu
        if idx % 500 == 0: 
            print(f"    > Step 2 Progress: Frame {idx} processing...", end="\r")

    segments = detect_rally_segments_gpu(energies, timestamps, effective_fps=info.fps/STRIDE)
    
    draft_events = [DraftPointEvent(id=f"r_{i+1:03d}", t_start=s.t_start, t_end=s.t_end) for i, s in enumerate(segments)]
    draft_match = DraftMatch(video_path=str(video_path.absolute()), video_fps=info.fps, points=draft_events)
    save_draft_match(draft_json_path, draft_match)
    print(f"\nFound {len(segments)} rallies. Data saved to {draft_json_path}")

    # --- STEP 3: OLLAMA WINNER PREDICTION ---
    print(f"[3/4] Running Ollama Vision ({OLLAMA_MODEL}) on 16GB VRAM...")
    ollama_client = OllamaVisionClient(model_name=OLLAMA_MODEL)
    for p in draft_match.points:
        img_path = extract_last_frame(VIDEO_INPUT, p.t_end)
        if img_path:
            p.winner = ollama_client.predict_winner(img_path)
            print(f"    > Rally {p.id}: {p.winner}")
    
    save_draft_match(refined_json_path, draft_match)

    # --- STEP 4: RENDER FINAL 1080P VIDEO ---
    print(f"[4/4] Rendering final video (1080p Downscaling)...")
    core_events = to_core_rally_events(draft_match)
    timeline = build_match_timeline(best_of=draft_match.best_of, events=core_events)
    
    render_scoreboard_1080p(VIDEO_INPUT, temp_no_audio, timeline)
    
    print(f"\n[FINAL] Merging high-quality audio...")
    mux_audio(temp_no_audio, VIDEO_INPUT, final_video_path)
    
    if os.path.exists(temp_no_audio): os.remove(temp_no_audio)
    if os.path.exists("temp_last_frame.jpg"): os.remove("temp_last_frame.jpg")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"====================================================")
    print(f"SUCCESS! Output: {final_video_path}")
    print(f"Processing Time: {elapsed:.2f}s")
    print(f"====================================================")

# --- HELPER FUNCTIONS ---

def extract_last_frame(video_path, t_end):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t_end * 1000)
    ret, frame = cap.read()
    cap.release()
    if ret:
        path = "temp_last_frame.jpg"
        cv2.imwrite(path, frame)
        return path
    return None

def render_scoreboard_1080p(input_v, output_v, timeline):
    from render.renderer import ScoreboardRenderer
    renderer = ScoreboardRenderer(input_path=input_v, output_path=output_v, timeline=timeline)
    
    cap = cv2.VideoCapture(input_v)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_w, target_h = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_v, fourcc, fps, (target_w, target_h))

    frame_count, state_idx = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        cur_t = frame_count / fps
        
        while (state_idx + 1 < len(timeline) and cur_t >= timeline[state_idx + 1].timestamp):
            state_idx += 1
        
        renderer._draw_scoreboard(frame, timeline[state_idx], target_w, target_h)
        out.write(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            pct = (frame_count / total_frames) * 100
            print(f"    > Rendering: Frame {frame_count}/{total_frames} ({pct:.1f}%)", end="\r")

    cap.release()
    out.release()

def mux_audio(v_no_a, a_src, final_out):
    cmd = ['ffmpeg', '-y', '-i', v_no_a, '-i', a_src, '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-shortest', final_out]
    subprocess.run(cmd, check=True, capture_output=True)

if __name__ == "__main__":
    main()