import sys
import cv2
import os
import subprocess
from pathlib import Path

# Add root directory to sys.path for backend imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_contract import load_draft_match, to_core_rally_events
from backend.timeline import build_match_timeline
from render.renderer import ScoreboardRenderer

def main():
    # --- CONFIGURATION ---
    REFINED_JSON = "matches/Vinh_set1_refined.json"
    DRAFT_JSON = "matches/Vinh_set1_draft.json"
    
    input_json = REFINED_JSON if Path(REFINED_JSON).exists() else DRAFT_JSON
    input_video = "Vinh_set1.mp4"
    temp_video = "temp_no_audio.mp4"
    final_output = "Vinh_set1_1080p_final.mp4"

    if not Path(input_json).exists():
        print(f"ERROR: No JSON found. Run main.py or ai_refine_draft.py first.")
        return

    print(f"--- STARTING FINAL RENDER WITH AUDIO ---")
    
    # 1. Load Data & Logic
    draft = load_draft_match(Path(input_json))
    for p in draft.points:
        if p.winner == "unknown":
            p.winner = "player_a" 

    core_events = to_core_rally_events(draft)
    timeline = build_match_timeline(best_of=draft.best_of, events=core_events)

    # 2. Render Video Frames (No Audio yet)
    print(f"Step 1: Rendering video frames to 1080p...")
    renderer = ScoreboardRenderer(
        input_path=input_video,
        output_path=temp_video,
        timeline=timeline
    )
    render_to_1080p(renderer)

    # 3. Mux Audio using FFmpeg
    print(f"Step 2: Merging original audio from {input_video}...")
    try:
        merge_audio(temp_video, input_video, final_output)
        print(f"--- SUCCESS: Final video with audio saved as {final_output} ---")
    except Exception as e:
        print(f"ERROR merging audio: {e}")
    finally:
        # Cleanup temporary video file
        if os.path.exists(temp_video):
            os.remove(temp_video)

def render_to_1080p(renderer: ScoreboardRenderer):
    cap = cv2.VideoCapture(renderer.input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_w, target_h = 1920, 1080
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(renderer.output_path, fourcc, fps, (target_w, target_h))

    frame_count = 0
    state_index = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        current_time = frame_count / fps

        while (state_index + 1 < len(renderer.timeline) and 
               current_time >= renderer.timeline[state_index + 1].timestamp):
            state_index += 1

        renderer._draw_scoreboard(frame, renderer.timeline[state_index], target_w, target_h)
        out.write(frame)
        frame_count += 1
        if frame_count % 500 == 0:
            print(f"   > Processed {frame_count} frames...", end="\r")

    cap.release()
    out.release()
    print("\nVideo frame rendering complete.")

def merge_audio(video_no_audio, audio_source, output_file):
    """
    Uses FFmpeg to combine the rendered video with the original audio.
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', video_no_audio,   # Input 0: Rendered video (no audio)
        '-i', audio_source,     # Input 1: Original video (has audio)
        '-map', '0:v:0',        # Take video from input 0
        '-map', '1:a:0',        # Take audio from input 1
        '-c:v', 'copy',         # Copy video stream (no re-encoding, fast)
        '-c:a', 'aac',          # Encode audio to AAC
        '-shortest',            # Finish when the shortest stream ends
        output_file
    ]
    subprocess.run(cmd, check=True, capture_output=True)

if __name__ == "__main__":
    main()