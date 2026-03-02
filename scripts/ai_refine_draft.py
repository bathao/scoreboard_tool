import sys
import cv2
import numpy as np
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_contract import load_draft_match, save_draft_match
from backend.ai_ollama_client import OllamaVisionClient

def extract_key_frames_grid(video_path, t_start, t_end):
    """
    Extract 3 frames and stitch them into one grid for AI to see context.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Timestamps: Middle, Near End, and End
    times = [t_start + (t_end - t_start)*0.5, t_end - 0.5, t_end]
    frames = []
    
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            # Resize to save VRAM/Bandwidth
            frame = cv2.resize(frame, (640, 360))
            frames.append(frame)
    
    cap.release()
    
    if len(frames) < 3: return None
    
    # Stack horizontally (Grid)
    grid = np.hstack(frames)
    temp_path = "matches/temp_rally_grid.jpg"
    cv2.imwrite(temp_path, grid)
    return temp_path

def main():
    DRAFT_JSON = "matches/Vinh_set1_draft.json"
    MODEL_NAME = "llama3.2-vision" # or "pixtral"
    
    print(f"--- AI Refinement with Ollama ({MODEL_NAME}) ---")
    draft = load_draft_match(Path(DRAFT_JSON))
    client = OllamaVisionClient(model_name=MODEL_NAME)
    
    processed_count = 0
    for p in draft.points:
        if p.winner == "unknown":
            print(f"Analyzing {p.id} ({p.t_start}s - {p.t_end}s)...")
            
            # 1. Prepare visual data
            grid_path = extract_key_frames_grid(draft.video_path, p.t_start, p.t_end)
            
            if grid_path:
                # 2. Ask AI
                prediction = client.predict_winner(grid_path)
                p.winner = prediction
                p.confidence = 0.8 # AI set confidence
                p.source = "ai"
                print(f"   > AI Result: {prediction}")
                processed_count += 1

    # 3. Save refined result
    output_path = Path(DRAFT_JSON.replace("_draft.json", "_refined.json"))
    save_draft_match(output_path, draft)
    print(f"\n--- DONE ---")
    print(f"Processed {processed_count} rallies. Refined JSON: {output_path}")

if __name__ == "__main__":
    main()