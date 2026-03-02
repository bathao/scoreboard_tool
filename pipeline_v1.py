# scripts/debug_pipeline.py
import json
import cv2
import sys
from pathlib import Path

# Fix import path for backend
sys.path.append(str(Path(__file__).parent.parent))

def run_debug(json_path_str, video_path_str):
    print(f"--- STARTING VISUAL DEBUG REPORT ---")
    
    json_path = Path(json_path_str)
    video_path = Path(video_path_str)
    
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cap = cv2.VideoCapture(str(video_path))
    output_dir = Path("debug_report")
    output_dir.mkdir(exist_ok=True)

    # 1. DEBUG STEP: TABLE ROI
    # Check if the ROI detected covers ONLY the table
    roi = data.get('roi')
    ret, first_frame = cap.read()
    if ret and roi:
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
        
        # Save a frame with a RED box showing the ROI
        roi_viz = first_frame.copy()
        cv2.rectangle(roi_viz, (x, y), (x + w, y + h), (0, 0, 255), 4)
        cv2.putText(roi_viz, "DETECTED TABLE AREA", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(output_dir / "01_roi_verification.jpg"), roi_viz)
        
        # Save the actual crop the AI is looking at
        crop = first_frame[y:y+h, x:x+w]
        cv2.imwrite(str(output_dir / "02_ai_crop_view.jpg"), crop)
        print("[DEBUG] ROI verification images saved to 01_... and 02_...")

    # 2. DEBUG STEP: RALLY WINNERS
    # Save the frame used by Ollama to decide the winner
    points = data.get('points', [])
    print(f"\nChecking {len(points)} rallies:")
    for p in points:
        p_id = p['id']
        t_end = p['t_end']
        winner = p.get('winner', 'unknown')
        
        print(f"  > Rally {p_id} ends at {t_end:.2f}s | Predicted Winner: {winner}")
        
        cap.set(cv2.CAP_PROP_POS_MSEC, t_end * 1000)
        ret, frame = cap.read()
        if ret:
            # Draw winner info on frame
            cv2.putText(frame, f"WINNER: {winner.upper()}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            fname = output_dir / f"rally_{p_id}_moment_of_death.jpg"
            cv2.imwrite(str(fname), frame)
            
    cap.release()
    print(f"\n--- DEBUG COMPLETE ---")
    print(f"Check folder: {output_dir.absolute()}")

if __name__ == "__main__":
    # Update these paths to your test files
    run_debug("matches/Vinh_set1_2events_draft.json", "Vinh_set1_2events.mp4")