# scripts/debug_pipeline.py
import json
import cv2
import sys
from pathlib import Path

# Ensure we can import from the backend folder in the root
sys.path.append(str(Path(__file__).parent.parent))

def run_debug(json_path_str: str, video_path_str: str):
    """
    Diagnostic tool to visualize why the scoreboard might be wrong.
    It checks:
    1. If the Table ROI is correctly identified.
    2. If the Rally segments are correctly timed.
    3. What frames Ollama saw to decide the winner.
    """
    print(f"--- STARTING VISUAL DEBUG REPORT ---")
    
    json_path = Path(json_path_str)
    video_path = Path(video_path_str)
    
    if not json_path.exists():
        print(f"CRITICAL ERROR: JSON file not found: {json_path}")
        return

    # Load the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"CRITICAL ERROR: Could not open video: {video_path}")
        return

    output_dir = Path("debug_report")
    output_dir.mkdir(exist_ok=True)

    # --- 1. DEBUG STEP: TABLE ROI VERIFICATION ---
    # Check if the ROI detected covers ONLY the table area
    roi = data.get('roi')
    ret, first_frame = cap.read()
    
    if ret and roi and all(k in roi for k in ['x', 'y', 'w', 'h']):
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
        
        # Save a full frame with a RED box showing the detected ROI
        roi_viz = first_frame.copy()
        cv2.rectangle(roi_viz, (x, y), (x + w, y + h), (0, 0, 255), 6)
        cv2.putText(roi_viz, f"ROI: {x},{y} {w}x{h}", (x, y - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        roi_img_path = output_dir / "01_roi_verification.jpg"
        cv2.imwrite(str(roi_img_path), roi_viz)
        
        # Save the actual cropped view that the AI processes
        # Adding a safety check to prevent empty crops
        if w > 0 and h > 0:
            crop = first_frame[y:y+h, x:x+w]
            crop_img_path = output_dir / "02_ai_crop_view.jpg"
            cv2.imwrite(str(crop_img_path), crop)
            print(f"[SUCCESS] ROI verification images saved to '{output_dir}/'")
    else:
        print("[WARNING] ROI data missing or invalid in JSON. Skipping ROI debug.")

    # --- 2. DEBUG STEP: RALLY SEGMENTS & WINNERS ---
    points = data.get('points', [])
    print(f"\nAnalyzing {len(points)} rallies found in JSON:")
    
    for p in points:
        p_id = p.get('id', 'unknown')
        t_start = p.get('t_start', 0.0)
        t_end = p.get('t_end', 0.0)
        winner = p.get('winner', 'unknown')
        
        # FIXED: Use :.2f for floats instead of :.2s
        print(f"  > {p_id}: Time {t_start:.2f}s to {t_end:.2f}s | Winner: {winner}")
        
        # Seek to the end of the rally (The "Moment of Death" for the ball)
        cap.set(cv2.CAP_PROP_POS_MSEC, t_end * 1000)
        success, frame = cap.read()
        
        if success:
            # Draw metadata on the frame for easier debugging
            overlay = frame.copy()
            text = f"ID: {p_id} | WINNER: {winner.upper()}"
            cv2.putText(overlay, text, (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
            
            fname = output_dir / f"rally_{p_id}_end_frame.jpg"
            cv2.imwrite(str(fname), overlay)
        else:
            print(f"    [!] Failed to extract frame for rally {p_id} at {t_end}s")
            
    cap.release()
    print(f"\n--- DEBUG COMPLETE ---")
    print(f"Please check the folder: {output_dir.absolute()}")

if __name__ == "__main__":
    # Update these filenames to match your current test case
    TARGET_JSON = "matches/Vinh_set1_2events_draft.json"
    TARGET_VIDEO = "Vinh_set1_2events.mp4"
    
    run_debug(TARGET_JSON, TARGET_VIDEO)