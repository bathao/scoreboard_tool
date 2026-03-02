# scripts/debug_pipeline.py
import json
import cv2
import sys
from pathlib import Path

# Ensure we can import from the backend folder in the root
sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_table_roi import TableROI

def run_debug(json_path_str: str, video_path_str: str):
    """
    Diagnostic tool to visualize:
    1. Table ROI vs Unified Play Zone (UPZ)
    2. Rally segments and their winners
    """
    print(f"--- STARTING VISUAL DEBUG REPORT (ROI + UPZ) ---")
    
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

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = Path("debug_report")
    output_dir.mkdir(exist_ok=True)

    # --- 1. DEBUG STEP: ROI & UPZ VERIFICATION ---
    roi_data = data.get('roi')
    ret, first_frame = cap.read()
    
    if ret and roi_data:
        # Convert dict to TableROI object to use its logic
        table_roi = TableROI.from_dict(roi_data)
        tx, ty, tw, th = table_roi.as_tuple()
        
        # Calculate UPZ based on the same logic used in main.py
        upz_x, upz_y, upz_w, upz_h = table_roi.get_unified_play_zone(W, H)
        
        # Visualization frame
        viz_frame = first_frame.copy()
        
        # Draw Unified Play Zone (RED) - The area used for motion analysis
        cv2.rectangle(viz_frame, (upz_x, upz_y), (upz_x + upz_w, upz_y + upz_h), (0, 0, 255), 8)
        cv2.putText(viz_frame, "UNIFIED PLAY ZONE (Motion Search)", (upz_x, upz_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        
        # Draw Table ROI (BLUE) - The precise table surface
        cv2.rectangle(viz_frame, (tx, ty), (tx + tw, ty + th), (255, 0, 0), 6)
        cv2.putText(viz_frame, "TABLE SURFACE", (tx + 10, ty + th - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        
        roi_img_path = output_dir / "01_roi_vs_upz_verification.jpg"
        cv2.imwrite(str(roi_img_path), viz_frame)
        
        # Save the actual crop that the AI "sees" during segmentation
        upz_crop = first_frame[upz_y:upz_y+upz_h, upz_x:upz_x+upz_w]
        cv2.imwrite(str(output_dir / "02_upz_ai_view.jpg"), upz_crop)
        
        print(f"[SUCCESS] ROI & UPZ images saved to '{output_dir}/'")
    else:
        print("[WARNING] ROI data missing. Skipping ROI debug.")

    # --- 2. DEBUG STEP: RALLY SEGMENTS ---
    points = data.get('points', [])
    print(f"\nVerifying {len(points)} rallies:")
    
    for p in points:
        p_id = p.get('id', 'unknown')
        t_end = p.get('t_end', 0.0)
        winner = p.get('winner', 'unknown')
        
        print(f"  > {p_id} ends at {t_end:.2f}s | Winner: {winner}")
        
        # Seek to the end of the rally
        cap.set(cv2.CAP_PROP_POS_MSEC, t_end * 1000)
        success, frame = cap.read()
        
        if success:
            cv2.putText(frame, f"ID: {p_id} | WINNER: {winner.upper()}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 5)
            fname = output_dir / f"rally_{p_id}_moment_of_death.jpg"
            cv2.imwrite(str(fname), frame)
            
    cap.release()
    print(f"\n--- DEBUG COMPLETE ---")
    print(f"Check results in: {output_dir.absolute()}")

if __name__ == "__main__":
    # Point to your latest draft JSON and source video
    TARGET_JSON = "matches/Vinh_set1_draft.json"
    TARGET_VIDEO = "Vinh_set1.mp4"
    
    run_debug(TARGET_JSON, TARGET_VIDEO)