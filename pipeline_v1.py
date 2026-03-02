import sys
from pathlib import Path
from datetime import datetime

# Import components from your backend
from backend.ai_table_roi_dl import detect_table_roi_dl, DLConfig
from backend.ai_rally_segmentation import detect_rally_segments_motion
from backend.ai_contract import DraftMatch, DraftPointEvent, save_draft_match

def run_video_pipeline(video_path: str, weights_path: str = None):
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"Error: Video file not found at {video_path}")
        return None

    print(f"--- Processing Video: {video_file.name} ---")

    # Step 1: Detect Table ROI
    print("[1/3] Detecting table ROI...")
    # Falls back to classical method automatically if weights or ultralytics missing
    roi = detect_table_roi_dl(
        video_path, 
        cfg=DLConfig(weights_path=weights_path) if weights_path and Path(weights_path).exists() else None,
        debug=False
    )
    print(f"ROI Detected: {roi.as_tuple()} | Method: {roi.method} | Conf: {roi.confidence:.2f}")

    # Step 2: Segment Rallies via Motion
    print("[2/3] Analyzing rally segments via motion...")
    # Focus motion detection only on the detected table ROI
    segments, video_fps = detect_rally_segments_motion(
        video_path, 
        roi=roi.as_tuple(),
        active_threshold=0.22, 
        merge_gap_sec=0.7
    )
    print(f"Found {len(segments)} potential rallies.")

    # Step 3: Package into AI Contract JSON (DraftMatch)
    print("[3/3] Generating Draft JSON...")
    draft_points = []
    for i, seg in enumerate(segments):
        draft_points.append(DraftPointEvent(
            id=f"rally_{i+1:03d}",
            t_start=round(seg.t_start, 2),
            t_end=round(seg.t_end, 2),
            winner="unknown",  # To be edited by human or refined by future AI
            confidence=seg.confidence,
            flags=seg.flags
        ))

    draft_match = DraftMatch(
        video_path=str(video_file.absolute()),
        video_fps=video_fps,
        points=draft_points,
        created_at=datetime.now().isoformat(),
        best_of=5
    )

    # Save output to matches folder
    output_path = Path("matches") / f"{video_file.stem}_draft.json"
    save_draft_match(output_path, draft_match)
    
    print(f"--- Pipeline Completed ---")
    print(f"Draft JSON saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Test run
    run_video_pipeline("Vinh_1280_2min.mp4")