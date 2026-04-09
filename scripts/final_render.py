import sys
import os
import argparse
from pathlib import Path

# Add root directory to sys.path for backend imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.rally_timeline_contract import load_rally_timeline
from backend.rendering import render_scoreboard_video

def main():
    parser = argparse.ArgumentParser(description="Render final 1080p scoreboard video and merge original audio.")
    parser.add_argument("--video", required=True, help="Path to input source video")
    parser.add_argument("--json", required=True, help="Path to rally timeline / refined JSON")
    parser.add_argument("--out", required=True, help="Path to final output video")
    parser.add_argument("--temp-video", default="temp_no_audio.mp4", help="Temporary intermediate video path")
    parser.add_argument("--player-a-name", default="PLAYER A", help="Display name for near-side player")
    parser.add_argument("--player-b-name", default="PLAYER B", help="Display name for far-side player")
    parser.add_argument(
        "--unknown-winner-policy",
        choices=["player_a", "player_b", "skip"],
        default="player_a",
        help="How to handle unresolved winner before building timeline",
    )
    args = parser.parse_args()

    input_json = args.json
    input_video = args.video
    temp_video = args.temp_video
    final_output = args.out
    Path(final_output).parent.mkdir(parents=True, exist_ok=True)

    if not Path(input_json).exists():
        print(f"ERROR: JSON not found: {input_json}")
        return

    print(f"--- STARTING FINAL RENDER WITH AUDIO ---")
    
    # 1. Load Data & Logic
    timeline = load_rally_timeline(Path(input_json))
    for p in timeline.points:
        if p.winner == "unknown":
            if args.unknown_winner_policy == "skip":
                continue
            p.winner = args.unknown_winner_policy

    # 2. Render Video Frames + Audio
    print(f"Step 1: Rendering video frames to 1080p and merging original audio...")
    try:
        render_scoreboard_video(
            input_video_path=input_video,
            timeline=timeline,
            output_video_path=final_output,
            player_a_name=args.player_a_name,
            player_b_name=args.player_b_name,
            temp_video_path=temp_video,
        )
        print(f"--- SUCCESS: Final video with audio saved as {final_output} ---")
    except Exception as e:
        print(f"ERROR rendering final output: {e}")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
