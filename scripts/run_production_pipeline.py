from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str], name: str) -> None:
    print(f"\n[STEP] {name}")
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run end-to-end production pipeline: rally timeline -> native-video winner refine -> render."
    )
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--weights", default="weights/yolov8x_table.pt", help="YOLO table weights")
    parser.add_argument("--timeline-out", required=True, help="Path to output rally timeline JSON")
    parser.add_argument("--refined-out", default=None, help="Path to output refined JSON")
    parser.add_argument("--final-out", required=True, help="Path to final rendered video")
    parser.add_argument("--winner-clip-dir", default=None, help="Directory for native-video winner review clips")
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument(
        "--model-dir",
        default="models/Qwen3-VL-4B-Instruct",
        help="Local Transformers model directory for native-video winner refinement",
    )
    parser.add_argument("--skip-refine", action="store_true", help="Skip AI refinement step")
    parser.add_argument(
        "--unknown-winner-policy",
        choices=["player_a", "player_b", "skip"],
        default="player_a",
    )
    args = parser.parse_args()

    timeline_path = Path(args.timeline_out)
    refined_path = Path(args.refined_out) if args.refined_out else Path(str(timeline_path).replace("_timeline.json", "_refined.json"))
    winner_clip_dir = (
        Path(args.winner_clip_dir)
        if args.winner_clip_dir
        else Path("debug_report") / f"{timeline_path.stem}_native_video_winner_clips"
    )

    run_step(
        [
            sys.executable,
            "scripts/generate_table_rally_timeline.py",
            "--video",
            args.video,
            "--weights",
            args.weights,
            "--out",
            str(timeline_path),
            "--best-of",
            str(args.best_of),
            "--stride",
            str(args.stride),
        ],
        "Generate Rally Timeline",
    )

    input_json_for_render = timeline_path
    if not args.skip_refine:
        refine_cmd = [
            sys.executable,
            "scripts/refine_rally_winners_native_video.py",
            "--timeline",
            str(timeline_path),
            "--out",
            str(refined_path),
            "--model-dir",
            args.model_dir,
            "--clip-dir",
            str(winner_clip_dir),
        ]

        run_step(
            refine_cmd,
            "Refine Winners (Native Video)",
        )
        input_json_for_render = refined_path

    run_step(
        [
            sys.executable,
            "scripts/final_render.py",
            "--video",
            args.video,
            "--json",
            str(input_json_for_render),
            "--out",
            args.final_out,
            "--unknown-winner-policy",
            args.unknown_winner_policy,
        ],
        "Render Final Video",
    )

    print("\n[OK] Production pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
