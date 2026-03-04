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
        description="Run end-to-end production pipeline: draft -> refine -> render."
    )
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--weights", default="weights/yolov8x_table.pt", help="YOLO table weights")
    parser.add_argument("--draft-out", required=True, help="Path to output draft JSON")
    parser.add_argument("--refined-out", default=None, help="Path to output refined JSON")
    parser.add_argument("--final-out", required=True, help="Path to final rendered video")
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--model", default="llama3.2-vision", help="Ollama model for winner refinement")
    parser.add_argument("--votes", type=int, default=3, help="Number of AI votes per rally for majority decision")
    parser.add_argument(
        "--expected-scope",
        choices=["any", "set", "match"],
        default="any",
        help="Expected clip scope for score-rule validation",
    )
    parser.add_argument(
        "--expected-final-set-score",
        default=None,
        help="Expected final set score in A-B format, e.g. 11-3",
    )
    parser.add_argument("--skip-refine", action="store_true", help="Skip AI refinement step")
    parser.add_argument(
        "--unknown-winner-policy",
        choices=["player_a", "player_b", "skip"],
        default="player_a",
    )
    args = parser.parse_args()

    draft_path = Path(args.draft_out)
    refined_path = Path(args.refined_out) if args.refined_out else Path(str(draft_path).replace("_draft.json", "_refined.json"))

    run_step(
        [
            sys.executable,
            "scripts/generate_draft_production.py",
            "--video",
            args.video,
            "--weights",
            args.weights,
            "--out",
            str(draft_path),
            "--best-of",
            str(args.best_of),
            "--stride",
            str(args.stride),
        ],
        "Generate Draft",
    )

    input_json_for_render = draft_path
    if not args.skip_refine:
        refine_cmd = [
            sys.executable,
            "scripts/ai_refine_draft.py",
            "--draft",
            str(draft_path),
            "--out",
            str(refined_path),
            "--model",
            args.model,
            "--votes",
            str(args.votes),
            "--expected-scope",
            args.expected_scope,
        ]
        if args.expected_final_set_score:
            refine_cmd.extend(["--expected-final-set-score", args.expected_final_set_score])

        run_step(
            refine_cmd,
            "Refine Winners",
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
