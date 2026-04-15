"""Quick standalone test of Step 2 (quick_identify_players_standalone) on a video.

Tests the production identification path used by run_initial_job_pipeline.
Reports: detected names, table ROI, player zone, status.

Usage:
    python test_step2_quick.py --video inputs/raw_matches/2_sets.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from backend.player_identification import quick_identify_players_standalone
from backend.player_identity import FaceDB
from backend.production_pipeline import ProductionPipelineConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return 1

    config = ProductionPipelineConfig()
    face_db_path = ROOT / "data" / "players" / "faces.json"
    face_model_path = ROOT / "data" / "models" / "face" / "w600k_r50.onnx"

    db = FaceDB(face_db_path)
    print(f"\nFace DB: {face_db_path}")
    print(f"  {len(db)} enrolled players: {[r.name for r in db.records]}")
    print(f"\nVideo: {video_path}")
    print(f"Table weights: {config.table_weights_path}")
    print(f"Pose weights:  {config.pose_weights_path}")
    print(f"\n{'='*70}")
    print("Running Step 2 (quick_identify_players_standalone)...")
    print(f"{'='*70}\n")

    result = quick_identify_players_standalone(
        video_path=str(video_path),
        pose_weights_path=config.pose_weights_path,
        face_db=db,
        face_model_path=face_model_path,
        table_weights_path=config.table_weights_path,
        log_fn=print,
    )

    print(f"\n{'='*70}")
    print(f"RESULT: status={result.status.upper()}")
    print(f"{'='*70}")
    print(f"  NEAR player : {result.near_name or '— not identified'}")
    print(f"  FAR  player : {result.far_name or '— not identified'}")
    if result.table_roi is not None:
        r = result.table_roi
        print(f"  Table ROI   : x={r.x} y={r.y} w={r.w} h={r.h}  (conf={r.confidence:.2f})")
    else:
        print(f"  Table ROI   : NOT detected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
