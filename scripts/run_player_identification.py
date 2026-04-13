"""Debug script — run player identification on a processed job.

Loads an existing job (that has already completed rally detection + set_number assignment),
then runs the Two-Tier identification pipeline and reports results.

Usage:
    python scripts/run_player_identification.py --job-id 20260413T120000Z__2_sets
    python scripts/run_player_identification.py --video inputs/raw_matches/2_sets.mp4 [--best-of 3]

If --job-id is provided, loads the existing timeline from that job.
If --video is provided, runs rally detection first (slow), then identification.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force UTF-8 output on Windows (needed for Vietnamese player names)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from backend.player_identification import run_player_identification, jersey_distance
from backend.player_identity import FaceDB
from backend.production_pipeline import ProductionPipelineConfig
from backend.rally_timeline_contract import load_rally_timeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--job-id", help="Existing job ID to load timeline from")
    parser.add_argument("--video", help="Input video path (runs detection first)")
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Face similarity threshold (default 0.35)",
    )
    parser.add_argument(
        "--enroll",
        action="store_true",
        help="After identification, interactively enroll unknown faces into DB",
    )
    parser.add_argument("--near", help="Name to enroll for NEAR player (skips interactive prompt)")
    parser.add_argument("--far",  help="Name to enroll for FAR player (skips interactive prompt)")
    args = parser.parse_args()

    if not args.job_id and not args.video:
        parser.print_help()
        return 1

    config = ProductionPipelineConfig()
    face_db_path = ROOT / "data" / "players" / "faces.json"
    face_model_path = ROOT / "data" / "models" / "face" / "w600k_r50.onnx"

    # --- Check model ---
    if not face_model_path.exists():
        print(f"ERROR: ArcFace model not found at {face_model_path}")
        print(f"Run first: python scripts/download_face_models.py")
        return 1

    # --- Load face DB ---
    db = FaceDB(face_db_path)
    print(f"\nFace DB: {face_db_path}")
    print(f"  {len(db)} enrolled players: {[r.name for r in db.records]}\n")

    # --- Load or build timeline ---
    if args.job_id:
        import json as _json
        job_dir = ROOT / "runtime_jobs" / args.job_id
        job_json = job_dir / "job.json"
        if not job_json.exists():
            print(f"ERROR: Job not found: {job_dir}")
            return 1
        job_data = _json.loads(job_json.read_text(encoding="utf-8"))
        timeline_path = Path(job_data["artifacts"]["timeline_json_path"])
        if not timeline_path.exists():
            print(f"ERROR: Timeline not found: {timeline_path}")
            return 1
        video_path = job_data["artifacts"]["working_video_path"]
        timeline = load_rally_timeline(timeline_path)
        print(f"Loaded timeline: {timeline_path.name}")
        print(f"Video: {video_path}")
        print(f"Loaded timeline from: {timeline_path}")
        print(f"Video: {video_path}")
    else:
        # Run detection first
        print("Building timeline (rally detection + set boundaries)...")
        from backend.production_jobs import create_match_job, update_job_runtime_state
        from backend.production_pipeline import trim_input_video, _load_build_rally_timeline, _job_log
        from backend.rally_timeline_contract import save_rally_timeline
        from backend.set_boundary import apply_set_numbers, populate_player_positions

        video_path_obj = Path(args.video).resolve()
        if not video_path_obj.exists():
            print(f"ERROR: Video not found: {video_path_obj}")
            return 1

        job = create_match_job(
            raw_video_path=str(video_path_obj),
            player_a_name="Player A",
            player_b_name="Player B",
            trim_start_sec=0.0,
            best_of=args.best_of,
            job_purpose="output_only",
        )
        job_dir = Path(job.artifacts.job_dir)
        print(f"Job ID: {job.job_id}")

        trim_input_video(job.raw_video_path, job.artifacts.working_video_path, 0.0)
        build_rally_timeline = _load_build_rally_timeline()
        timeline = build_rally_timeline(
            job.artifacts.working_video_path,
            config.table_weights_path,
            pose_weights_path=config.pose_weights_path,
            best_of=job.best_of,
            stride=config.rally_stride,
            mode=config.rally_mode,
            player_margin_px=config.rally_player_margin_px,
            player_fuse_gain=config.rally_player_fuse_gain,
            player_signal_source=config.rally_player_signal_source,
            ball_fuse_gain=config.rally_ball_fuse_gain,
            ball_signal_source=config.rally_ball_signal_source,
            log_fn=lambda msg: _job_log(job_dir, msg),
        )
        populate_player_positions(timeline, job.artifacts.working_video_path, config.pose_weights_path)
        apply_set_numbers(timeline, best_of=job.best_of)
        save_rally_timeline(Path(job.artifacts.timeline_json_path), timeline)
        video_path = job.artifacts.working_video_path

    # Print timeline summary
    pts = timeline.points
    set_nums = sorted(set(p.set_number for p in pts))
    print(f"\nTimeline: {len(pts)} rallies, sets={set_nums}")
    for sn in set_nums:
        sp = [p for p in pts if p.set_number == sn]
        print(f"  Set {sn}: {len(sp)} rallies  ({sp[0].t_end:.0f}s .. {sp[-1].t_end:.0f}s)")

    # --- Run identification ---
    print(f"\n{'='*60}")
    print("Running Two-Tier Player Identification...")
    print(f"{'='*60}\n")

    result = run_player_identification(
        timeline=timeline,
        video_path=str(video_path),
        pose_weights_path=config.pose_weights_path,
        face_db=db,
        face_model_path=face_model_path,
        match_threshold=args.threshold,
        log_fn=print,
    )

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"Result: {result.status.upper()}")
    print(f"{'='*60}")
    print(f"  NEAR player : {result.near_name or '— not identified'}")
    print(f"  FAR  player : {result.far_name or '— not identified'}")

    if result.near_jersey_hist is not None and result.far_jersey_hist is not None:
        dist = jersey_distance(result.near_jersey_hist, result.far_jersey_hist)
        print(f"  Jersey dist : {dist:.4f}  ({'AMBIGUOUS' if dist < 0.12 else 'OK'})")
    else:
        print(f"  Jersey      : not extracted")

    if result.unknown_faces:
        import cv2 as _cv2
        # Save top-3 crops per role so operator can pick the best one
        crops_dir = ROOT / "data" / "face_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        crop_paths = []
        for uf in result.unknown_faces:
            saved_paths = []
            for rank_i, (score, t_sec, crop_bgr) in enumerate(uf.top_crops):
                fname = f"unknown_{uf.body_role}_top{rank_i+1}_t{t_sec:.1f}s_score{score:.2f}.jpg"
                crop_path = crops_dir / fname
                _cv2.imwrite(str(crop_path), crop_bgr)
                saved_paths.append(crop_path)
                print(f"  [{uf.body_role.upper()} top-{rank_i+1}] score={score:.2f}  t={t_sec:.1f}s  -> {crop_path.name}")
            if saved_paths:
                crop_paths.append((uf, saved_paths[0]))  # best crop for enrollment

        # --near / --far flags: enroll without interactive prompt
        preset = {"near": args.near, "far": args.far}

        if (args.enroll or args.near or args.far) and crop_paths:
            import cv2 as _cv2
            for uf, crop_path in crop_paths:
                name = preset.get(uf.body_role)
                if name:
                    db.enroll(name, uf.face_embedding)
                    print(f"  Enrolled: {name!r} as {uf.body_role.upper()}")
                elif args.enroll:
                    # Interactive: show the crop first, then ask
                    img = _cv2.imread(str(crop_path))
                    if img is not None:
                        display = _cv2.resize(img, (336, 336), interpolation=_cv2.INTER_NEAREST)
                        _cv2.imshow(f"{uf.body_role.upper()} player", display)
                        _cv2.waitKey(1)
                    name = input(f"  Enter name for {uf.body_role.upper()} player (or Enter to skip): ").strip()
                    _cv2.destroyAllWindows()
                    if name:
                        db.enroll(name, uf.face_embedding)
                        print(f"  Enrolled: {name!r} as {uf.body_role.upper()}")
            if db._dirty:
                db.save()
                print(f"  DB saved: {face_db_path}  ({len(db)} players total)")
        elif result.unknown_faces:
            print(f"\n  Re-run with --enroll (interactive) or --near 'Name' --far 'Name' to register.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
