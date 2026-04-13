"""Debug script — run rally detection only (Steps 1+2), then report set boundary analysis.

Skips Steps 3+4 (clip export, AI winner prediction) to save time.
Useful for quickly verifying gap/set detection on a new video.

Usage:
    python scripts/debug_set_boundaries.py --video "inputs/raw_matches/2_sets.mp4" [--best-of 3] [--trim 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.production_defaults import PRODUCTION_RALLY_DEFAULTS
from backend.production_jobs import create_match_job, update_job_runtime_state
from backend.production_pipeline import (
    ProductionPipelineConfig,
    trim_input_video,
    _load_build_rally_timeline,
    _job_log,
)
from backend.rally_timeline_contract import save_rally_timeline
from backend.set_boundary import (
    GAP_THRESHOLD_SECONDS,
    apply_set_numbers,
    detect_boundaries_by_gap,
    detect_boundaries_by_position,
    detect_boundaries_by_score,
    get_inter_rally_gaps,
    populate_player_positions,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--best-of", type=int, default=5, help="Best-of format (3, 5, 7)")
    parser.add_argument("--trim", type=float, default=0.0, help="Trim start in seconds")
    parser.add_argument("--gap-threshold", type=float, default=GAP_THRESHOLD_SECONDS, help=f"Gap threshold in seconds (default {GAP_THRESHOLD_SECONDS})")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return 1

    config = ProductionPipelineConfig()

    print(f"\n{'='*60}")
    print(f"Debug Set Boundary Detection")
    print(f"{'='*60}")
    print(f"Video   : {video_path.name}")
    print(f"Best-of : {args.best_of}")
    print(f"Trim    : {args.trim}s")
    print(f"Gap thr : {args.gap_threshold}s")
    print(f"{'='*60}\n")

    # Create job (saves to runtime_jobs/)
    job = create_match_job(
        raw_video_path=str(video_path),
        player_a_name="Player A",
        player_b_name="Player B",
        trim_start_sec=args.trim,
        best_of=args.best_of,
        job_purpose="output_only",
    )
    print(f"Job ID  : {job.job_id}")
    print(f"Job dir : {job.artifacts.job_dir}\n")

    job_dir = job.artifacts.job_dir

    # Step 1: trim
    print("[Step 1/2] Trimming input video...")
    trim_input_video(job.raw_video_path, job.artifacts.working_video_path, job.trim_start_sec)
    update_job_runtime_state(job, status="running", current_step="generate_rally_timeline")
    print("[Step 1/2] Done.\n")

    # Step 2: rally detection (the slow step)
    print("[Step 2/2] Detecting rallies with YOLO (this is the slow step)...")
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
    timeline.video_path = str(Path(job.artifacts.working_video_path).resolve()).replace("\\", "/")

    pts = timeline.points
    print(f"[Step 2/2] Done — {len(pts)} rallies detected.\n")

    if not pts:
        print("No rallies found. Check video content and weights.")
        return 1

    # Step 3: lightweight player-position extraction for Signal 3
    print("[Signal 3] Extracting player X positions per rally (fast YOLO pass)...")
    populate_player_positions(
        timeline,
        video_path=job.artifacts.working_video_path,
        pose_weights_path=config.pose_weights_path,
    )
    has_positions = any(p.player_a_mean_x is not None for p in pts)
    print(f"[Signal 3] Done — positions populated: {has_positions}\n")

    # Apply cross-validated set boundary detection, then save
    apply_set_numbers(timeline, best_of=job.best_of, min_gap_sec=args.gap_threshold)
    save_rally_timeline(Path(job.artifacts.timeline_json_path), timeline)
    update_job_runtime_state(job, status="needs_review", current_step="ai_ready", timeline=timeline)

    # --- Analysis table ---
    gaps = get_inter_rally_gaps(timeline)
    gap_candidates = detect_boundaries_by_gap(timeline, args.gap_threshold)
    score_candidates = detect_boundaries_by_score(timeline, job.best_of)
    pos_candidates = detect_boundaries_by_position(timeline)

    print(f"{'='*95}")
    print(f"{'#':<4} {'ID':<14} {'t_start':>9} {'t_end':>9} {'gap':>7}  {'near_x':>7}  {'far_x':>7}  set#  signals")
    print(f"{'-'*95}")
    for i, pt in enumerate(pts):
        gap_after = gaps[i] if i < len(gaps) else None
        gap_str = f"{gap_after:>6.1f}s" if gap_after is not None else "       "
        ax_str = f"{pt.player_a_mean_x:>6.0f}" if pt.player_a_mean_x is not None else "     -"
        bx_str = f"{pt.player_b_mean_x:>6.0f}" if pt.player_b_mean_x is not None else "     -"
        flags = []
        if i + 1 in gap_candidates:
            flags.append("GAP")
        if i + 1 in score_candidates:
            flags.append("SCORE")
        if i + 1 in pos_candidates:
            flags.append("SWAP")
        flag_str = " ".join(flags)
        boundary_mark = "  <<< BOUNDARY" if flags else ""
        print(f"{i:<4} {pt.id:<14} {pt.t_start:>8.2f}s {pt.t_end:>8.2f}s {gap_str}  {ax_str}  {bx_str}  set={pt.set_number}  {flag_str}{boundary_mark}")

    print(f"\n{'='*80}")
    max_gap = max(gaps) if gaps else 0
    print(f"Signal 1 (score)    boundaries: {score_candidates}")
    print(f"Signal 2 (gap≥{args.gap_threshold:.0f}s) boundaries: {gap_candidates}  (max gap = {max_gap:.1f}s)")
    print(f"Signal 3 (side swap) boundaries: {pos_candidates}")

    all_detected = sorted(set(gap_candidates) | set(score_candidates) | set(pos_candidates))
    if all_detected:
        print(f"\n✓ Set boundaries detected at rally indices: {all_detected}")
    else:
        print(f"\n✗ No set boundaries detected")
        print(f"  Max gap = {max_gap:.1f}s  (threshold = {args.gap_threshold}s)")
        if not has_positions:
            print(f"  Signal 3 not available (YOLO position extraction failed)")

    set_nums = sorted(set(pt.set_number for pt in pts))
    print(f"\nSet numbers assigned: {set_nums}")
    for sn in set_nums:
        rallies_in_set = [pt.id for pt in pts if pt.set_number == sn]
        print(f"  Set {sn}: {len(rallies_in_set)} rallies  ({rallies_in_set[0]} .. {rallies_in_set[-1]})")

    print(f"\nJob ID (for Web UI): {job.job_id}")
    print(f"Diagnose later with: python scripts/diagnose_set_boundaries.py {job.job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
