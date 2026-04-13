"""Diagnostic script — analyze set boundary detection on any job timeline.

Usage:
    python scripts/diagnose_set_boundaries.py <job_id_or_timeline_path>

Examples:
    python scripts/diagnose_set_boundaries.py 20260411T122603Z__match_test
    python scripts/diagnose_set_boundaries.py runtime_jobs/20260411T122603Z__match_test/timeline_review.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.rally_timeline_contract import load_rally_timeline
from backend.set_boundary import (
    GAP_THRESHOLD_SECONDS,
    detect_boundaries_by_gap,
    detect_boundaries_by_score,
    assign_set_numbers,
    get_inter_rally_gaps,
)
from backend.production_jobs import effective_rally_winner


def _find_timeline(arg: str) -> Path:
    p = Path(arg)
    if p.exists() and p.suffix == ".json":
        return p
    # Maybe a job_id
    candidate = ROOT / "runtime_jobs" / arg / "timeline_review.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot find timeline for: {arg}")


def main(arg: str) -> None:
    tl_path = _find_timeline(arg)
    print(f"Loading: {tl_path}")
    timeline = load_rally_timeline(tl_path)
    pts = timeline.points
    print(f"Total rallies: {len(pts)}")
    print()

    # --- Gap analysis ---
    gaps = get_inter_rally_gaps(timeline)
    gap_candidates = detect_boundaries_by_gap(timeline, GAP_THRESHOLD_SECONDS)
    score_candidates = detect_boundaries_by_score(timeline)

    print(f"{'#':<4} {'ID':<14} {'t_start':>8} {'t_end':>8} {'gap_after':>10}  {'winner':<10}  {'set# (stored)':>13}")
    print("-" * 75)
    for i, pt in enumerate(pts):
        gap_after = gaps[i] if i < len(gaps) else None
        winner = effective_rally_winner(pt) or "unknown"
        gap_str = f"{gap_after:>8.1f}s" if gap_after is not None else "        "
        boundary_flag = ""
        if gap_after is not None and gap_after >= GAP_THRESHOLD_SECONDS:
            boundary_flag = "  <-- GAP BOUNDARY"
        if i + 1 in score_candidates:
            boundary_flag += "  <-- SCORE BOUNDARY"
        print(f"{i:<4} {pt.id:<14} {pt.t_start:>8.2f}s {pt.t_end:>8.2f}s {gap_str}{boundary_flag}   {winner:<10}  set={pt.set_number}")
    print()

    # --- Assigned set numbers ---
    assigned = assign_set_numbers(timeline)
    print("Assigned set_number by detection:")
    for i, (pt, sn) in enumerate(zip(pts, assigned)):
        stored = pt.set_number
        diff = " <-- MISMATCH" if sn != stored else ""
        print(f"  rally {i:>2} {pt.id}: detected={sn}  stored={stored}{diff}")
    print()

    # --- Summary ---
    print(f"Gap threshold used: {GAP_THRESHOLD_SECONDS}s")
    print(f"Gap boundary candidates (rally indices): {gap_candidates}")
    print(f"Score boundary candidates (rally indices): {score_candidates}")
    print(f"Final boundary set: {sorted(set(assign_set_numbers.__wrapped__(timeline) if hasattr(assign_set_numbers, '__wrapped__') else []))}")

    if not gap_candidates:
        print()
        print("NOTE: No gap boundaries found.")
        print("  -- If this is a multi-set video, possible reasons:")
        print(f"     1. Between-set break is shorter than {GAP_THRESHOLD_SECONDS}s threshold")
        print("     2. Rally timestamps are off (pipeline may not have detected all rallies)")
        max_gap = max(gaps) if gaps else 0
        print(f"     Largest gap observed: {max_gap:.1f}s")
        print(f"     Try lowering GAP_THRESHOLD_SECONDS in backend/set_boundary.py if needed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
