"""Detect the side-swap timestamp in a table tennis video.

Algorithm (independent of rally detection):
  1. Detect Table ROI and derive player zone.
  2. Sample frames at fixed step through the video.
  3. Per frame: detect bodies inside player zone, classify each as Side L
     (cx < table_center_x) or Side R, identify via face DB.
  4. Build per-identity timeline of (t, side).
  5. Smooth the side per identity over a sliding window.
  6. Establish baseline state (Set 1 sides) from the early window.
  7. Walk forward to find the earliest timestamp T* where BOTH players have
     swapped to the opposite side AND the new state is stable for >= 15 s.
  8. Backtrack from T* to find the earliest moment the swapped state began.

Output: T_swap_start (timestamp where Set 2 begins) plus a summary of how
the input timeline splits into Set 1 / Set 2 ranges.

Usage:
    python scripts/detect_side_swap.py --video inputs/raw_matches/2_sets.mp4
    python scripts/detect_side_swap.py --video inputs/raw_matches/2_sets.mp4 --sample-step 1.5 --baseline-end 60
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

from backend.player_identity import FaceDB, FaceEmbedder, face_similarity
from backend.player_identification import (
    DEFAULT_MATCH_THRESHOLD,
    detect_table_roi_and_player_zone,
    _detect_bodies_and_faces,
    _try_embed_face,
)
from backend.production_pipeline import ProductionPipelineConfig


# --- side classification --------------------------------------------------

SIDE_L = "L"
SIDE_R = "R"
SIDE_UNK = "?"


def classify_side(cx: float, table_center_x: float) -> str:
    return SIDE_L if cx < table_center_x else SIDE_R


# --- per-frame sampling ---------------------------------------------------

def sample_positions(
    video_path: str,
    yolo,
    embedder: FaceEmbedder,
    face_db: FaceDB,
    player_zone: tuple,
    table_center_x: float,
    sample_step: float,
    match_threshold: float,
    log_fn=None,
) -> list[dict]:
    """Walk the video at sample_step intervals and return per-detection records.

    Each record:
        {
            "t": float seconds,
            "identity": str | None,   player name from face DB or None
            "side": "L" | "R",
            "sim": float | None,      cosine similarity to matched record
            "rank": int,              0 = largest bbox, 1 = second largest
            "face_score": float,
        }
    """
    def _log(msg):
        if log_fn:
            log_fn(msg)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n_frames / fps if fps > 0 else 0.0
    _log(f"video: fps={fps:.2f} frames={n_frames} duration={duration:.1f}s")

    records: list[dict] = []
    t = 0.0
    while t <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if not ret:
            break
        dets = _detect_bodies_and_faces(frame, yolo, roi_xyxy=player_zone)
        for rank, det in enumerate(dets[:2]):
            cx = float((det["bbox_xyxy"][0] + det["bbox_xyxy"][2]) / 2.0)
            side = classify_side(cx, table_center_x)
            embed_result = _try_embed_face(frame, det, embedder)
            if embed_result is None:
                # Body present but no usable face this frame.
                records.append({
                    "t": t, "identity": None, "side": side,
                    "sim": None, "rank": rank, "face_score": 0.0,
                })
                continue
            emb, _disp, face_score = embed_result
            best_record = None
            best_sim = -1.0
            for rec in face_db.records:
                sim = face_similarity(emb, rec.embedding_array())
                if sim > best_sim:
                    best_sim = sim
                    best_record = rec
            identity = best_record.name if best_record is not None and best_sim >= match_threshold else None
            records.append({
                "t": t, "identity": identity, "side": side,
                "sim": float(best_sim), "rank": rank, "face_score": float(face_score),
            })
        t += sample_step
    cap.release()
    return records


# --- smoothing / state detection -----------------------------------------

def smoothed_side(timeline_for_player: list[tuple[float, str]],
                  t: float,
                  window: float = 10.0,
                  min_samples: int = 2,
                  min_majority_frac: float = 0.6) -> str | None:
    """Return the dominant side (L/R) in [t-window, t+window] for one player.

    Returns None when there are too few samples or when the majority is not
    strong enough (likely transitioning).
    """
    nearby = [s for (ts, s) in timeline_for_player if abs(ts - t) <= window]
    if len(nearby) < min_samples:
        return None
    counts = Counter(nearby)
    dominant_side, dominant_count = counts.most_common(1)[0]
    if dominant_count / len(nearby) < min_majority_frac:
        return None
    return dominant_side


def baseline_state(timeline_for_player: list[tuple[float, str]],
                   t_start: float, t_end: float) -> str | None:
    """Return the dominant side observed in [t_start, t_end]."""
    nearby = [s for (ts, s) in timeline_for_player if t_start <= ts <= t_end]
    if not nearby:
        return None
    counts = Counter(nearby)
    return counts.most_common(1)[0][0]


def _flipped_or_consistent(side: str | None, init: str, flipped: str) -> bool:
    """True if side is None (no info) or matches the flipped state.
    Used to allow the partner-symmetry inference: when one player has clearly
    flipped and the other has no contrary evidence, treat as a swap."""
    return side is None or side == flipped


def find_swap(tl_a: list[tuple[float, str]],
              tl_b: list[tuple[float, str]],
              search_start: float,
              search_end: float,
              step: float,
              init_a: str,
              init_b: str,
              stability_seconds: float,
              window: float) -> tuple[float, str] | None:
    """Walk forward from search_start; return (t, mode) at the earliest moment
    one or both players have flipped side AND state remains stable for
    >= stability_seconds.

    mode = "both"   when both players' smoothed sides are observed flipped
    mode = "a-only" when only player A has observed flipped state and B has
                    no contrary evidence (symmetry inference)
    mode = "b-only" mirror of the above
    """
    flipped_a = SIDE_R if init_a == SIDE_L else SIDE_L
    flipped_b = SIDE_R if init_b == SIDE_L else SIDE_L

    t = search_start
    while t <= search_end:
        sa = smoothed_side(tl_a, t, window=window)
        sb = smoothed_side(tl_b, t, window=window)

        a_flipped = sa == flipped_a
        b_flipped = sb == flipped_b

        candidate_mode = None
        if a_flipped and b_flipped:
            candidate_mode = "both"
        elif a_flipped and _flipped_or_consistent(sb, init_b, flipped_b):
            candidate_mode = "a-only"
        elif b_flipped and _flipped_or_consistent(sa, init_a, flipped_a):
            candidate_mode = "b-only"

        if candidate_mode is not None:
            stable = True
            tt = t
            while tt < t + stability_seconds:
                tt += step
                ts_check = smoothed_side(tl_a, tt, window=window)
                vs_check = smoothed_side(tl_b, tt, window=window)
                # Reject only if we observe an explicit return to baseline.
                if (ts_check is not None and ts_check == init_a) or \
                   (vs_check is not None and vs_check == init_b):
                    stable = False
                    break
            if stable:
                return t, candidate_mode
        t += step
    return None


def backtrack_swap_start(tl_a: list[tuple[float, str]],
                         tl_b: list[tuple[float, str]],
                         t_swap: float,
                         after_a: str,
                         after_b: str,
                         init_a: str,
                         init_b: str,
                         step: float,
                         max_lookback: float,
                         window: float) -> float:
    """Walk backward from t_swap as long as the state is consistent with the
    swapped (post-swap) configuration.  Stops when an explicit return to the
    baseline state is observed.  Treats None (no data) as compatible."""
    t = t_swap
    while t - step >= t_swap - max_lookback:
        prev = t - step
        sa = smoothed_side(tl_a, prev, window=window)
        sb = smoothed_side(tl_b, prev, window=window)
        # Reject only on explicit return to baseline state.
        if (sa is not None and sa == init_a) or (sb is not None and sb == init_b):
            break
        t = prev
    return t


# --- main -----------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--sample-step", type=float, default=2.0,
                        help="Frame sampling interval in seconds (default 2.0)")
    parser.add_argument("--baseline-end", type=float, default=60.0,
                        help="End of baseline window for Set 1 reference (default 60s)")
    parser.add_argument("--baseline-start", type=float, default=10.0)
    parser.add_argument("--smooth-window", type=float, default=10.0,
                        help="±seconds for sliding-window majority vote (default 10)")
    parser.add_argument("--stability-seconds", type=float, default=15.0,
                        help="Post-swap state must persist for this many seconds (default 15)")
    parser.add_argument("--match-threshold", type=float, default=DEFAULT_MATCH_THRESHOLD,
                        help=f"Face-DB match threshold (default {DEFAULT_MATCH_THRESHOLD})")
    parser.add_argument("--save-csv", type=str, default=None,
                        help="Optional path to dump per-sample records as CSV")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        return 1

    config = ProductionPipelineConfig()
    face_db_path = ROOT / "data" / "players" / "faces.json"
    face_model_path = ROOT / "data" / "models" / "face" / "w600k_r50.onnx"

    print(f"Video:           {video_path}")
    print(f"Table weights:   {config.table_weights_path}")
    print(f"Pose weights:    {config.pose_weights_path}")
    print(f"Face DB:         {face_db_path}")
    print()

    # Load FaceDB + face embedder
    face_db = FaceDB(face_db_path)
    if len(face_db) < 2:
        print(f"WARNING: face DB has only {len(face_db)} player(s); need at least 2 for swap detection.")
    print(f"Enrolled players: {[r.name for r in face_db.records]}")
    embedder = FaceEmbedder(face_model_path)

    # Detect table ROI + player zone
    print("\n[1/5] Detecting table ROI + player zone...")
    table_roi, player_zone = detect_table_roi_and_player_zone(
        str(video_path), config.table_weights_path,
    )
    if table_roi is None:
        print("ERROR: table ROI detection failed.")
        return 1
    table_center_x = table_roi.x + table_roi.w / 2.0
    print(f"  table ROI:    x={table_roi.x} y={table_roi.y} w={table_roi.w} h={table_roi.h}")
    print(f"  table center: x={table_center_x:.0f}")
    print(f"  player zone:  x=[{player_zone[0]:.0f},{player_zone[2]:.0f}] y=[{player_zone[1]:.0f},{player_zone[3]:.0f}]")

    # Load YOLO pose
    print("\n[2/5] Loading YOLO pose...")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required: torch.cuda.is_available() returned False")
    from ultralytics import YOLO
    yolo = YOLO(str(config.pose_weights_path))

    # Sample frames
    print(f"\n[3/5] Sampling frames every {args.sample_step}s...")
    records = sample_positions(
        str(video_path), yolo, embedder, face_db,
        player_zone=player_zone,
        table_center_x=table_center_x,
        sample_step=args.sample_step,
        match_threshold=args.match_threshold,
        log_fn=lambda m: print(f"  {m}"),
    )
    print(f"  collected {len(records)} body detections")

    # Optional CSV dump for inspection
    if args.save_csv:
        csv_path = Path(args.save_csv)
        with csv_path.open("w", encoding="utf-8") as fh:
            fh.write("t,identity,side,sim,rank,face_score\n")
            for r in records:
                fh.write(f"{r['t']:.2f},{r['identity'] or ''},{r['side']},"
                         f"{'' if r['sim'] is None else f'{r['sim']:.3f}'},"
                         f"{r['rank']},{r['face_score']:.2f}\n")
        print(f"  CSV saved: {csv_path}")

    # Determine the two main players from the actual sample distribution.
    # The face DB may contain many enrolled players; the two playing in this
    # video are the two identities that account for the most samples.
    print(f"\n[4/5] Identifying the two main players from sampled detections...")
    identity_counts = Counter(r["identity"] for r in records if r["identity"])
    if not identity_counts:
        print("ABORT: no face matched any DB record — cannot determine players.")
        print("Re-enroll players or lower --match-threshold.")
        return 1
    print("  Detection counts per identity:")
    for n, c in identity_counts.most_common():
        print(f"    {n}: {c}")
    top2 = identity_counts.most_common(2)
    if len(top2) < 2:
        print(f"ABORT: only one identity matched ({top2[0][0]} x {top2[0][1]}). "
              "Need 2 distinct players for swap detection.")
        return 1
    name_a, count_a = top2[0]
    name_b, count_b = top2[1]
    tl_a = [(r["t"], r["side"]) for r in records if r["identity"] == name_a]
    tl_b = [(r["t"], r["side"]) for r in records if r["identity"] == name_b]
    print(f"  Main players selected: A={name_a} ({count_a} samples)  B={name_b} ({count_b} samples)")
    if len(tl_a) < 5 or len(tl_b) < 3:
        print("ABORT: too few identified samples for one of the main players.")
        return 1

    # Baseline (Set 1 sides)
    init_a = baseline_state(tl_a, args.baseline_start, args.baseline_end)
    init_b = baseline_state(tl_b, args.baseline_start, args.baseline_end)
    print(f"  baseline ({args.baseline_start:.0f}s..{args.baseline_end:.0f}s):"
          f"  {name_a}={init_a}  {name_b}={init_b}")
    if init_a is None or init_b is None or init_a == init_b:
        print("ABORT: baseline sides are not opposite — cannot determine swap.")
        print("Try a different --baseline-start / --baseline-end window.")
        return 1

    # Find swap
    print(f"\n[5/5] Searching for swap (stability >= {args.stability_seconds}s)...")
    duration = max(r["t"] for r in records) if records else 0.0
    swap_result = find_swap(
        tl_a, tl_b,
        search_start=args.baseline_end,
        search_end=duration,
        step=args.sample_step,
        init_a=init_a, init_b=init_b,
        stability_seconds=args.stability_seconds,
        window=args.smooth_window,
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    if swap_result is None:
        print(f"NO SWAP DETECTED in window [{args.baseline_end:.0f}s..{duration:.0f}s].")
        print(f"  Set 1 sides: {name_a}={init_a}  {name_b}={init_b}")
        return 0
    t_swap, swap_mode = swap_result
    print(f"  Swap evidence mode: {swap_mode}")

    # Post-swap state — derived by symmetry when one player has no data.
    flipped_a = SIDE_R if init_a == SIDE_L else SIDE_L
    flipped_b = SIDE_R if init_b == SIDE_L else SIDE_L
    after_a = smoothed_side(tl_a, t_swap, window=args.smooth_window) or flipped_a
    after_b = smoothed_side(tl_b, t_swap, window=args.smooth_window) or flipped_b
    t_swap_start = backtrack_swap_start(
        tl_a, tl_b, t_swap, after_a, after_b,
        init_a=init_a, init_b=init_b,
        step=args.sample_step,
        max_lookback=30.0,
        window=args.smooth_window,
    )

    # Find last timestamp where Set 1 state was still stable
    t_last_set1 = None
    for tt in sorted({rt for rt in (r["t"] for r in records) if rt < t_swap_start}, reverse=True):
        sa = smoothed_side(tl_a, tt, window=args.smooth_window)
        sb = smoothed_side(tl_b, tt, window=args.smooth_window)
        if sa == init_a and sb == init_b:
            t_last_set1 = tt
            break

    print(f"  Set 1 state:  {name_a}={init_a}  {name_b}={init_b}")
    print(f"  Set 2 state:  {name_a}={after_a}  {name_b}={after_b}")
    print()
    print(f"  Swap window:  [{t_last_set1:.2f}s .. {t_swap_start:.2f}s]"
          if t_last_set1 is not None else
          f"  Swap window starts before sampled baseline; t_swap_start={t_swap_start:.2f}s")
    print(f"  Set 1 timeline range:  t < {t_swap_start:.2f}s")
    print(f"  Set 2 timeline range:  t >= {t_swap_start:.2f}s")
    print()
    print(f"==> SET-SWAP START TIMESTAMP: {t_swap_start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
