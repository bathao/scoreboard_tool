from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def parse_score(score: str) -> Tuple[int, int]:
    a, b = score.split("-")
    return int(a.strip()), int(b.strip())


def load_points(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("points", []))


def winner_counts(points: List[dict]) -> Tuple[int, int, int]:
    a = sum(1 for p in points if p.get("winner") == "player_a")
    b = sum(1 for p in points if p.get("winner") == "player_b")
    u = sum(1 for p in points if p.get("winner") == "unknown")
    return a, b, u


def rank_flip_candidates(points: List[dict], from_winner: str) -> List[dict]:
    cands = [p for p in points if p.get("winner") == from_winner]
    # Low confidence first, then longer rally first (often merged/noisy)
    cands.sort(
        key=lambda p: (
            float(p.get("confidence", 0.0)),
            -float(p.get("t_end", 0.0) - p.get("t_start", 0.0)),
        )
    )
    return cands


def main() -> int:
    ap = argparse.ArgumentParser(description="Suggest minimal manual winner flips to match target final set score.")
    ap.add_argument("--json", required=True, help="Refined JSON path")
    ap.add_argument("--expected-final-set-score", required=True, help="A-B, e.g. 11-3")
    ap.add_argument("--top", type=int, default=10, help="How many candidate rows to print")
    args = ap.parse_args()

    path = Path(args.json)
    points = load_points(path)
    exp_a, exp_b = parse_score(args.expected_final_set_score)
    cur_a, cur_b, cur_u = winner_counts(points)

    print(f"File: {path}")
    print(f"Current: A={cur_a}, B={cur_b}, unknown={cur_u}, total={len(points)}")
    print(f"Expected final set score: A={exp_a}, B={exp_b}")

    delta_a = exp_a - cur_a
    delta_b = exp_b - cur_b
    print(f"Delta: A={delta_a:+d}, B={delta_b:+d}")

    if cur_u > 0:
        print("\nNote: Unknown winners exist. Resolve them first, then re-run.")

    # Minimal flip plan if totals match
    if (cur_a + cur_b) != (exp_a + exp_b):
        print("\nCannot reconcile by winner flips only: total known points != expected total points.")
        print("This indicates segmentation error (missing/extra rallies).")
        return 0

    if delta_a == 0 and delta_b == 0:
        print("\nNo winner flips needed. Score already matches.")
        return 0

    print("\nSuggested manual flips (not auto-applied):")
    if delta_b > 0 and delta_a < 0:
        needed = min(-delta_a, delta_b)
        cands = rank_flip_candidates(points, "player_a")[:needed]
        print(f"Flip {needed} rallies from player_a -> player_b")
        for p in cands[: args.top]:
            dur = float(p.get("t_end", 0.0)) - float(p.get("t_start", 0.0))
            print(
                f"- {p.get('id')}  {p.get('t_start'):.3f}-{p.get('t_end'):.3f}s"
                f"  conf={float(p.get('confidence', 0.0)):.2f}  dur={dur:.2f}s"
            )
    elif delta_a > 0 and delta_b < 0:
        needed = min(delta_a, -delta_b)
        cands = rank_flip_candidates(points, "player_b")[:needed]
        print(f"Flip {needed} rallies from player_b -> player_a")
        for p in cands[: args.top]:
            dur = float(p.get("t_end", 0.0)) - float(p.get("t_start", 0.0))
            print(
                f"- {p.get('id')}  {p.get('t_start'):.3f}-{p.get('t_end'):.3f}s"
                f"  conf={float(p.get('confidence', 0.0)):.2f}  dur={dur:.2f}s"
            )
    else:
        print("Delta pattern is not a simple 1-to-1 flip. Check segmentation and unknown labels.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
