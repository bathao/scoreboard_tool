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


def winner_counts(points: List[dict], *, source: str) -> Tuple[int, int, int]:
    if source == "final":
        getter = lambda p: p.get("winner")
    else:
        getter = lambda p: p.get("winner_candidate", p.get("winner"))
    a = sum(1 for p in points if getter(p) == "player_a")
    b = sum(1 for p in points if getter(p) == "player_b")
    u = sum(1 for p in points if getter(p) == "unknown")
    return a, b, u


def rank_flip_candidates(points: List[dict], from_winner: str, *, source: str) -> List[dict]:
    getter = (lambda p: p.get("winner")) if source == "final" else (lambda p: p.get("winner_candidate", p.get("winner")))
    cands = [p for p in points if getter(p) == from_winner]
    # Low confidence first, then longer rally first (often merged/noisy)
    cands.sort(
        key=lambda p: (
            float(p.get("winner_confidence", p.get("confidence", 0.0))),
            -float(p.get("t_end", 0.0) - p.get("t_start", 0.0)),
        )
    )
    return cands


def _scores(point: dict) -> Tuple[float, float]:
    score_a = float(point.get("winner_score_a", 0.0) or 0.0)
    score_b = float(point.get("winner_score_b", 0.0) or 0.0)
    if score_a > 0.0 or score_b > 0.0:
        return score_a, score_b

    cand = str(point.get("winner_candidate", point.get("winner", "unknown")))
    conf = float(point.get("winner_confidence", 0.0) or 0.0)
    if cand == "player_a":
        return conf, 1.0 - conf
    if cand == "player_b":
        return 1.0 - conf, conf
    return 0.5, 0.5


def constrained_assignment(points: List[dict], expected_a: int, expected_b: int) -> List[dict]:
    rows = []
    for p in points:
        score_a, score_b = _scores(p)
        far_advantage = float(score_b - score_a)
        rows.append(
            {
                "id": p.get("id"),
                "t_start": float(p.get("t_start", 0.0)),
                "t_end": float(p.get("t_end", 0.0)),
                "candidate": p.get("winner_candidate", p.get("winner", "unknown")),
                "confidence": float(p.get("winner_confidence", 0.0) or 0.0),
                "score_a": score_a,
                "score_b": score_b,
                "far_advantage": far_advantage,
                "point_end_event": p.get("point_end_event"),
            }
        )
    rows.sort(key=lambda r: (r["far_advantage"], r["score_b"], -r["score_a"]), reverse=True)
    far_ids = {r["id"] for r in rows[:expected_b]}
    for row in rows:
        row["constrained_winner"] = "player_b" if row["id"] in far_ids else "player_a"
    rows.sort(key=lambda r: r["t_start"])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Suggest minimal manual winner flips to match target final set score.")
    ap.add_argument("--json", required=True, help="Refined JSON path")
    ap.add_argument("--expected-final-set-score", required=True, help="A-B, e.g. 11-3")
    ap.add_argument("--top", type=int, default=10, help="How many candidate rows to print")
    ap.add_argument("--winner-source", choices=["final", "candidate"], default="candidate", help="Use final winner or winner_candidate as the current source")
    ap.add_argument("--show-constrained", action="store_true", help="Print the rallies picked by score-constrained assignment")
    args = ap.parse_args()

    path = Path(args.json)
    points = load_points(path)
    exp_a, exp_b = parse_score(args.expected_final_set_score)
    cur_a, cur_b, cur_u = winner_counts(points, source=args.winner_source)

    print(f"File: {path}")
    print(f"Current: A={cur_a}, B={cur_b}, unknown={cur_u}, total={len(points)}")
    print(f"Expected final set score: A={exp_a}, B={exp_b}")

    delta_a = exp_a - cur_a
    delta_b = exp_b - cur_b
    print(f"Delta: A={delta_a:+d}, B={delta_b:+d}")

    if cur_u > 0 and args.winner_source == "final":
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
        cands = rank_flip_candidates(points, "player_a", source=args.winner_source)[:needed]
        print(f"Flip {needed} rallies from player_a -> player_b")
        for p in cands[: args.top]:
            dur = float(p.get("t_end", 0.0)) - float(p.get("t_start", 0.0))
            print(
                f"- {p.get('id')}  {p.get('t_start'):.3f}-{p.get('t_end'):.3f}s"
                f"  conf={float(p.get('winner_confidence', p.get('confidence', 0.0))):.2f}  dur={dur:.2f}s"
            )
    elif delta_a > 0 and delta_b < 0:
        needed = min(delta_a, -delta_b)
        cands = rank_flip_candidates(points, "player_b", source=args.winner_source)[:needed]
        print(f"Flip {needed} rallies from player_b -> player_a")
        for p in cands[: args.top]:
            dur = float(p.get("t_end", 0.0)) - float(p.get("t_start", 0.0))
            print(
                f"- {p.get('id')}  {p.get('t_start'):.3f}-{p.get('t_end'):.3f}s"
                f"  conf={float(p.get('winner_confidence', p.get('confidence', 0.0))):.2f}  dur={dur:.2f}s"
            )
    else:
        print("Delta pattern is not a simple 1-to-1 flip. Check segmentation and unknown labels.")

    if args.show_constrained:
        print("\nScore-constrained assignment (best guess under the expected final score):")
        rows = constrained_assignment(points, exp_a, exp_b)
        chosen_far = [r for r in rows if r["constrained_winner"] == "player_b"]
        print(f"Selected {len(chosen_far)} far-win candidates:")
        for row in chosen_far[: args.top]:
            print(
                f"- {row['id']}  {row['t_start']:.3f}-{row['t_end']:.3f}s"
                f"  cand={row['candidate']}  conf={row['confidence']:.2f}"
                f"  score_a={row['score_a']:.2f}  score_b={row['score_b']:.2f}"
                f"  far_adv={row['far_advantage']:.2f}  event={row['point_end_event']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
