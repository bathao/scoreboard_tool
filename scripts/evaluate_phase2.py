from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_contract import DraftMatch, DraftPointEvent, load_draft_match, needs_human_review


@dataclass(frozen=True)
class Pair:
    name: str
    pred_path: Path
    gt_path: Path


@dataclass
class ClipMetrics:
    name: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_iou: float
    winner_accuracy: float
    winner_known_pairs: int
    review_rate: float
    unknown_rate: float
    pred_points: int
    gt_points: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_iou": self.mean_iou,
            "winner_accuracy": self.winner_accuracy,
            "winner_known_pairs": self.winner_known_pairs,
            "review_rate": self.review_rate,
            "unknown_rate": self.unknown_rate,
            "pred_points": self.pred_points,
            "gt_points": self.gt_points,
        }


def _safe_div(n: float, d: float) -> float:
    return n / d if d > 0 else 0.0


def segment_iou(a: DraftPointEvent, b: DraftPointEvent) -> float:
    left = max(float(a.t_start), float(b.t_start))
    right = min(float(a.t_end), float(b.t_end))
    inter = max(0.0, right - left)
    if inter <= 0:
        return 0.0
    union = max(float(a.t_end), float(b.t_end)) - min(float(a.t_start), float(b.t_start))
    if union <= 0:
        return 0.0
    return inter / union


def best_matches(
    pred_points: List[DraftPointEvent],
    gt_points: List[DraftPointEvent],
    *,
    iou_threshold: float,
) -> List[Tuple[int, int, float]]:
    candidates: List[Tuple[float, int, int]] = []
    for pi, p in enumerate(pred_points):
        for gi, g in enumerate(gt_points):
            iou = segment_iou(p, g)
            if iou >= iou_threshold:
                candidates.append((iou, pi, gi))

    candidates.sort(key=lambda x: x[0], reverse=True)

    used_pred = set()
    used_gt = set()
    matches: List[Tuple[int, int, float]] = []
    for iou, pi, gi in candidates:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, iou))
    return matches


def winner_is_known(w: str) -> bool:
    return w in ("player_a", "player_b")


def evaluate_clip(name: str, pred: DraftMatch, gt: DraftMatch, iou_threshold: float) -> ClipMetrics:
    pred_points = list(pred.points)
    gt_points = list(gt.points)

    matches = best_matches(pred_points, gt_points, iou_threshold=iou_threshold)
    tp = len(matches)
    fp = max(0, len(pred_points) - tp)
    fn = max(0, len(gt_points) - tp)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    mean_iou = _safe_div(sum(m[2] for m in matches), len(matches))

    # Winner accuracy only on matched pairs where GT winner is known.
    winner_total = 0
    winner_correct = 0
    for pi, gi, _ in matches:
        p = pred_points[pi]
        g = gt_points[gi]
        if not winner_is_known(g.winner):
            continue
        winner_total += 1
        if p.winner == g.winner:
            winner_correct += 1
    winner_accuracy = _safe_div(winner_correct, winner_total)

    review_count = sum(1 for p in pred_points if needs_human_review(p))
    unknown_count = sum(1 for p in pred_points if p.winner == "unknown")
    review_rate = _safe_div(review_count, len(pred_points))
    unknown_rate = _safe_div(unknown_count, len(pred_points))

    return ClipMetrics(
        name=name,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=mean_iou,
        winner_accuracy=winner_accuracy,
        winner_known_pairs=winner_total,
        review_rate=review_rate,
        unknown_rate=unknown_rate,
        pred_points=len(pred_points),
        gt_points=len(gt_points),
    )


def parse_pairs(
    *,
    pred: Optional[str],
    gt: Optional[str],
    name: Optional[str],
    manifest: Optional[str],
) -> List[Pair]:
    if manifest:
        data = json.loads(Path(manifest).read_text(encoding="utf-8"))
        raw_pairs = data.get("pairs", [])
        out: List[Pair] = []
        for item in raw_pairs:
            out.append(
                Pair(
                    name=str(item.get("name", Path(str(item["pred"])).stem)),
                    pred_path=Path(str(item["pred"])),
                    gt_path=Path(str(item["gt"])),
                )
            )
        return out

    if not pred or not gt:
        raise ValueError("Either --manifest OR both --pred and --gt are required.")

    clip_name = name if name else Path(pred).stem
    return [Pair(name=clip_name, pred_path=Path(pred), gt_path=Path(gt))]


def print_table(rows: List[ClipMetrics]) -> None:
    print(
        "name".ljust(18),
        "P".rjust(6),
        "R".rjust(6),
        "F1".rjust(6),
        "mIoU".rjust(7),
        "WinAcc".rjust(8),
        "Review".rjust(8),
        "Unknown".rjust(8),
        "TP/FP/FN".rjust(12),
    )
    print("-" * 90)
    for r in rows:
        print(
            r.name.ljust(18),
            f"{r.precision:.3f}".rjust(6),
            f"{r.recall:.3f}".rjust(6),
            f"{r.f1:.3f}".rjust(6),
            f"{r.mean_iou:.3f}".rjust(7),
            f"{r.winner_accuracy:.3f}".rjust(8),
            f"{r.review_rate:.3f}".rjust(8),
            f"{r.unknown_rate:.3f}".rjust(8),
            f"{r.tp}/{r.fp}/{r.fn}".rjust(12),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 benchmark evaluator for rally and winner quality.")
    parser.add_argument("--pred", help="Predicted draft/refined JSON")
    parser.add_argument("--gt", help="Ground-truth JSON (same ai_contract format)")
    parser.add_argument("--name", default=None, help="Optional clip name when using --pred/--gt")
    parser.add_argument("--manifest", default=None, help="Manifest JSON with list of {name,pred,gt}")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for rally matching")
    parser.add_argument("--out", default=None, help="Optional output JSON report path")
    args = parser.parse_args()

    pairs = parse_pairs(pred=args.pred, gt=args.gt, name=args.name, manifest=args.manifest)
    clip_metrics: List[ClipMetrics] = []

    for pair in pairs:
        pred_match = load_draft_match(pair.pred_path)
        gt_match = load_draft_match(pair.gt_path)
        m = evaluate_clip(pair.name, pred_match, gt_match, iou_threshold=float(args.iou_threshold))
        clip_metrics.append(m)

    print_table(clip_metrics)

    # Aggregate (micro for detection counts, macro for rates)
    tp = sum(x.tp for x in clip_metrics)
    fp = sum(x.fp for x in clip_metrics)
    fn = sum(x.fn for x in clip_metrics)
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * p * r, p + r)

    agg = {
        "clips": len(clip_metrics),
        "iou_threshold": float(args.iou_threshold),
        "micro_detection": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f1,
        },
        "macro_quality": {
            "mean_iou": _safe_div(sum(x.mean_iou for x in clip_metrics), len(clip_metrics)),
            "winner_accuracy": _safe_div(sum(x.winner_accuracy for x in clip_metrics), len(clip_metrics)),
            "review_rate": _safe_div(sum(x.review_rate for x in clip_metrics), len(clip_metrics)),
            "unknown_rate": _safe_div(sum(x.unknown_rate for x in clip_metrics), len(clip_metrics)),
        },
    }

    print("\nAggregate:")
    print(json.dumps(agg, indent=2))

    if args.out:
        report = {
            "aggregate": agg,
            "clips": [m.to_dict() for m in clip_metrics],
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
