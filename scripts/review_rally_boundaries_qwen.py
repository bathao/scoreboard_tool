from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import ollama

sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_contract import DraftMatch, DraftPointEvent, load_draft_match, save_draft_match


@dataclass
class BoundaryCandidate:
    left_idx: int
    right_idx: int
    boundary_time: float
    gap_sec: float
    left_point: DraftPointEvent
    right_point: DraftPointEvent


def _extract_json_block(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _candidate_boundaries(draft: DraftMatch, *, max_gap_sec: float = 0.12) -> List[BoundaryCandidate]:
    points = list(draft.points)
    candidates: List[BoundaryCandidate] = []
    for idx in range(len(points) - 1):
        left = points[idx]
        right = points[idx + 1]
        gap = float(right.t_start - left.t_end)
        if gap > max_gap_sec:
            continue
        left_split = "split_long" in left.flags
        right_split = "split_long" in right.flags
        if gap <= 0.02 or left_split or right_split:
            boundary_time = float((left.t_end + right.t_start) / 2.0)
            candidates.append(
                BoundaryCandidate(
                    left_idx=idx,
                    right_idx=idx + 1,
                    boundary_time=boundary_time,
                    gap_sec=gap,
                    left_point=left,
                    right_point=right,
                )
            )
    return candidates


def _read_frame_at(cap: cv2.VideoCapture, t_sec: float) -> Optional[Any]:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t_sec)) * 1000.0)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def _annotate_frame(frame: Any, label: str) -> Any:
    out = frame.copy()
    cv2.putText(
        out,
        label,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def build_boundary_grid(
    video_path: str,
    candidate: BoundaryCandidate,
    out_dir: Path,
) -> Path:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    bt = float(candidate.boundary_time)
    offsets = [-1.2, -0.7, -0.25, 0.25, 0.7, 1.2]
    frames: List[Any] = []
    for off in offsets:
        t_sec = max(0.0, bt + off)
        frame = _read_frame_at(cap, t_sec)
        if frame is None:
            continue
        frame = cv2.resize(frame, (640, 360))
        frames.append(_annotate_frame(frame, f"t={t_sec:.2f}s"))
    cap.release()

    if len(frames) != 6:
        raise RuntimeError(f"Unable to build complete boundary grid for {candidate.left_point.id}/{candidate.right_point.id}")

    top = cv2.hconcat(frames[:3])
    bottom = cv2.hconcat(frames[3:])
    grid = cv2.vconcat([top, bottom])
    cv2.putText(
        grid,
        f"Boundary: {candidate.left_point.id} -> {candidate.right_point.id}",
        (18, grid.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{candidate.left_point.id}_{candidate.right_point.id}_boundary.jpg"
    cv2.imwrite(str(out_path), grid)
    return out_path


def review_boundary_with_vision(
    *,
    model_name: str,
    image_path: Path,
    candidate: BoundaryCandidate,
) -> Dict[str, Any]:
    with open(image_path, "rb") as f:
        image_data = f.read()

    prompt = (
        "You are reviewing a proposed boundary between two detected table-tennis rally segments. "
        "The 6 frames are in chronological order around the boundary time. "
        "Decide whether the rally continues across the boundary or whether the ball is dead/reset there. "
        "Return strict JSON only with keys: "
        'boundary_label ("continuous_rally" | "true_split" | "uncertain"), '
        '"confidence" (0.0-1.0), '
        '"reason" (max 20 words).'
    )
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_data],
            }
        ],
        options={"temperature": 0.0},
    )
    raw = response["message"]["content"]
    data = _extract_json_block(raw)
    return {
        "raw": raw,
        "boundary_label": str(data.get("boundary_label", "uncertain")),
        "confidence": float(data.get("confidence", 0.0) or 0.0),
        "reason": str(data.get("reason", "")).strip(),
    }


def review_boundary_with_reasoner(
    *,
    model_name: str,
    candidate: BoundaryCandidate,
    vision_result: Dict[str, Any],
) -> Dict[str, Any]:
    payload = {
        "left_point": {
            "id": candidate.left_point.id,
            "t_start": candidate.left_point.t_start,
            "t_end": candidate.left_point.t_end,
            "duration_sec": candidate.left_point.t_end - candidate.left_point.t_start,
            "confidence": candidate.left_point.confidence,
            "flags": candidate.left_point.flags,
        },
        "right_point": {
            "id": candidate.right_point.id,
            "t_start": candidate.right_point.t_start,
            "t_end": candidate.right_point.t_end,
            "duration_sec": candidate.right_point.t_end - candidate.right_point.t_start,
            "confidence": candidate.right_point.confidence,
            "flags": candidate.right_point.flags,
        },
        "boundary": {
            "time_sec": candidate.boundary_time,
            "gap_sec": candidate.gap_sec,
        },
        "vision_review": {
            "boundary_label": vision_result.get("boundary_label"),
            "confidence": vision_result.get("confidence"),
            "reason": vision_result.get("reason"),
        },
    }
    prompt = (
        "You are a strict table-tennis rally-boundary reviewer. "
        "Use the payload below to decide whether two adjacent segments should be merged into one rally. "
        "Be conservative: merge only when evidence strongly suggests one continuous rally. "
        "Return strict JSON only with keys: "
        'action ("merge" | "keep_split" | "review"), '
        '"confidence" (0.0-1.0), '
        '"reason" (max 20 words).'
        "\n\nPayload:\n"
        f"{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )
    raw = response["message"]["content"]
    data = _extract_json_block(raw)
    return {
        "raw": raw,
        "action": str(data.get("action", "review")),
        "confidence": float(data.get("confidence", 0.0) or 0.0),
        "reason": str(data.get("reason", "")).strip(),
    }


def _merge_points(
    left: DraftPointEvent,
    right: DraftPointEvent,
) -> DraftPointEvent:
    merged_flags = sorted(
        set(left.flags + right.flags + ["qwen3_vl_boundary_merge", "qwen3_reason_merge"])
    )
    merged_conf = max(float(left.confidence), float(right.confidence))
    return DraftPointEvent(
        id=left.id,
        t_start=float(left.t_start),
        t_end=float(right.t_end),
        winner=left.winner if left.winner == right.winner else "unknown",
        confidence=float(merged_conf),
        flags=merged_flags,
        source="ai",
        corrections=list(left.corrections),
    )


def apply_boundary_decisions(
    draft: DraftMatch,
    candidate_reports: List[Dict[str, Any]],
    *,
    min_vision_conf: float = 0.70,
    min_reason_conf: float = 0.80,
) -> DraftMatch:
    merge_left_indices = {
        int(r["left_idx"])
        for r in candidate_reports
        if r.get("vision_boundary_label") == "continuous_rally"
        and float(r.get("vision_confidence", 0.0)) >= min_vision_conf
        and r.get("reason_action") == "merge"
        and float(r.get("reason_confidence", 0.0)) >= min_reason_conf
    }

    new_points: List[DraftPointEvent] = []
    idx = 0
    points = list(draft.points)
    while idx < len(points):
        current = points[idx]
        while idx in merge_left_indices and idx + 1 < len(points):
            current = _merge_points(current, points[idx + 1])
            idx += 1
        new_points.append(current)
        idx += 1

    renumbered: List[DraftPointEvent] = []
    for i, point in enumerate(new_points, start=1):
        renumbered.append(
            DraftPointEvent(
                id=f"pt_{i:04d}",
                t_start=point.t_start,
                t_end=point.t_end,
                winner=point.winner,
                confidence=point.confidence,
                flags=list(point.flags),
                source=point.source,
                corrections=list(point.corrections),
            )
        )

    return DraftMatch(
        schema_version=draft.schema_version,
        sport=draft.sport,
        video_path=draft.video_path,
        video_fps=draft.video_fps,
        best_of=draft.best_of,
        created_at=draft.created_at,
        roi=dict(draft.roi),
        points=renumbered,
        score_validation=dict(draft.score_validation),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Review rally boundaries with Qwen3-VL + Qwen3.")
    parser.add_argument("--draft", required=True, help="Input draft JSON")
    parser.add_argument("--out", required=True, help="Output reviewed draft JSON")
    parser.add_argument("--report-out", required=True, help="Output review report JSON")
    parser.add_argument("--vision-model", default="qwen3-vl:8b")
    parser.add_argument("--reason-model", default="qwen3:14b")
    parser.add_argument("--max-gap-sec", type=float, default=0.12)
    parser.add_argument("--min-vision-conf", type=float, default=0.70)
    parser.add_argument("--min-reason-conf", type=float, default=0.80)
    parser.add_argument("--image-dir", default="matches/qwen_boundary_review")
    args = parser.parse_args()

    draft = load_draft_match(Path(args.draft))
    candidates = _candidate_boundaries(draft, max_gap_sec=float(args.max_gap_sec))
    image_dir = Path(args.image_dir)

    reports: List[Dict[str, Any]] = []
    for cand in candidates:
        print(f"[Boundary] {cand.left_point.id} -> {cand.right_point.id} | gap={cand.gap_sec:.3f}s")
        image_path = build_boundary_grid(draft.video_path, cand, image_dir)
        vision = review_boundary_with_vision(
            model_name=args.vision_model,
            image_path=image_path,
            candidate=cand,
        )
        reason = review_boundary_with_reasoner(
            model_name=args.reason_model,
            candidate=cand,
            vision_result=vision,
        )
        report = {
            "left_idx": cand.left_idx,
            "right_idx": cand.right_idx,
            "left_point_id": cand.left_point.id,
            "right_point_id": cand.right_point.id,
            "left_t_start": cand.left_point.t_start,
            "left_t_end": cand.left_point.t_end,
            "right_t_start": cand.right_point.t_start,
            "right_t_end": cand.right_point.t_end,
            "gap_sec": cand.gap_sec,
            "image_path": str(image_path),
            "vision_boundary_label": vision["boundary_label"],
            "vision_confidence": vision["confidence"],
            "vision_reason": vision["reason"],
            "reason_action": reason["action"],
            "reason_confidence": reason["confidence"],
            "reason_reason": reason["reason"],
        }
        reports.append(report)
        print(
            f"  vision={vision['boundary_label']} ({vision['confidence']:.2f}) | "
            f"reason={reason['action']} ({reason['confidence']:.2f})"
        )

    reviewed = apply_boundary_decisions(
        draft,
        reports,
        min_vision_conf=float(args.min_vision_conf),
        min_reason_conf=float(args.min_reason_conf),
    )
    save_draft_match(Path(args.out), reviewed)

    report_payload = {
        "input_draft": str(Path(args.draft).resolve()),
        "output_draft": str(Path(args.out).resolve()),
        "vision_model": args.vision_model,
        "reason_model": args.reason_model,
        "input_count": len(draft.points),
        "output_count": len(reviewed.points),
        "candidate_count": len(reports),
        "candidates": reports,
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] Reviewed draft saved: {args.out} | points={len(reviewed.points)}")
    print(f"[OK] Review report saved: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
