from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import ollama

sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_contract import DraftMatch, DraftPointEvent, load_draft_match, save_draft_match
from backend.ai_multistream_rally import _smooth_and_normalize, extract_multistream_signals


@dataclass
class SplitCandidate:
    point_idx: int
    point: DraftPointEvent
    candidate_rank: int
    split_t: float
    quiet_start_t: float
    quiet_end_t: float
    quiet_duration_sec: float
    quiet_score: float
    table_score: float
    player_score: float
    ball_score: float


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


def _find_quiet_runs(mask: np.ndarray) -> List[tuple[int, int]]:
    runs: List[tuple[int, int]] = []
    run_start: Optional[int] = None
    for idx, is_quiet in enumerate(mask.tolist()):
        if is_quiet:
            if run_start is None:
                run_start = idx
            continue
        if run_start is not None:
            runs.append((run_start, idx - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(mask) - 1))
    return runs


def _pick_local_minima(
    ts: np.ndarray,
    scores: np.ndarray,
    *,
    abs_start_idx: int,
    abs_end_idx: int,
    max_candidates: int,
    min_spacing_sec: float,
    max_score: float,
    neighborhood: int = 4,
) -> List[int]:
    if abs_end_idx - abs_start_idx < (2 * neighborhood + 1):
        return []

    chosen: List[int] = []
    order = np.argsort(scores[abs_start_idx : abs_end_idx + 1])
    for local_idx in order.tolist():
        idx = abs_start_idx + int(local_idx)
        score = float(scores[idx])
        if score > float(max_score):
            continue

        left = max(abs_start_idx, idx - neighborhood)
        right = min(abs_end_idx, idx + neighborhood)
        window = scores[left : right + 1]
        if score > float(window.min()) + 1e-6:
            continue

        t = float(ts[idx])
        if any(abs(t - float(ts[c])) < float(min_spacing_sec) for c in chosen):
            continue

        chosen.append(idx)
        if len(chosen) >= int(max_candidates):
            break

    return sorted(chosen)


def extract_split_candidates(
    draft: DraftMatch,
    *,
    video_path: str,
    table_weights_path: str,
    pose_weights_path: str,
    stride: int = 2,
    min_segment_sec: float = 12.0,
    edge_guard_sec: float = 1.2,
    quiet_thresh: float = 0.12,
    min_quiet_run_sec: float = 0.70,
    max_candidates_per_point: int = 2,
    min_candidate_spacing_sec: float = 2.5,
    fallback_score_thresh: float = 0.18,
) -> List[SplitCandidate]:
    signals = extract_multistream_signals(
        video_path,
        table_weights_path,
        pose_weights_path=pose_weights_path,
        stride=int(stride),
        player_signal_source="role_tracker",
        ball_signal_source="classical",
        device="cuda",
    )

    ts = np.asarray(signals.timestamps, dtype=np.float32)
    table_norm = _smooth_and_normalize(signals.table_energies)
    player_norm = _smooth_and_normalize(signals.player_energies)
    ball_norm = _smooth_and_normalize(signals.ball_energies)
    combo = np.maximum.reduce([table_norm, player_norm * 0.90, ball_norm])

    candidates: List[SplitCandidate] = []
    for idx, point in enumerate(draft.points):
        duration = float(point.t_end - point.t_start)
        if duration < min_segment_sec:
            continue

        inner_start_t = float(point.t_start + edge_guard_sec)
        inner_end_t = float(point.t_end - edge_guard_sec)
        if inner_end_t <= inner_start_t:
            continue

        start_idx = int(np.searchsorted(ts, inner_start_t, side="left"))
        end_idx = int(np.searchsorted(ts, inner_end_t, side="right")) - 1
        start_idx = max(0, min(start_idx, len(ts) - 1))
        end_idx = max(start_idx, min(end_idx, len(ts) - 1))
        if end_idx <= start_idx:
            continue

        seg_combo = combo[start_idx : end_idx + 1]
        quiet_runs = _find_quiet_runs(seg_combo < float(quiet_thresh))
        point_candidates: List[SplitCandidate] = []

        for local_start, local_end in quiet_runs:
            abs_start = start_idx + local_start
            abs_end = start_idx + local_end
            quiet_duration = float(ts[abs_end] - ts[abs_start])
            if quiet_duration < float(min_quiet_run_sec):
                continue

            mid_idx = int((abs_start + abs_end) // 2)
            point_candidates.append(
                SplitCandidate(
                    point_idx=idx,
                    point=point,
                    candidate_rank=0,
                    split_t=float(ts[mid_idx]),
                    quiet_start_t=float(ts[abs_start]),
                    quiet_end_t=float(ts[abs_end]),
                    quiet_duration_sec=quiet_duration,
                    quiet_score=float(combo[mid_idx]),
                    table_score=float(table_norm[mid_idx]),
                    player_score=float(player_norm[mid_idx]),
                    ball_score=float(ball_norm[mid_idx]),
                )
            )

        if len(point_candidates) < int(max_candidates_per_point):
            minima_indices = _pick_local_minima(
                ts,
                combo,
                abs_start_idx=start_idx,
                abs_end_idx=end_idx,
                max_candidates=int(max_candidates_per_point),
                min_spacing_sec=float(min_candidate_spacing_sec),
                max_score=float(fallback_score_thresh),
            )
            for min_idx in minima_indices:
                min_t = float(ts[min_idx])
                if any(abs(min_t - cand.split_t) < float(min_candidate_spacing_sec) for cand in point_candidates):
                    continue
                left_idx = max(start_idx, min_idx - 1)
                right_idx = min(end_idx, min_idx + 1)
                point_candidates.append(
                    SplitCandidate(
                        point_idx=idx,
                        point=point,
                        candidate_rank=0,
                        split_t=min_t,
                        quiet_start_t=float(ts[left_idx]),
                        quiet_end_t=float(ts[right_idx]),
                        quiet_duration_sec=float(ts[right_idx] - ts[left_idx]),
                        quiet_score=float(combo[min_idx]),
                        table_score=float(table_norm[min_idx]),
                        player_score=float(player_norm[min_idx]),
                        ball_score=float(ball_norm[min_idx]),
                    )
                )

        point_candidates.sort(key=lambda cand: (cand.quiet_score, -cand.quiet_duration_sec, cand.split_t))
        point_candidates = point_candidates[: max(1, int(max_candidates_per_point))]
        for rank, candidate in enumerate(point_candidates, start=1):
            candidates.append(
                SplitCandidate(
                    point_idx=idx,
                    point=point,
                    candidate_rank=rank,
                    split_t=candidate.split_t,
                    quiet_start_t=candidate.quiet_start_t,
                    quiet_end_t=candidate.quiet_end_t,
                    quiet_duration_sec=candidate.quiet_duration_sec,
                    quiet_score=candidate.quiet_score,
                    table_score=candidate.table_score,
                    player_score=candidate.player_score,
                    ball_score=candidate.ball_score,
                )
            )

    return candidates


def _read_frame_at(cap: cv2.VideoCapture, t_sec: float) -> Optional[Any]:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t_sec)) * 1000.0)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def _annotate(frame: Any, label: str) -> Any:
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


def build_split_grid(video_path: str, candidate: SplitCandidate, out_dir: Path) -> Path:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    offsets = [-1.6, -1.0, -0.4, 0.4, 1.0, 1.6]
    frames: List[Any] = []
    for off in offsets:
        t_sec = max(0.0, float(candidate.split_t + off))
        frame = _read_frame_at(cap, t_sec)
        if frame is None:
            continue
        frame = cv2.resize(frame, (640, 360))
        frames.append(_annotate(frame, f"t={t_sec:.2f}s"))
    cap.release()

    if len(frames) != 6:
        raise RuntimeError(f"Unable to build split grid for {candidate.point.id}")

    top = cv2.hconcat(frames[:3])
    bottom = cv2.hconcat(frames[3:])
    grid = cv2.vconcat([top, bottom])
    cv2.putText(
        grid,
        f"Split check: {candidate.point.id} @ {candidate.split_t:.2f}s",
        (18, grid.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{candidate.point.id}_cand{candidate.candidate_rank}_split_{candidate.split_t:.3f}.jpg"
    cv2.imwrite(str(out_path), grid)
    return out_path


def review_split_with_vision(model_name: str, image_path: Path, candidate: SplitCandidate) -> Dict[str, Any]:
    with open(image_path, "rb") as f:
        image_data = f.read()

    prompt = (
        "You are reviewing whether one detected table-tennis rally segment should be split into two rallies. "
        "The 6 frames are chronological around the proposed split time. "
        "Decide whether there is a real dead-ball / reset / serve-preparation between the 3rd and 4th frames. "
        "Return strict JSON only with keys: "
        'split_label ("true_split" | "continuous_rally" | "uncertain"), '
        '"confidence" (0.0-1.0), '
        '"reason" (max 20 words). '
        "Be conservative: if unsure, return uncertain."
    )
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt, "images": [image_data]}],
        options={"temperature": 0.0},
    )
    raw = response["message"]["content"]
    data = _extract_json_block(raw)
    return {
        "raw": raw,
        "split_label": str(data.get("split_label", "uncertain")),
        "confidence": float(data.get("confidence", 0.0) or 0.0),
        "reason": str(data.get("reason", "")).strip(),
    }


def review_split_with_reasoner(model_name: str, candidate: SplitCandidate, vision_result: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "point": {
            "id": candidate.point.id,
            "t_start": candidate.point.t_start,
            "t_end": candidate.point.t_end,
            "duration_sec": candidate.point.t_end - candidate.point.t_start,
            "confidence": candidate.point.confidence,
            "flags": candidate.point.flags,
        },
        "split_candidate": {
            "split_t": candidate.split_t,
            "quiet_start_t": candidate.quiet_start_t,
            "quiet_end_t": candidate.quiet_end_t,
            "quiet_duration_sec": candidate.quiet_duration_sec,
            "quiet_score": candidate.quiet_score,
            "table_score": candidate.table_score,
            "player_score": candidate.player_score,
            "ball_score": candidate.ball_score,
        },
        "vision_review": {
            "split_label": vision_result.get("split_label"),
            "confidence": vision_result.get("confidence"),
            "reason": vision_result.get("reason"),
        },
    }
    prompt = (
        "You are a strict table-tennis rally reviewer. "
        "Decide whether one long detected segment should be split into two rallies at the proposed split time. "
        "Use the quiet-run signal and the vision review. "
        "Split only when both suggest a real dead-ball/reset between rallies. "
        "Return strict JSON only with keys: "
        'action ("split" | "keep" | "review"), '
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


def _split_point(point: DraftPointEvent, split_t: float) -> tuple[DraftPointEvent, DraftPointEvent]:
    left = DraftPointEvent(
        id=point.id,
        t_start=float(point.t_start),
        t_end=float(split_t),
        winner="unknown",
        confidence=float(point.confidence),
        flags=sorted(set(point.flags + ["qwen3_vl_split", "qwen3_reason_split"])),
        source="ai",
        corrections=list(point.corrections),
    )
    right = DraftPointEvent(
        id=point.id,
        t_start=float(split_t),
        t_end=float(point.t_end),
        winner="unknown",
        confidence=float(point.confidence),
        flags=sorted(set(point.flags + ["qwen3_vl_split", "qwen3_reason_split"])),
        source="ai",
        corrections=list(point.corrections),
    )
    return left, right


def apply_split_decisions(
    draft: DraftMatch,
    reports: List[Dict[str, Any]],
    *,
    min_vision_conf: float,
    min_reason_conf: float,
) -> DraftMatch:
    split_map: Dict[int, List[float]] = {}
    for r in reports:
        if not (
            r.get("vision_split_label") == "true_split"
            and float(r.get("vision_confidence", 0.0)) >= min_vision_conf
            and r.get("reason_action") == "split"
            and float(r.get("reason_confidence", 0.0)) >= min_reason_conf
        ):
            continue
        split_map.setdefault(int(r["point_idx"]), []).append(float(r["split_t"]))

    new_points: List[DraftPointEvent] = []
    for idx, point in enumerate(draft.points):
        if idx not in split_map:
            new_points.append(point)
            continue
        valid_splits = sorted(
            s for s in split_map[idx]
            if float(point.t_start) < float(s) < float(point.t_end)
        )
        if not valid_splits:
            new_points.append(point)
            continue

        current_start = float(point.t_start)
        for split_t in valid_splits:
            if split_t <= current_start:
                continue
            left, _ = _split_point(
                DraftPointEvent(
                    id=point.id,
                    t_start=current_start,
                    t_end=float(point.t_end),
                    winner=point.winner,
                    confidence=point.confidence,
                    flags=list(point.flags),
                    source=point.source,
                    corrections=list(point.corrections),
                ),
                split_t,
            )
            new_points.append(left)
            current_start = float(split_t)

        if current_start >= float(point.t_end):
            continue

        new_points.append(
            DraftPointEvent(
                id=point.id,
                t_start=current_start,
                t_end=float(point.t_end),
                winner="unknown",
                confidence=float(point.confidence),
                flags=sorted(set(point.flags + ["qwen3_vl_split", "qwen3_reason_split"])),
                source="ai",
                corrections=list(point.corrections),
            )
        )

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
    parser = argparse.ArgumentParser(description="Review possible rally splits with Qwen3-VL + Qwen3.")
    parser.add_argument("--draft", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", default="weights/yolov8x_table.pt")
    parser.add_argument("--pose-weights", default="weights/yolov8x-pose.pt")
    parser.add_argument("--vision-model", default="qwen3-vl:8b")
    parser.add_argument("--reason-model", default="qwen3:14b")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--min-segment-sec", type=float, default=12.0)
    parser.add_argument("--edge-guard-sec", type=float, default=1.2)
    parser.add_argument("--quiet-thresh", type=float, default=0.12)
    parser.add_argument("--min-quiet-run-sec", type=float, default=0.70)
    parser.add_argument("--max-candidates-per-point", type=int, default=2)
    parser.add_argument("--min-candidate-spacing-sec", type=float, default=2.5)
    parser.add_argument("--fallback-score-thresh", type=float, default=0.18)
    parser.add_argument("--min-vision-conf", type=float, default=0.78)
    parser.add_argument("--min-reason-conf", type=float, default=0.82)
    parser.add_argument("--image-dir", default="matches/qwen_split_review")
    parser.add_argument("--skip-models", action="store_true", help="Only extract split candidates and write report without calling Qwen models")
    parser.add_argument("--build-images", action="store_true", help="Still build candidate grids when --skip-models is used")
    args = parser.parse_args()

    draft = load_draft_match(Path(args.draft))
    candidates = extract_split_candidates(
        draft,
        video_path=args.video,
        table_weights_path=args.weights,
        pose_weights_path=args.pose_weights,
        stride=int(args.stride),
        min_segment_sec=float(args.min_segment_sec),
        edge_guard_sec=float(args.edge_guard_sec),
        quiet_thresh=float(args.quiet_thresh),
        min_quiet_run_sec=float(args.min_quiet_run_sec),
        max_candidates_per_point=int(args.max_candidates_per_point),
        min_candidate_spacing_sec=float(args.min_candidate_spacing_sec),
        fallback_score_thresh=float(args.fallback_score_thresh),
    )

    image_dir = Path(args.image_dir)
    reports: List[Dict[str, Any]] = []
    for candidate in candidates:
        print(
            f"[Split] {candidate.point.id} @ {candidate.split_t:.3f}s | "
            f"quiet={candidate.quiet_score:.3f} run={candidate.quiet_duration_sec:.3f}s"
        )
        image_path: Optional[Path] = None
        if not args.skip_models or args.build_images:
            image_path = build_split_grid(args.video, candidate, image_dir)

        report_row = {
            "point_idx": candidate.point_idx,
            "point_id": candidate.point.id,
            "candidate_rank": candidate.candidate_rank,
            "t_start": candidate.point.t_start,
            "t_end": candidate.point.t_end,
            "split_t": candidate.split_t,
            "quiet_start_t": candidate.quiet_start_t,
            "quiet_end_t": candidate.quiet_end_t,
            "quiet_duration_sec": candidate.quiet_duration_sec,
            "quiet_score": candidate.quiet_score,
            "table_score": candidate.table_score,
            "player_score": candidate.player_score,
            "ball_score": candidate.ball_score,
            "image_path": "" if image_path is None else str(image_path),
        }

        if args.skip_models:
            report_row.update(
                {
                    "vision_split_label": "skipped",
                    "vision_confidence": 0.0,
                    "vision_reason": "",
                    "reason_action": "skipped",
                    "reason_confidence": 0.0,
                    "reason_reason": "",
                }
            )
            reports.append(report_row)
            print("  review=models skipped")
            continue

        assert image_path is not None
        vision = review_split_with_vision(args.vision_model, image_path, candidate)
        reason = review_split_with_reasoner(args.reason_model, candidate, vision)
        report_row.update(
            {
                "vision_split_label": vision["split_label"],
                "vision_confidence": vision["confidence"],
                "vision_reason": vision["reason"],
                "reason_action": reason["action"],
                "reason_confidence": reason["confidence"],
                "reason_reason": reason["reason"],
            }
        )
        reports.append(report_row)
        print(
            f"  vision={vision['split_label']} ({vision['confidence']:.2f}) | "
            f"reason={reason['action']} ({reason['confidence']:.2f})"
        )

    if args.skip_models:
        reviewed = draft
    else:
        reviewed = apply_split_decisions(
            draft,
            reports,
            min_vision_conf=float(args.min_vision_conf),
            min_reason_conf=float(args.min_reason_conf),
        )
    save_draft_match(Path(args.out), reviewed)

    payload = {
        "input_draft": str(Path(args.draft).resolve()),
        "output_draft": str(Path(args.out).resolve()),
        "vision_model": args.vision_model,
        "reason_model": args.reason_model,
        "skip_models": bool(args.skip_models),
        "input_count": len(draft.points),
        "output_count": len(reviewed.points),
        "candidate_count": len(reports),
        "candidates": reports,
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] Reviewed draft saved: {args.out} | points={len(reviewed.points)}")
    print(f"[OK] Review report saved: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
