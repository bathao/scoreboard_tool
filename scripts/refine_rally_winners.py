from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_ollama_client import OllamaVisionClient
from backend.rally_timeline_contract import Correction, RallyTimelinePoint, load_rally_timeline, save_rally_timeline
from backend.score_validation import build_score_validation


def _frozen_boundary_signature(points: Iterable[RallyTimelinePoint]) -> list[tuple[str, float, float]]:
    return [(str(point.id), float(point.t_start), float(point.t_end)) for point in points]


def _winner_window(
    point: RallyTimelinePoint,
    *,
    ratio: float,
    min_window_sec: float,
    max_window_sec: float,
) -> tuple[float, float]:
    rally_start = float(point.t_start)
    rally_end = float(max(point.t_end, point.t_start + 0.01))
    duration = rally_end - rally_start
    desired = max(min_window_sec, min(max_window_sec, duration * ratio))
    window_start = max(rally_start, rally_end - desired)
    return float(window_start), float(rally_end)


def _read_frame_at(cap: cv2.VideoCapture, t_sec: float) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t_sec)) * 1000.0)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def _annotate_frame(frame: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    cv2.putText(
        out,
        label,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _winner_crop_box(
    frame_shape: tuple[int, int, int],
    roi: dict[str, int] | None,
) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    if not roi:
        return 0, 0, frame_w, frame_h

    x = int(roi.get("x", 0))
    y = int(roi.get("y", 0))
    w = int(roi.get("w", 0))
    h = int(roi.get("h", 0))
    if w <= 0 or h <= 0:
        return 0, 0, frame_w, frame_h

    x1 = max(0, int(round(x - (0.45 * w))))
    x2 = min(frame_w, int(round(x + (1.45 * w))))
    y1 = max(0, int(round(y - (1.20 * h))))
    y2 = min(frame_h, int(round(y + (3.10 * h))))
    if x2 - x1 < 64 or y2 - y1 < 64:
        return 0, 0, frame_w, frame_h
    return x1, y1, x2, y2


def build_winner_evidence_grid(
    *,
    video_path: str,
    point: RallyTimelinePoint,
    roi: dict[str, int] | None,
    out_dir: Path,
    window_ratio: float = 0.5,
    min_window_sec: float = 1.2,
    max_window_sec: float = 4.0,
    frame_count: int = 8,
    frame_size: tuple[int, int] = (480, 270),
) -> tuple[Path, List[Path], float, float, List[float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    window_start, window_end = _winner_window(
        point,
        ratio=float(window_ratio),
        min_window_sec=float(min_window_sec),
        max_window_sec=float(max_window_sec),
    )
    if frame_count < 2:
        frame_times = [window_start]
    else:
        frame_times = np.linspace(window_start, window_end, num=int(frame_count), dtype=np.float32).tolist()

    frames: List[np.ndarray] = []
    frame_paths: List[Path] = []
    kept_times: List[float] = []
    for idx, t_sec in enumerate(frame_times, start=1):
        frame = _read_frame_at(cap, float(t_sec))
        if frame is None:
            continue
        full_frame = cv2.resize(frame, frame_size)
        x1, y1, x2, y2 = _winner_crop_box(frame.shape, roi)
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, frame_size)
        out_dir.mkdir(parents=True, exist_ok=True)
        full_path = out_dir / f"{point.id}_winner_frame_{idx:02d}_full.jpg"
        crop_path = out_dir / f"{point.id}_winner_frame_{idx:02d}_crop.jpg"
        cv2.imwrite(str(full_path), full_frame)
        cv2.imwrite(str(crop_path), crop_frame)
        frame_paths.extend([full_path, crop_path])
        frames.append(_annotate_frame(full_frame, f"T{idx} full"))
        frames.append(_annotate_frame(crop_frame, f"T{idx} crop"))
        kept_times.append(float(t_sec))
    cap.release()

    if len(frames) < 8:
        raise RuntimeError(f"Unable to build enough winner evidence frames for {point.id}")

    cols = 4
    rows = int(math.ceil(len(frames) / cols))
    blank = np.zeros_like(frames[0])
    padded = frames + [blank] * (rows * cols - len(frames))
    row_images = []
    for row_idx in range(rows):
        row = cv2.hconcat(padded[row_idx * cols : (row_idx + 1) * cols])
        row_images.append(row)
    grid = cv2.vconcat(row_images)
    footer = f"{point.id} | for each time step: full then crop | earlier -> later | near=player_a | far=player_b"
    cv2.putText(
        grid,
        footer,
        (18, grid.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_path = out_dir / f"{point.id}_winner_evidence_grid.jpg"
    cv2.imwrite(str(out_path), grid)
    return out_path, frame_paths, window_start, window_end, kept_times


def _maybe_process_point(point: RallyTimelinePoint, *, only_unknown_final: bool) -> bool:
    if only_unknown_final and point.winner != "unknown":
        return False
    return True


def _selected_point_ids(raw_values: Iterable[str]) -> set[str]:
    selected: set[str] = set()
    for raw in raw_values:
        for item in str(raw).split(","):
            item = item.strip()
            if item:
                selected.add(item)
    return selected


def _update_point_from_prediction(
    point: RallyTimelinePoint,
    *,
    prediction,
    image_path: Path,
) -> None:
    point.winner_candidate = prediction.winner
    point.winner_confidence = float(prediction.confidence)
    point.winner_decision = prediction.decision
    point.winner_reason = prediction.reason or None
    point.winner_model = prediction.model
    point.winner_score_a = float(prediction.score_a)
    point.winner_score_b = float(prediction.score_b)
    if prediction.decision == "auto" and prediction.winner in {"player_a", "player_b"}:
        point.winner = prediction.winner
    else:
        point.winner = "unknown"
    point.source = "ai"

    model_flag = f"winner_model_{prediction.model.replace(':', '_').replace('-', '_')}"
    for flag in ["winner_local_vlm", model_flag]:
        if flag not in point.flags:
            point.flags.append(flag)
    if "winner_evidence_grid" not in point.flags:
        point.flags.append("winner_evidence_grid")
    point.flags = sorted(set(point.flags))

    correction_note = {
        "winner_candidate": prediction.winner,
        "winner_confidence": prediction.confidence,
        "winner_decision": prediction.decision,
        "winner_reason": prediction.reason,
        "winner_model": prediction.model,
        "winner_score_a": prediction.score_a,
        "winner_score_b": prediction.score_b,
        "winner_evidence_grid": str(image_path),
    }
    point.corrections.append(
        Correction(
            at="",
            by="local_vlm",
            changes={"winner_vlm": correction_note},
            note="local VLM winner inference",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Refine rally winners using a local VLM evidence-grid workflow.")
    parser.add_argument("--timeline", required=True, help="Path to input rally timeline JSON")
    parser.add_argument("--out", default=None, help="Path to output refined timeline JSON")
    parser.add_argument("--model", default="qwen3-vl:8b", help="Local Ollama vision model name")
    parser.add_argument("--image-dir", default="matches/winner_evidence_local_vlm", help="Directory for exported evidence grids")
    parser.add_argument("--window-ratio", type=float, default=(2.0 / 3.0), help="Use this fraction of the rally from the end backward")
    parser.add_argument("--min-window-sec", type=float, default=1.2, help="Minimum winner window length")
    parser.add_argument("--max-window-sec", type=float, default=4.0, help="Maximum winner window length")
    parser.add_argument("--frame-count", type=int, default=8, help="Number of ordered frame timestamps in each evidence grid")
    parser.add_argument("--frame-width", type=int, default=480, help="Width of each frame in the evidence grid")
    parser.add_argument("--frame-height", type=int, default=270, help="Height of each frame in the evidence grid")
    parser.add_argument("--only-unknown-final", action="store_true", help="Only run local VLM on rallies whose final winner is still unknown")
    parser.add_argument("--point-ids", nargs="*", default=[], help="Optional point ids to process, e.g. pt_0001 pt_0002")
    parser.add_argument("--max-points", type=int, default=None, help="Optional cap on how many rallies to process in this run")
    parser.add_argument("--resume-if-exists", action="store_true", help="If --out already exists, resume from it instead of reloading --timeline")
    parser.add_argument(
        "--expected-scope",
        choices=["any", "set", "match"],
        default="any",
        help="Expected clip scope for score-rule validation",
    )
    parser.add_argument(
        "--expected-final-set-score",
        default=None,
        help="Expected final set score in A-B format, e.g. 11-3",
    )
    args = parser.parse_args()

    timeline_path = Path(args.timeline)
    output_path = Path(args.out) if args.out else timeline_path
    image_dir = Path(args.image_dir)

    print(f"--- Local VLM Winner Refinement ({args.model}) ---")
    if args.resume_if_exists and output_path.exists():
        timeline = load_rally_timeline(output_path)
        print(f"Resuming from existing output: {output_path}")
    else:
        timeline = load_rally_timeline(timeline_path)
    boundary_signature_before = _frozen_boundary_signature(timeline.points)
    client = OllamaVisionClient(model_name=args.model)
    selected_ids = _selected_point_ids(args.point_ids)
    processed_limit = None if args.max_points is None else max(0, int(args.max_points))

    processed_count = 0
    for point in timeline.points:
        if selected_ids and point.id not in selected_ids:
            continue
        if processed_limit is not None and processed_count >= processed_limit:
            break
        if not _maybe_process_point(point, only_unknown_final=bool(args.only_unknown_final)):
            continue
        print(f"Analyzing {point.id} ({point.t_start:.3f}s - {point.t_end:.3f}s)...")
        image_path, frame_paths, window_start, window_end, frame_times = build_winner_evidence_grid(
            video_path=timeline.video_path,
            point=point,
            roi=timeline.roi,
            out_dir=image_dir,
            window_ratio=float(args.window_ratio),
            min_window_sec=float(args.min_window_sec),
            max_window_sec=float(args.max_window_sec),
            frame_count=int(args.frame_count),
            frame_size=(max(160, int(args.frame_width)), max(90, int(args.frame_height))),
        )
        evidence_paths: List[Path | str]
        if args.model.startswith("qwen3-vl"):
            evidence_paths = frame_paths
        else:
            evidence_paths = [image_path]
        prediction = client.predict_winner_structured(evidence_paths)
        _update_point_from_prediction(point, prediction=prediction, image_path=image_path)
        print(
            f"   > {prediction.winner} | decision={prediction.decision} | "
            f"confidence={prediction.confidence:.3f} | reason={prediction.reason!r} | "
            f"window={window_start:.2f}->{window_end:.2f} | frames={len(frame_times)}"
        )
        processed_count += 1
        timeline.analysis_metadata["winner_inference_mode"] = "local_vlm_qwen3_vl_8b_ordered_multiframe_v2"
        timeline.analysis_metadata["winner_vlm_model"] = args.model
        timeline.analysis_metadata["winner_vlm_window_ratio"] = float(args.window_ratio)
        timeline.analysis_metadata["winner_vlm_min_window_sec"] = float(args.min_window_sec)
        timeline.analysis_metadata["winner_vlm_max_window_sec"] = float(args.max_window_sec)
        timeline.analysis_metadata["winner_vlm_frame_count"] = int(args.frame_count)
        timeline.analysis_metadata["winner_vlm_frame_width"] = max(160, int(args.frame_width))
        timeline.analysis_metadata["winner_vlm_frame_height"] = max(90, int(args.frame_height))
        timeline.analysis_metadata["winner_vlm_image_dir"] = str(image_dir)
        save_rally_timeline(output_path, timeline)

    timeline.analysis_metadata["winner_inference_mode"] = "local_vlm_qwen3_vl_8b_ordered_multiframe_v2"
    timeline.analysis_metadata["winner_vlm_model"] = args.model
    timeline.analysis_metadata["winner_vlm_window_ratio"] = float(args.window_ratio)
    timeline.analysis_metadata["winner_vlm_min_window_sec"] = float(args.min_window_sec)
    timeline.analysis_metadata["winner_vlm_max_window_sec"] = float(args.max_window_sec)
    timeline.analysis_metadata["winner_vlm_frame_count"] = int(args.frame_count)
    timeline.analysis_metadata["winner_vlm_frame_width"] = max(160, int(args.frame_width))
    timeline.analysis_metadata["winner_vlm_frame_height"] = max(90, int(args.frame_height))
    timeline.analysis_metadata["winner_vlm_image_dir"] = str(image_dir)

    timeline.score_validation = build_score_validation(
        timeline,
        expected_scope=args.expected_scope,
        expected_final_set_score=args.expected_final_set_score,
    )
    boundary_signature_after = _frozen_boundary_signature(timeline.points)
    if boundary_signature_after != boundary_signature_before:
        raise RuntimeError("Winner phase must not modify frozen rally boundaries (id/t_start/t_end).")
    save_rally_timeline(output_path, timeline)
    print(f"Validation: status={timeline.score_validation.get('status')} | inferred={timeline.score_validation.get('inferred_scoreline')}")
    for msg in timeline.score_validation.get("issues", []):
        print(f"  - {msg}")
    print(f"\n--- DONE ---")
    print(f"Processed {processed_count} rallies. Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
