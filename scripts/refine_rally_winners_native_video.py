from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_ball_tracking import _extract_ball_candidates, _get_ball_tracking_profile, _pick_best_candidate
from backend.rally_timeline_contract import Correction, RallyTimelinePoint, load_rally_timeline, save_rally_timeline


def _frozen_boundary_signature(points: Iterable[RallyTimelinePoint]) -> list[tuple[str, float, float]]:
    return [(str(point.id), float(point.t_start), float(point.t_end)) for point in points]


def _winner_window(
    point: RallyTimelinePoint,
    *,
    ratio: float,
    full_rally_threshold_sec: float,
    min_window_sec: float,
    max_window_sec: float,
) -> tuple[float, float]:
    rally_start = float(point.t_start)
    rally_end = float(max(point.t_end, point.t_start + 0.01))
    duration = rally_end - rally_start
    if duration <= full_rally_threshold_sec:
        window_start = rally_start
    else:
        desired = max(min_window_sec, duration * ratio)
        if max_window_sec > 0:
            desired = min(max_window_sec, desired)
        window_start = max(rally_start, rally_end - desired)
    return float(window_start), float(rally_end)


def _selected_point_ids(raw_values: Iterable[str]) -> set[str]:
    selected: set[str] = set()
    for raw in raw_values:
        for item in str(raw).split(","):
            item = item.strip()
            if item:
                selected.add(item)
    return selected


def _extract_winner_label(text: str) -> str:
    raw = str(text or "").strip().lower()
    match = re.search(r"\b(player_a|player_b)\b", raw)
    if match:
        return match.group(1)
    if "near" in raw and "far" not in raw:
        return "player_a"
    if "far" in raw and "near" not in raw:
        return "player_b"
    return "unknown"


def _extract_field_label(text: str, field_name: str) -> str:
    raw = str(text or "").strip().lower()
    match = re.search(rf"\b{re.escape(field_name.lower())}\s*[:=]\s*(player_a|player_b)\b", raw)
    if match:
        return match.group(1)
    return "unknown"


def _extract_comparative_winner(text: str) -> str:
    winner = _extract_field_label(text, "winner")
    if winner in {"player_a", "player_b"}:
        return winner
    loser = _extract_field_label(text, "loser")
    if loser == "player_a":
        return "player_b"
    if loser == "player_b":
        return "player_a"
    return _extract_winner_label(text)


def _extract_yes_no(text: str) -> str:
    raw = str(text or "").strip().lower()
    if re.search(r"\byes\b", raw):
        return "yes"
    if re.search(r"\bno\b", raw):
        return "no"
    return "unknown"


def _slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(model_name).strip().lower())
    slug = slug.strip("_")
    return slug or "native_video_model"


def _clip_window_video(
    *,
    source_video: str,
    clip_path: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        source_video,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        str(clip_path),
    ]
    subprocess.run(cmd, check=True)


def _video_dimensions(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    raw = result.stdout.strip()
    width_str, height_str = raw.split("x", 1)
    return int(width_str), int(height_str)


def _build_composite_clip(
    *,
    source_clip: Path,
    composite_clip: Path,
    roi: dict,
) -> None:
    video_w, video_h = _video_dimensions(source_clip)
    roi_x = int(roi.get("x", 0))
    roi_y = int(roi.get("y", 0))
    roi_w = int(roi.get("w", max(1, video_w // 4)))
    roi_h = int(roi.get("h", max(1, video_h // 8)))
    crop_w = min(video_w, max(1400, int(round(roi_w * 2.0))))
    crop_h = min(video_h, max(900, int(round(roi_h * 4.0))))
    center_x = roi_x + roi_w / 2.0
    center_y = roi_y + roi_h / 2.0
    crop_x = max(0, min(video_w - crop_w, int(round(center_x - crop_w / 2.0))))
    crop_y = max(0, min(video_h - crop_h, int(round(center_y - crop_h / 2.0))))

    composite_clip.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_clip),
        "-filter_complex",
        (
            f"[0:v]scale=960:540[left];"
            f"[0:v]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale=960:540[right];"
            f"[left][right]hstack=inputs=2[v]"
        ),
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        str(composite_clip),
    ]
    subprocess.run(cmd, check=True)


def _compute_roi_crop_bounds(
    *,
    video_w: int,
    video_h: int,
    roi: dict,
    margin_ratio: float,
    margin_y_ratio: float,
) -> tuple[int, int, int, int]:
    roi_x = int(roi.get("x", 0))
    roi_y = int(roi.get("y", 0))
    roi_w = int(roi.get("w", max(1, video_w // 4)))
    roi_h = int(roi.get("h", max(1, video_h // 8)))
    expand_w = int(round(roi_w * float(margin_ratio)))
    expand_h = int(round(roi_h * float(margin_y_ratio)))
    crop_x = max(0, roi_x - expand_w)
    crop_y = max(0, roi_y - expand_h)
    crop_w = min(video_w - crop_x, roi_w + (2 * expand_w))
    crop_h = min(video_h - crop_y, roi_h + (2 * expand_h))
    return crop_x, crop_y, crop_w, crop_h


def _build_roi_clip(
    *,
    source_clip: Path,
    roi_clip: Path,
    roi: dict,
    margin_ratio: float,
    margin_y_ratio: float,
) -> None:
    video_w, video_h = _video_dimensions(source_clip)
    crop_x, crop_y, crop_w, crop_h = _compute_roi_crop_bounds(
        video_w=video_w,
        video_h=video_h,
        roi=roi,
        margin_ratio=margin_ratio,
        margin_y_ratio=margin_y_ratio,
    )

    roi_clip.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_clip),
        "-filter:v",
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        str(roi_clip),
    ]
    subprocess.run(cmd, check=True)


def _draw_table_overlay(frame: np.ndarray, table_xywh: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = table_xywh
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (60, 180, 60), thickness=-1)
    blended = cv2.addWeighted(overlay, 0.14, frame, 0.86, 0.0)
    cv2.rectangle(blended, (x, y), (x + w, y + h), (70, 255, 70), thickness=3)
    return blended


def _draw_ball_trail(
    frame: np.ndarray,
    trail_points: list[tuple[int, int] | None],
) -> np.ndarray:
    recent = [pt for pt in trail_points if pt is not None]
    if not recent:
        return frame
    for idx, pt in enumerate(recent):
        alpha = float(idx + 1) / float(max(1, len(recent)))
        color = (0, int(40 + (160 * alpha)), 255)
        radius = 3 if idx < len(recent) - 1 else 5
        cv2.circle(frame, pt, radius, color, thickness=-1)
        if idx > 0:
            prev_pt = recent[idx - 1]
            if float(np.linalg.norm(np.asarray(pt, dtype=np.float32) - np.asarray(prev_pt, dtype=np.float32))) <= 120.0:
                cv2.line(frame, prev_pt, pt, color, thickness=2)
    return frame


def _in_tight_trail_zone(point_xy: tuple[int, int], table_xywh: tuple[int, int, int, int]) -> bool:
    x, y = point_xy
    tx, ty, tw, th = table_xywh
    zone_x1 = tx - int(round(tw * 0.08))
    zone_x2 = tx + tw + int(round(tw * 0.08))
    zone_y1 = ty - int(round(th * 0.10))
    zone_y2 = ty + th + int(round(th * 0.38))
    return zone_x1 <= x <= zone_x2 and zone_y1 <= y <= zone_y2


def _candidate_visual_ball_score(
    cropped_frame: np.ndarray,
    *,
    candidate_local: np.ndarray,
    search_x: int,
    search_y: int,
    table_xywh: tuple[int, int, int, int],
) -> tuple[tuple[int, int], float]:
    cx = int(round(search_x + float(candidate_local[0])))
    cy = int(round(search_y + float(candidate_local[1])))
    h, w = cropped_frame.shape[:2]
    px1 = max(0, cx - 4)
    py1 = max(0, cy - 4)
    px2 = min(w, cx + 5)
    py2 = min(h, cy + 5)
    patch = cropped_frame[py1:py2, px1:px2]
    if patch.size == 0:
        return (cx, cy), 0.0

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[..., 1].mean()) / 255.0
    mean_val = float(hsv[..., 2].mean()) / 255.0
    whiteness = max(0.0, 1.0 - (mean_sat * 1.55))
    brightness = mean_val

    tx, ty, tw, th = table_xywh
    zone_x1 = tx - int(round(tw * 0.20))
    zone_x2 = tx + tw + int(round(tw * 0.20))
    zone_y1 = ty - int(round(th * 0.28))
    zone_y2 = ty + th + int(round(th * 0.62))
    in_zone = zone_x1 <= cx <= zone_x2 and zone_y1 <= cy <= zone_y2
    proximity = 1.0 if in_zone else 0.05
    if cy < (ty - int(round(th * 0.45))):
        proximity *= 0.25
    if cy > (ty + th + int(round(th * 0.95))):
        proximity *= 0.55

    visual_score = (0.42 * whiteness) + (0.38 * brightness) + (0.20 * proximity)
    if mean_sat > 0.42 and mean_val < 0.86:
        visual_score *= 0.22
    return (cx, cy), float(visual_score)


def _build_augmented_v1_clip(
    *,
    source_clip: Path,
    augmented_clip: Path,
    roi: dict,
    margin_ratio: float,
    margin_y_ratio: float,
    trail_length: int,
    ball_profile: str,
) -> None:
    video_w, video_h = _video_dimensions(source_clip)
    crop_x, crop_y, crop_w, crop_h = _compute_roi_crop_bounds(
        video_w=video_w,
        video_h=video_h,
        roi=roi,
        margin_ratio=margin_ratio,
        margin_y_ratio=margin_y_ratio,
    )
    roi_x = int(roi.get("x", 0))
    roi_y = int(roi.get("y", 0))
    roi_w = int(roi.get("w", max(1, video_w // 4)))
    roi_h = int(roi.get("h", max(1, video_h // 8)))
    table_xywh = (roi_x - crop_x, roi_y - crop_y, roi_w, roi_h)

    cfg = _get_ball_tracking_profile(ball_profile)
    search_x = max(0, table_xywh[0] - int(round(table_xywh[2] * cfg.pad_x_ratio)))
    search_y = max(0, table_xywh[1] - int(round(table_xywh[3] * cfg.pad_top_ratio)))
    search_x2 = min(crop_w, table_xywh[0] + table_xywh[2] + int(round(table_xywh[2] * cfg.pad_x_ratio)))
    search_y2 = min(crop_h, table_xywh[1] + table_xywh[3] + int(round(table_xywh[3] * cfg.pad_bottom_ratio)))
    search_w = max(1, search_x2 - search_x)
    search_h = max(1, search_y2 - search_y)

    cap = cv2.VideoCapture(str(source_clip))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip for augmented overlay: {source_clip}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    augmented_clip.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(augmented_clip), fourcc, fps, (crop_w, crop_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer for augmented clip: {augmented_clip}")

    prev_gray: np.ndarray | None = None
    prev_center: np.ndarray | None = None
    prev_velocity = np.zeros(2, dtype=np.float32)
    missing_count = 0
    trail_history: deque[tuple[int, int] | None] = deque(maxlen=max(4, int(trail_length)))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cropped = frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w].copy()
            gray = cv2.cvtColor(cropped[search_y : search_y + search_h, search_x : search_x + search_w], cv2.COLOR_BGR2GRAY)
            chosen_xy: tuple[int, int] | None = None
            if prev_gray is not None:
                diff_gray = cv2.absdiff(gray, prev_gray)
                raw_candidates = _extract_ball_candidates(diff_gray)
                candidates: list[tuple[np.ndarray, float]] = []
                for cand_center, cand_score in raw_candidates:
                    _xy, visual_score = _candidate_visual_ball_score(
                        cropped,
                        candidate_local=cand_center,
                        search_x=search_x,
                        search_y=search_y,
                        table_xywh=table_xywh,
                    )
                    if visual_score < 0.34:
                        continue
                    combined_score = (0.60 * float(cand_score)) + (0.40 * visual_score)
                    candidates.append((cand_center, float(combined_score)))
                candidates.sort(key=lambda item: item[1], reverse=True)
                chosen_center: np.ndarray | None = None
                chosen_score = 0.0
                if candidates:
                    if prev_center is None:
                        chosen_center, chosen_score = _pick_best_candidate(
                            candidates,
                            min_score=cfg.min_start_score,
                        )
                    else:
                        predicted = prev_center + prev_velocity
                        max_jump = cfg.max_jump_px + (cfg.max_jump_missing_gain * float(missing_count))
                        chosen_center, chosen_score = _pick_best_candidate(
                            candidates,
                            min_score=cfg.min_continue_score,
                            predicted_center=predicted,
                            max_jump_px=max_jump,
                        )
                        if chosen_center is None and cfg.allow_top_fallback:
                            chosen_center, chosen_score = _pick_best_candidate(
                                candidates,
                                min_score=max(cfg.min_start_score, cfg.min_continue_score),
                            )
                if chosen_center is not None:
                    if prev_center is not None:
                        delta = chosen_center - prev_center
                        speed = float(np.linalg.norm(delta))
                        if speed >= cfg.min_continue_motion_px or chosen_score >= cfg.strong_score:
                            prev_velocity = (0.45 * prev_velocity) + (0.55 * delta)
                            prev_center = chosen_center
                            missing_count = 0
                            raw_xy = (
                                int(round(search_x + float(chosen_center[0]))),
                                int(round(search_y + float(chosen_center[1]))),
                            )
                            chosen_xy = raw_xy if _in_tight_trail_zone(raw_xy, table_xywh) else None
                        else:
                            missing_count += 1
                    else:
                        prev_center = chosen_center
                        prev_velocity = np.zeros(2, dtype=np.float32)
                        missing_count = 0
                        raw_xy = (
                            int(round(search_x + float(chosen_center[0]))),
                            int(round(search_y + float(chosen_center[1]))),
                        )
                        chosen_xy = raw_xy if _in_tight_trail_zone(raw_xy, table_xywh) else None
                else:
                    missing_count += 1
                    if missing_count > cfg.hold_misses:
                        prev_center = None
                        prev_velocity = np.zeros(2, dtype=np.float32)
            prev_gray = gray
            trail_history.append(chosen_xy)
            augmented = _draw_table_overlay(cropped, table_xywh)
            augmented = _draw_ball_trail(augmented, list(trail_history))
            writer.write(augmented)
    finally:
        writer.release()
        cap.release()


def _build_flipped_clip(
    *,
    source_clip: Path,
    flipped_clip: Path,
) -> None:
    flipped_clip.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_clip),
        "-filter:v",
        "hflip",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        str(flipped_clip),
    ]
    subprocess.run(cmd, check=True)


def _rename_clip_with_winner(
    clip_path: Path,
    *,
    point_id: str,
    winner: str,
    model_slug: str,
) -> Path:
    side = "near" if winner == "player_a" else "far" if winner == "player_b" else "unknown"
    final_name = f"{point_id}__pick_{side}__native_video_{model_slug}.mp4"
    final_path = clip_path.with_name(final_name)
    if final_path.exists():
        final_path.unlink()
    clip_path.rename(final_path)
    return final_path


def _update_point(
    point: RallyTimelinePoint,
    *,
    winner: str,
    raw_text: str,
    model_name: str,
    model_slug: str,
    clip_path: Path,
    score_a: float,
    score_b: float,
) -> None:
    point.winner_candidate = winner  # type: ignore[assignment]
    point.winner_confidence = max(score_a, score_b) if winner in {"player_a", "player_b"} else 0.0
    point.winner_decision = "review" if winner in {"player_a", "player_b"} else "blocked"
    point.winner_reason = raw_text[:160].strip() or None
    point.winner_model = model_name
    point.winner_score_a = float(score_a)
    point.winner_score_b = float(score_b)
    point.winner = "unknown"
    point.source = "ai"
    point.flags = sorted(
        set(
            point.flags
            + [
                "winner_native_video",
                f"winner_model_{model_slug}_transformers",
                "winner_pairwise_yes_no",
                "winner_dense_video_config",
            ]
        )
    )
    point.corrections.append(
        Correction(
            at="",
            by="local_vlm_native_video",
            changes={
                "winner_native_video": {
                    "winner_candidate": winner,
                    "winner_confidence": point.winner_confidence,
                    "winner_decision": point.winner_decision,
                    "winner_reason": point.winner_reason,
                    "winner_model": model_name,
                    "winner_score_a": point.winner_score_a,
                    "winner_score_b": point.winner_score_b,
                    "winner_clip": str(clip_path),
                }
            },
            note=f"native-video {model_name} winner inference",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Refine rally winners using native-video Transformers.")
    parser.add_argument("--timeline", required=True, help="Path to input rally timeline JSON")
    parser.add_argument("--out", required=True, help="Path to output JSON")
    parser.add_argument("--model-dir", default="models/Qwen3-VL-4B-Instruct", help="Local HF model directory")
    parser.add_argument("--clip-dir", required=True, help="Directory for exported review clips")
    parser.add_argument("--window-ratio", type=float, default=(2.0 / 3.0), help="Use this fraction of the rally from the end backward")
    parser.add_argument(
        "--full-rally-threshold-sec",
        type=float,
        default=4.0,
        help="If rally duration is at or below this threshold, keep the full rally instead of cutting to the ratio window",
    )
    parser.add_argument("--min-window-sec", type=float, default=1.2, help="Minimum winner window length")
    parser.add_argument(
        "--max-window-sec",
        type=float,
        default=0.0,
        help="Maximum winner window length; use 0 to disable the cap and keep the full ratio-derived window",
    )
    parser.add_argument("--fps-sample", type=float, default=4.0, help="Native-video sampling fps")
    parser.add_argument("--min-frames", type=int, default=12, help="Minimum sampled frames")
    parser.add_argument("--max-frames", type=int, default=16, help="Maximum sampled frames")
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=(1280 * 720),
        help="Maximum visual pixel budget for the sampled video tokens; use 0 to disable",
    )
    parser.add_argument("--size-shortest-edge", type=int, default=576, help="Video processor shortest edge")
    parser.add_argument("--size-longest-edge", type=int, default=1048576, help="Video processor longest edge")
    parser.add_argument("--point-ids", nargs="*", default=[], help="Optional point ids to process")
    parser.add_argument("--max-points", type=int, default=None, help="Optional cap on processed rallies")
    parser.add_argument(
        "--main-pass-view",
        choices=["full", "roi"],
        default="roi",
        help="Video view used for the main A/B prompt pass",
    )
    parser.add_argument(
        "--roi-margin-ratio",
        type=float,
        default=0.4,
        help="Extra margin around table ROI when main-pass-view=roi",
    )
    parser.add_argument(
        "--roi-margin-y-ratio",
        type=float,
        default=0.9,
        help="Optional extra vertical margin ratio around table ROI; negative means reuse roi-margin-ratio",
    )
    parser.add_argument(
        "--flip-main-pass",
        action="store_true",
        help="Horizontally flip the main-pass clip before asking A/B winner prompts",
    )
    parser.add_argument(
        "--main-pass-overlay",
        choices=["none", "augmented_v1"],
        default="none",
        help="Optional overlay mode applied to the main-pass clip before inference",
    )
    parser.add_argument(
        "--aug-ball-profile",
        choices=["support", "standalone"],
        default="support",
        help="Ball-tracking profile used for augmented_v1 ball trail overlay",
    )
    parser.add_argument(
        "--aug-ball-trail-length",
        type=int,
        default=18,
        help="Maximum recent trail points rendered for augmented_v1",
    )
    args = parser.parse_args()

    timeline = load_rally_timeline(Path(args.timeline))
    boundary_signature_before = _frozen_boundary_signature(timeline.points)
    output_path = Path(args.out)
    clip_dir = Path(args.clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(args.model_dir).name
    model_slug = _slugify_model_name(model_name)

    print(f"--- Native-video Winner Refinement ({args.model_dir}) ---")
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model loaded on {model.device}")

    selected_ids = _selected_point_ids(args.point_ids)
    processed = 0
    csv_rows: list[dict[str, object]] = []

    def ask_text_for_video(video_path: Path, prompt_text: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path.resolve())},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processor_kwargs = {
            "text": [text],
            "videos": [str(video_path.resolve())],
            "return_tensors": "pt",
            "fps": float(args.fps_sample),
            "min_frames": int(args.min_frames),
            "max_frames": int(args.max_frames),
            "size": {
                "shortest_edge": int(args.size_shortest_edge),
                "longest_edge": int(args.size_longest_edge),
            },
        }
        if int(args.max_pixels) > 0:
            processor_kwargs["max_pixels"] = int(args.max_pixels)
        inputs = processor(**processor_kwargs)
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return str(output_text[0]).strip() if output_text else ""

    for point in timeline.points:
        if selected_ids and point.id not in selected_ids:
            continue
        if args.max_points is not None and processed >= max(0, int(args.max_points)):
            break

        window_start, window_end = _winner_window(
            point,
            ratio=float(args.window_ratio),
            full_rally_threshold_sec=float(args.full_rally_threshold_sec),
            min_window_sec=float(args.min_window_sec),
            max_window_sec=float(args.max_window_sec),
        )
        temp_clip = clip_dir / f"{point.id}__native_window.mp4"
        _clip_window_video(
            source_video=timeline.video_path,
            clip_path=temp_clip,
            start_sec=window_start,
            end_sec=window_end,
        )
        main_pass_clip = temp_clip
        roi_clip: Path | None = None
        flip_clip: Path | None = None
        augmented_clip: Path | None = None
        if str(args.main_pass_view) == "roi":
            roi_clip = clip_dir / f"{point.id}__native_roi.mp4"
            _build_roi_clip(
                source_clip=temp_clip,
                roi_clip=roi_clip,
                roi=timeline.roi,
                margin_ratio=float(args.roi_margin_ratio),
                margin_y_ratio=(float(args.roi_margin_ratio) if float(args.roi_margin_y_ratio) < 0 else float(args.roi_margin_y_ratio)),
            )
            main_pass_clip = roi_clip
        if str(args.main_pass_overlay) == "augmented_v1":
            augmented_clip = clip_dir / f"{point.id}__native_augv1.mp4"
            _build_augmented_v1_clip(
                source_clip=temp_clip,
                augmented_clip=augmented_clip,
                roi=timeline.roi,
                margin_ratio=float(args.roi_margin_ratio),
                margin_y_ratio=(float(args.roi_margin_ratio) if float(args.roi_margin_y_ratio) < 0 else float(args.roi_margin_y_ratio)),
                trail_length=int(args.aug_ball_trail_length),
                ball_profile=str(args.aug_ball_profile),
            )
            main_pass_clip = augmented_clip
        if bool(args.flip_main_pass):
            flip_clip = clip_dir / f"{point.id}__native_mainpass_flip.mp4"
            _build_flipped_clip(
                source_clip=main_pass_clip,
                flipped_clip=flip_clip,
            )
            main_pass_clip = flip_clip

        clip_scope = "full" if window_start <= float(point.t_start) + 1e-6 else "partial"
        raw_text_a = ask_text_for_video(
            main_pass_clip,
            (
                (
                    f"This {'full table-tennis rally' if clip_scope == 'full' else 'table-tennis rally segment'} may include visual overlays. "
                    "If present: the green box marks the table and the red trail marks recent ball motion. "
                    "Use those overlays first, and use player posture only as secondary evidence. "
                    "Did Player A (near side) win? "
                    "Answer with Yes or No, then one short reason based on the last successful shot and failed return."
                )
                if str(args.main_pass_overlay) == "augmented_v1"
                else (
                    f"Did Player A (near side) win this {'full table-tennis rally' if clip_scope == 'full' else 'table-tennis rally segment'}? "
                    "Answer with Yes or No, then one short reason based on the last successful shot and failed return."
                )
            ),
        )
        raw_text_b = ask_text_for_video(
            main_pass_clip,
            (
                (
                    f"This {'full table-tennis rally' if clip_scope == 'full' else 'table-tennis rally segment'} may include visual overlays. "
                    "If present: the green box marks the table and the red trail marks recent ball motion. "
                    "Use those overlays first, and use player posture only as secondary evidence. "
                    "Did Player B (far side) win? "
                    "Answer with Yes or No, then one short reason based on the last successful shot and failed return."
                )
                if str(args.main_pass_overlay) == "augmented_v1"
                else (
                    f"Did Player B (far side) win this {'full table-tennis rally' if clip_scope == 'full' else 'table-tennis rally segment'}? "
                    "Answer with Yes or No, then one short reason based on the last successful shot and failed return."
                )
            ),
        )
        raw_text_tiebreak = ""
        composite_clip: Path | None = None
        answer_a = _extract_yes_no(raw_text_a)
        answer_b = _extract_yes_no(raw_text_b)
        score_a = 1.0 if answer_a == "yes" else 0.0
        score_b = 1.0 if answer_b == "yes" else 0.0
        if answer_a == "yes" and answer_b != "yes":
            winner = "player_a"
        elif answer_b == "yes" and answer_a != "yes":
            winner = "player_b"
        else:
            composite_clip = clip_dir / f"{point.id}__native_composite.mp4"
            _build_composite_clip(
                source_clip=temp_clip,
                composite_clip=composite_clip,
                roi=timeline.roi,
            )
            raw_text_tiebreak = ask_text_for_video(
                (main_pass_clip if str(args.main_pass_overlay) == "augmented_v1" else composite_clip),
                (
                    "This rally video may include visual overlays. "
                    "If present: the green box marks the table and the red trail marks recent ball motion. "
                    "Player A is the near-side player. Player B is the far-side player. "
                    "Decide the winner primarily from the trail ending and table box, then use player reaction only as secondary evidence. "
                    "Do not prefer the near-side player by default. "
                    "Answer with one short sentence: Winner=player_a or Winner=player_b, then a brief reason."
                    if str(args.main_pass_overlay) == "augmented_v1"
                    else (
                        f"This video shows one {'full table-tennis rally' if clip_scope == 'full' else 'table-tennis rally segment'}. "
                        "The LEFT half is the original full frame. The RIGHT half is a zoom around the table and players. "
                        "Player A is the near-side player. Player B is the far-side player. "
                        "Decide the winner from the final successful shot and the failed return. "
                        "Do not prefer the near-side player by default. "
                        "Answer with one short sentence: Winner=player_a or Winner=player_b, then a brief reason."
                    )
                ),
            )
            tiebreak_winner = _extract_winner_label(raw_text_tiebreak)
            if tiebreak_winner in {"player_a", "player_b"}:
                winner = tiebreak_winner
                score_a = 1.0 if winner == "player_a" else 0.0
                score_b = 1.0 if winner == "player_b" else 0.0
            else:
                winner = "unknown"
                score_a = 0.0
                score_b = 0.0
            if composite_clip is not None and composite_clip.exists():
                composite_clip.unlink()
        raw_text = f"A? {raw_text_a} || B? {raw_text_b}"
        if raw_text_tiebreak:
            raw_text += f" || T? {raw_text_tiebreak}"
        if roi_clip is not None and roi_clip.exists():
            roi_clip.unlink()
        if augmented_clip is not None and augmented_clip.exists() and augmented_clip != main_pass_clip:
            augmented_clip.unlink()
        if flip_clip is not None and flip_clip.exists():
            flip_clip.unlink()
        if composite_clip is not None and composite_clip.exists():
            composite_clip.unlink()

        export_source = main_pass_clip if str(args.main_pass_overlay) == "augmented_v1" else temp_clip
        if export_source != temp_clip and temp_clip.exists():
            temp_clip.unlink()
        final_clip = _rename_clip_with_winner(
            export_source,
            point_id=point.id,
            winner=winner,
            model_slug=model_slug,
        )
        _update_point(
            point,
            winner=winner,
            raw_text=raw_text,
            model_name=model_name,
            model_slug=model_slug,
            clip_path=final_clip,
            score_a=score_a,
            score_b=score_b,
        )
        csv_rows.append(
            {
                "id": point.id,
                "t_start": float(point.t_start),
                "t_end": float(point.t_end),
                "clip_start": window_start,
                "clip_end": window_end,
                "winner_candidate": winner,
                "winner_decision": point.winner_decision,
                "winner_confidence": point.winner_confidence,
                "winner_score_a": point.winner_score_a,
                "winner_score_b": point.winner_score_b,
                "raw_output": raw_text,
                "raw_output_a": raw_text_a,
                "raw_output_b": raw_text_b,
                "raw_output_tiebreak": raw_text_tiebreak,
                "file": final_clip.name,
            }
        )
        processed += 1
        print(
            f"   > {point.id}: {winner} | {point.winner_decision} | "
            f"window={window_start:.2f}->{window_end:.2f} | "
            f"raw={raw_text!r}"
        )

    timeline.analysis_metadata["winner_inference_mode"] = f"transformers_native_video_{model_slug}_roi40y90_main_v4"
    timeline.analysis_metadata["winner_native_video_model_dir"] = str(Path(args.model_dir))
    timeline.analysis_metadata["winner_vlm_window_ratio"] = float(args.window_ratio)
    timeline.analysis_metadata["winner_full_rally_threshold_sec"] = float(args.full_rally_threshold_sec)
    timeline.analysis_metadata["winner_native_video_fps_sample"] = float(args.fps_sample)
    timeline.analysis_metadata["winner_native_video_min_frames"] = int(args.min_frames)
    timeline.analysis_metadata["winner_native_video_max_frames"] = int(args.max_frames)
    timeline.analysis_metadata["winner_native_video_max_pixels"] = int(args.max_pixels)
    timeline.analysis_metadata["winner_native_video_size_shortest_edge"] = int(args.size_shortest_edge)
    timeline.analysis_metadata["winner_native_video_size_longest_edge"] = int(args.size_longest_edge)
    timeline.analysis_metadata["winner_native_video_main_pass_view"] = str(args.main_pass_view)
    timeline.analysis_metadata["winner_native_video_roi_margin_ratio"] = float(args.roi_margin_ratio)
    timeline.analysis_metadata["winner_native_video_roi_margin_y_ratio"] = (
        float(args.roi_margin_ratio) if float(args.roi_margin_y_ratio) < 0 else float(args.roi_margin_y_ratio)
    )
    timeline.analysis_metadata["winner_native_video_flip_main_pass"] = bool(args.flip_main_pass)
    timeline.analysis_metadata["winner_native_video_clip_dir"] = str(clip_dir)
    timeline.analysis_metadata["winner_native_video_main_pass_overlay"] = str(args.main_pass_overlay)
    timeline.analysis_metadata["winner_native_video_aug_ball_profile"] = str(args.aug_ball_profile)
    timeline.analysis_metadata["winner_native_video_aug_ball_trail_length"] = int(args.aug_ball_trail_length)
    boundary_signature_after = _frozen_boundary_signature(timeline.points)
    if boundary_signature_after != boundary_signature_before:
        raise RuntimeError("Winner phase must not modify frozen rally boundaries (id/t_start/t_end).")
    save_rally_timeline(output_path, timeline)

    csv_path = clip_dir / "rally_clips.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "t_start",
                "t_end",
                "clip_start",
                "clip_end",
                "winner_candidate",
                "winner_decision",
                "winner_confidence",
                "winner_score_a",
                "winner_score_b",
                "raw_output",
                "raw_output_a",
                "raw_output_b",
                "raw_output_tiebreak",
                "file",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n--- DONE ---")
    print(f"Processed {processed} rallies. Output: {output_path}")
    print(f"Clips: {clip_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
