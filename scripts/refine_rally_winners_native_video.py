from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.append(str(Path(__file__).parent.parent))

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


def _build_roi_clip(
    *,
    source_clip: Path,
    roi_clip: Path,
    roi: dict,
    margin_ratio: float,
    margin_y_ratio: float,
) -> None:
    video_w, video_h = _video_dimensions(source_clip)
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
                f"Did Player A (near side) win this {'full table-tennis rally' if clip_scope == 'full' else 'table-tennis rally segment'}? "
                "Answer with Yes or No, then one short reason based on the last successful shot and failed return."
            ),
        )
        raw_text_b = ask_text_for_video(
            main_pass_clip,
            (
                f"Did Player B (far side) win this {'full table-tennis rally' if clip_scope == 'full' else 'table-tennis rally segment'}? "
                "Answer with Yes or No, then one short reason based on the last successful shot and failed return."
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
                composite_clip,
                (
                    f"This video shows one {'full table-tennis rally' if clip_scope == 'full' else 'table-tennis rally segment'}. "
                    "The LEFT half is the original full frame. The RIGHT half is a zoom around the table and players. "
                    "Player A is the near-side player. Player B is the far-side player. "
                    "Decide the winner from the final successful shot and the failed return. "
                    "Do not prefer the near-side player by default. "
                    "Answer with one short sentence: Winner=player_a or Winner=player_b, then a brief reason."
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
        if flip_clip is not None and flip_clip.exists():
            flip_clip.unlink()
        if composite_clip is not None and composite_clip.exists():
            composite_clip.unlink()

        final_clip = _rename_clip_with_winner(
            temp_clip,
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
