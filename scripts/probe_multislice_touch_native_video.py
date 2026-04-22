from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import sys

sys.path.append(str(Path(__file__).parent.parent))

from backend.rally_timeline_contract import load_rally_timeline
from scripts.refine_rally_winners_native_video import (
    _build_augmented_v2_clip,
    _build_roi_clip,
    _build_table_only_clip,
    _clip_window_video,
)


def _selected_point_ids(raw_values: list[str]) -> set[str]:
    selected: set[str] = set()
    for raw in raw_values:
        for item in str(raw).split(","):
            item = item.strip()
            if item:
                selected.add(item)
    return selected


def _slice_windows(duration_sec: float, slice_sec: float, stride_sec: float) -> list[tuple[float, float]]:
    duration_sec = max(0.01, float(duration_sec))
    slice_sec = max(0.25, float(slice_sec))
    stride_sec = max(0.10, float(stride_sec))
    if duration_sec <= slice_sec:
        return [(0.0, duration_sec)]
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_sec:
        end = min(duration_sec, start + slice_sec)
        windows.append((start, end))
        if end >= duration_sec:
            break
        start += stride_sec
    if windows and windows[-1][1] < duration_sec:
        windows.append((max(0.0, duration_sec - slice_sec), duration_sec))
    deduped: list[tuple[float, float]] = []
    for item in windows:
        if not deduped or deduped[-1] != item:
            deduped.append(item)
    return deduped


def _extract_touch_json(text: str) -> dict[str, object]:
    raw = str(text or "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {
        "top_touch": "unknown",
        "bottom_touch": "unknown",
        "latest_touch_position": "unknown",
        "confidence": 0.0,
        "reason": raw[:160],
    }


def _rename_slice_clip(clip_path: Path, *, point_id: str, idx: int, latest_touch: str) -> Path:
    latest_slug = latest_touch if latest_touch in {"top", "bottom"} else "unknown"
    final_path = clip_path.with_name(f"{point_id}__slice_{idx:02d}__latest_{latest_slug}.mp4")
    if final_path.exists():
        final_path.unlink()
    clip_path.rename(final_path)
    return final_path


def _touch_value(value: str) -> int:
    if value == "yes":
        return 1
    if value == "no":
        return 0
    return -1


def _candidate_event_score(
    *,
    top_touch: str,
    bottom_touch: str,
    latest_touch: str,
    slice_start_sec: float,
    slice_end_sec: float,
    duration_sec: float,
    confidence: float,
) -> float:
    score = 0.0
    top_val = _touch_value(top_touch)
    bottom_val = _touch_value(bottom_touch)
    one_touch_only = (top_val, bottom_val) in {(1, 0), (0, 1)}
    both_touch = (top_val, bottom_val) == (1, 1)

    if one_touch_only:
        score += 4.0
        if latest_touch == "top" and top_val == 1:
            score += 0.75
        if latest_touch == "bottom" and bottom_val == 1:
            score += 0.75
    elif both_touch:
        score += 1.0

    midpoint_ratio = ((float(slice_start_sec) + float(slice_end_sec)) * 0.5) / max(0.01, float(duration_sec))
    if 0.45 <= midpoint_ratio <= 0.85:
        score += 2.0
    elif 0.30 <= midpoint_ratio <= 0.95:
        score += 1.0

    if float(slice_end_sec) >= (float(duration_sec) - 0.15):
        score -= 1.5

    score += max(0.0, min(1.0, float(confidence)))
    return float(score)


def _extract_stage2_json(text: str) -> dict[str, object]:
    raw = str(text or "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {
        "point_ending_touch": "unknown",
        "touch_position": "unknown",
        "post_point_noise": "unknown",
        "confidence": 0.0,
        "reason": raw[:160],
    }


def _stage2_verifier_prompt() -> str:
    return (
        "This is one candidate slice from a longer table-tennis rally. "
        "The TOP player is the far-side player. The BOTTOM player is the near-side player. "
        "A point-ending touch means the last live racket-ball contact before the rally is over. "
        "Ignore post-point movement, recovery motion, and dead-tail noise after the point is already finished. "
        "Return strict JSON only with these keys: "
        '{"point_ending_touch":"yes|no|unknown","touch_position":"top|bottom|unknown","post_point_noise":"yes|no|unknown","confidence":0.0,"reason":"short phrase"}.'
    )


def _stage2_score(
    *,
    stage1_score: float,
    point_ending_touch: str,
    touch_position: str,
    post_point_noise: str,
    latest_touch_position: str,
    confidence: float,
) -> float:
    score = float(stage1_score)
    if point_ending_touch == "yes":
        score += 4.0
    elif point_ending_touch == "no":
        score -= 4.0

    if touch_position in {"top", "bottom"} and latest_touch_position in {"top", "bottom"}:
        if touch_position == latest_touch_position:
            score += 1.25
        else:
            score -= 1.25

    if post_point_noise == "yes":
        score -= 2.0
    elif post_point_noise == "no":
        score += 0.5

    score += max(0.0, min(1.0, float(confidence)))
    return float(score)


def _choose_stage3_candidate(
    *,
    best_stage1: dict[str, object] | None,
    stage2_candidates: list[dict[str, object]],
) -> tuple[dict[str, object] | None, str]:
    agreeing = [
        item
        for item in stage2_candidates
        if str(item.get("stage2_point_ending_touch", "unknown")) == "yes"
        and str(item.get("stage2_post_point_noise", "unknown")) != "yes"
        and str(item.get("stage2_touch_position", "unknown")) in {"top", "bottom"}
        and str(item.get("stage2_touch_position", "unknown")) == str(item.get("latest_touch_position", "unknown"))
    ]
    if agreeing:
        agreeing_sorted = sorted(
            agreeing,
            key=lambda item: (float(item.get("slice_end_sec", 0.0)), float(item.get("candidate_event_score", 0.0))),
            reverse=True,
        )
        return agreeing_sorted[0], "agreeing_latest"

    if best_stage1 is not None:
        return best_stage1, "fallback_stage1"
    return None, "none"


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe overlapping rally slices for final-touch evidence.")
    parser.add_argument("--timeline", required=True, help="Input rally timeline JSON")
    parser.add_argument("--clip-dir", required=True, help="Directory for exported probe clips")
    parser.add_argument("--out", required=True, help="Output JSON summary path")
    parser.add_argument("--model-dir", default="models/Qwen3-VL-4B-Instruct", help="Local HF model directory")
    parser.add_argument("--point-ids", nargs="*", default=[], help="Comma-separated or repeated point ids")
    parser.add_argument("--main-pass-view", choices=["full", "roi", "table_only"], default="table_only")
    parser.add_argument("--main-pass-overlay", choices=["none", "augmented_v2"], default="none")
    parser.add_argument("--roi-margin-ratio", type=float, default=0.4)
    parser.add_argument("--roi-margin-y-ratio", type=float, default=0.9)
    parser.add_argument("--table-only-x-margin-ratio", type=float, default=0.25)
    parser.add_argument("--table-only-top-margin-ratio", type=float, default=0.8)
    parser.add_argument("--table-only-bottom-margin-ratio", type=float, default=0.35)
    parser.add_argument("--aug-ball-profile", choices=["support", "standalone"], default="support")
    parser.add_argument("--aug-ball-trail-length", type=int, default=18)
    parser.add_argument("--slice-sec", type=float, default=2.5)
    parser.add_argument("--slice-stride-sec", type=float, default=1.25)
    parser.add_argument("--stage2-top-k", type=int, default=2)
    parser.add_argument("--fps-sample", type=float, default=4.0)
    parser.add_argument("--min-frames", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=(1280 * 720))
    parser.add_argument("--size-shortest-edge", type=int, default=576)
    parser.add_argument("--size-longest-edge", type=int, default=1048576)
    args = parser.parse_args()

    timeline = load_rally_timeline(Path(args.timeline))
    clip_dir = Path(args.clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    selected_ids = _selected_point_ids(args.point_ids)
    if not selected_ids:
        raise SystemExit("No point ids provided.")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required: torch.cuda.is_available() returned False for native-video multislice probing.")
    torch.cuda.set_device(0)

    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

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
        generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return str(output_text[0]).strip() if output_text else ""

    prompt = (
        "This is a short slice from a table-tennis rally. "
        "The TOP player is the far-side player. The BOTTOM player is the near-side player. "
        "Focus only on visible racket-ball contact in this slice. "
        "Return strict JSON only with these keys: "
        '{"top_touch":"yes|no|unknown","bottom_touch":"yes|no|unknown","latest_touch_position":"top|bottom|unknown","confidence":0.0,"reason":"short phrase"}.'
    )

    results: dict[str, object] = {
        "timeline": str(Path(args.timeline)),
        "model_dir": str(args.model_dir),
        "main_pass_view": str(args.main_pass_view),
        "main_pass_overlay": str(args.main_pass_overlay),
        "slice_sec": float(args.slice_sec),
        "slice_stride_sec": float(args.slice_stride_sec),
        "stage2_top_k": int(args.stage2_top_k),
        "points": {},
    }

    for point in timeline.points:
        if point.id not in selected_ids:
            continue

        point_results: list[dict[str, object]] = []
        full_clip = clip_dir / f"{point.id}__full_rally.mp4"
        _clip_window_video(
            source_video=timeline.video_path,
            clip_path=full_clip,
            start_sec=float(point.t_start),
            end_sec=float(point.t_end),
        )

        source_for_slices = full_clip
        if str(args.main_pass_view) == "roi":
            roi_clip = clip_dir / f"{point.id}__base_roi.mp4"
            _build_roi_clip(
                source_clip=full_clip,
                roi_clip=roi_clip,
                roi=timeline.roi,
                margin_ratio=float(args.roi_margin_ratio),
                margin_y_ratio=float(args.roi_margin_y_ratio),
            )
            source_for_slices = roi_clip
        elif str(args.main_pass_view) == "table_only":
            table_clip = clip_dir / f"{point.id}__base_tableonly.mp4"
            _build_table_only_clip(
                source_clip=full_clip,
                table_only_clip=table_clip,
                roi=timeline.roi,
                x_margin_ratio=float(args.table_only_x_margin_ratio),
                top_margin_ratio=float(args.table_only_top_margin_ratio),
                bottom_margin_ratio=float(args.table_only_bottom_margin_ratio),
            )
            source_for_slices = table_clip

        if str(args.main_pass_overlay) == "augmented_v2":
            aug_clip = clip_dir / f"{point.id}__base_augv2.mp4"
            _build_augmented_v2_clip(
                source_clip=full_clip,
                augmented_clip=aug_clip,
                roi=timeline.roi,
                margin_ratio=float(args.roi_margin_ratio),
                margin_y_ratio=float(args.roi_margin_y_ratio),
                trail_length=int(args.aug_ball_trail_length),
                ball_profile=str(args.aug_ball_profile),
            )
            source_for_slices = aug_clip

        duration_sec = max(0.01, float(point.t_end) - float(point.t_start))
        windows = _slice_windows(duration_sec, float(args.slice_sec), float(args.slice_stride_sec))
        for idx, (slice_start, slice_end) in enumerate(windows):
            slice_clip = clip_dir / f"{point.id}__slice_tmp_{idx:02d}.mp4"
            _clip_window_video(
                source_video=str(source_for_slices),
                clip_path=slice_clip,
                start_sec=float(slice_start),
                end_sec=float(slice_end),
            )
            raw_text = ask_text_for_video(slice_clip, prompt)
            parsed = _extract_touch_json(raw_text)
            latest_touch = str(parsed.get("latest_touch_position", "unknown")).strip().lower()
            confidence = parsed.get("confidence", 0.0)
            top_touch = str(parsed.get("top_touch", "unknown")).strip().lower()
            bottom_touch = str(parsed.get("bottom_touch", "unknown")).strip().lower()
            event_score = _candidate_event_score(
                top_touch=top_touch,
                bottom_touch=bottom_touch,
                latest_touch=latest_touch,
                slice_start_sec=float(slice_start),
                slice_end_sec=float(slice_end),
                duration_sec=duration_sec,
                confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.0,
            )
            final_clip = _rename_slice_clip(slice_clip, point_id=point.id, idx=idx, latest_touch=latest_touch)
            point_results.append(
                {
                    "slice_index": idx,
                    "slice_start_sec": float(slice_start),
                    "slice_end_sec": float(slice_end),
                    "latest_touch_position": latest_touch,
                    "top_touch": top_touch,
                    "bottom_touch": bottom_touch,
                    "confidence": confidence,
                    "candidate_event_score": event_score,
                    "reason": str(parsed.get("reason", "")),
                    "raw_text": raw_text,
                    "file": final_clip.name,
                }
            )

        if source_for_slices != full_clip and source_for_slices.exists():
            source_for_slices.unlink()

        ranked = sorted(point_results, key=lambda item: float(item.get("candidate_event_score", 0.0)), reverse=True)
        top_candidates = ranked[: max(1, int(args.stage2_top_k))]
        stage2_candidates: list[dict[str, object]] = []
        for candidate in top_candidates:
            candidate_clip = clip_dir / str(candidate["file"])
            raw_text_stage2 = ask_text_for_video(candidate_clip, _stage2_verifier_prompt())
            parsed_stage2 = _extract_stage2_json(raw_text_stage2)
            point_ending_touch = str(parsed_stage2.get("point_ending_touch", "unknown")).strip().lower()
            touch_position = str(parsed_stage2.get("touch_position", "unknown")).strip().lower()
            post_point_noise = str(parsed_stage2.get("post_point_noise", "unknown")).strip().lower()
            stage2_confidence = parsed_stage2.get("confidence", 0.0)
            final_score = _stage2_score(
                stage1_score=float(candidate.get("candidate_event_score", 0.0)),
                point_ending_touch=point_ending_touch,
                touch_position=touch_position,
                post_point_noise=post_point_noise,
                latest_touch_position=str(candidate.get("latest_touch_position", "unknown")).strip().lower(),
                confidence=float(stage2_confidence) if isinstance(stage2_confidence, (int, float)) else 0.0,
            )
            enriched = dict(candidate)
            enriched.update(
                {
                    "stage2_point_ending_touch": point_ending_touch,
                    "stage2_touch_position": touch_position,
                    "stage2_post_point_noise": post_point_noise,
                    "stage2_confidence": stage2_confidence,
                    "stage2_reason": str(parsed_stage2.get("reason", "")),
                    "stage2_raw_text": raw_text_stage2,
                    "stage2_final_score": final_score,
                }
            )
            stage2_candidates.append(enriched)
        stage2_ranked = sorted(stage2_candidates, key=lambda item: float(item.get("stage2_final_score", 0.0)), reverse=True)
        best_stage3, stage3_strategy = _choose_stage3_candidate(
            best_stage1=(ranked[0] if ranked else None),
            stage2_candidates=stage2_candidates,
        )
        results["points"][point.id] = {
            "best_slice": (ranked[0] if ranked else None),
            "stage2_candidates": stage2_candidates,
            "best_slice_stage2": (stage2_ranked[0] if stage2_ranked else None),
            "best_slice_stage3": best_stage3,
            "stage3_strategy": stage3_strategy,
            "slices": point_results,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved probe summary to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
