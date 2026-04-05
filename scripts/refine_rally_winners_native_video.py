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


def _rename_clip_with_winner(
    clip_path: Path,
    *,
    point_id: str,
    winner: str,
) -> Path:
    side = "near" if winner == "player_a" else "far" if winner == "player_b" else "unknown"
    final_name = f"{point_id}__pick_{side}__native_video_qwen3vl4b.mp4"
    final_path = clip_path.with_name(final_name)
    if final_path.exists():
        final_path.unlink()
    clip_path.rename(final_path)
    return final_path


def _update_point(point: RallyTimelinePoint, *, winner: str, raw_text: str, model_name: str, clip_path: Path) -> None:
    point.winner_candidate = winner  # type: ignore[assignment]
    point.winner_confidence = 0.50 if winner in {"player_a", "player_b"} else 0.0
    point.winner_decision = "review" if winner in {"player_a", "player_b"} else "blocked"
    point.winner_reason = raw_text[:160].strip() or None
    point.winner_model = model_name
    point.winner_score_a = 1.0 if winner == "player_a" else 0.0
    point.winner_score_b = 1.0 if winner == "player_b" else 0.0
    point.winner = "unknown"
    point.source = "ai"
    point.flags = sorted(
        set(
            point.flags
            + [
                "winner_native_video",
                "winner_model_qwen3_vl_4b_transformers",
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
                    "winner_clip": str(clip_path),
                }
            },
            note="native-video Qwen3-VL-4B winner inference",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Refine rally winners using native-video Transformers with Qwen3-VL-4B.")
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
    parser.add_argument("--fps-sample", type=float, default=1.0, help="Native-video sampling fps")
    parser.add_argument("--min-frames", type=int, default=4, help="Minimum sampled frames")
    parser.add_argument("--max-frames", type=int, default=4, help="Maximum sampled frames")
    parser.add_argument("--size-shortest-edge", type=int, default=1024, help="Video processor shortest edge")
    parser.add_argument("--size-longest-edge", type=int, default=1048576, help="Video processor longest edge")
    parser.add_argument("--point-ids", nargs="*", default=[], help="Optional point ids to process")
    parser.add_argument("--max-points", type=int, default=None, help="Optional cap on processed rallies")
    args = parser.parse_args()

    timeline = load_rally_timeline(Path(args.timeline))
    boundary_signature_before = _frozen_boundary_signature(timeline.points)
    output_path = Path(args.out)
    clip_dir = Path(args.clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)

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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(temp_clip.resolve())},
                    {
                        "type": "text",
                        "text": (
                            "This is the last two-thirds of one table-tennis rally. "
                            "Player A is the near-side player. Player B is the far-side player. "
                            "Reply with only one token: player_a or player_b."
                        ),
                    },
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            videos=[str(temp_clip.resolve())],
            return_tensors="pt",
            fps=float(args.fps_sample),
            min_frames=int(args.min_frames),
            max_frames=int(args.max_frames),
            size={
                "shortest_edge": int(args.size_shortest_edge),
                "longest_edge": int(args.size_longest_edge),
            },
        )
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        generated_ids = model.generate(**inputs, max_new_tokens=12, do_sample=False)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        raw_text = str(output_text[0]).strip() if output_text else ""
        winner = _extract_winner_label(raw_text)

        final_clip = _rename_clip_with_winner(temp_clip, point_id=point.id, winner=winner)
        _update_point(point, winner=winner, raw_text=raw_text, model_name="Qwen3-VL-4B-Instruct", clip_path=final_clip)
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
                "file": final_clip.name,
            }
        )
        processed += 1
        print(
            f"   > {point.id}: {winner} | review | "
            f"window={window_start:.2f}->{window_end:.2f} | raw={raw_text!r}"
        )

    timeline.analysis_metadata["winner_inference_mode"] = "transformers_native_video_qwen3_vl_4b_v1"
    timeline.analysis_metadata["winner_native_video_model_dir"] = str(Path(args.model_dir))
    timeline.analysis_metadata["winner_vlm_window_ratio"] = float(args.window_ratio)
    timeline.analysis_metadata["winner_full_rally_threshold_sec"] = float(args.full_rally_threshold_sec)
    timeline.analysis_metadata["winner_native_video_fps_sample"] = float(args.fps_sample)
    timeline.analysis_metadata["winner_native_video_min_frames"] = int(args.min_frames)
    timeline.analysis_metadata["winner_native_video_max_frames"] = int(args.max_frames)
    timeline.analysis_metadata["winner_native_video_size_shortest_edge"] = int(args.size_shortest_edge)
    timeline.analysis_metadata["winner_native_video_size_longest_edge"] = int(args.size_longest_edge)
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
