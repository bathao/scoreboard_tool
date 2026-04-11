from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from backend.config import PROJECT_ROOT
from backend.production_jobs import (
    KNOWN_WINNERS,
    MatchJob,
    accept_point_prediction,
    apply_point_no_score,
    apply_point_review,
    build_review_status,
    load_match_job,
    save_match_job,
    update_job_runtime_state,
)
from backend.rally_timeline_contract import RallyTimeline, counts_toward_score, load_rally_timeline, save_rally_timeline
from backend.rendering import render_scoreboard_video
from backend.score_validation import build_score_validation


SCRIPTS_DIR = PROJECT_ROOT / "scripts"


LOGS_DIR = PROJECT_ROOT / "logs"


def _job_log(job_dir: str | Path, message: str) -> None:
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {message}\n"
    try:
        job_id = Path(job_dir).name
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOGS_DIR / f"{job_id}.log", "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    print(line, end="", flush=True)


@dataclass
class ProductionPipelineConfig:
    table_weights_path: str = "weights/yolov8x_table.pt"
    pose_weights_path: str = "weights/yolov8x-pose.pt"
    base_model_dir: str = "models/Qwen3-VL-4B-Instruct"
    adapter_dir: str = "models/adapters/qwen3vl4b_table_tennis_pilot_4ep_cache_v2"
    fps_sample: float = 1.0
    min_frames: int = 4
    max_frames: int = 4
    size_shortest_edge: int = 384
    size_longest_edge: int = 1048576
    max_pixels: int = 262144
    max_new_tokens: int = 64


def _ensure_scripts_importable() -> None:
    scripts_path = str(SCRIPTS_DIR)
    if scripts_path not in sys.path:
        sys.path.append(scripts_path)


def _load_build_rally_timeline():
    _ensure_scripts_importable()
    from generate_rally_timeline import build_rally_timeline  # type: ignore

    return build_rally_timeline


def _load_winner_prompt_helpers():
    _ensure_scripts_importable()
    from winner_finetune_common import (  # type: ignore
        ACTIVE_PILOT_TAXONOMY_ORDER,
        build_training_prompt,
        parse_prediction_json,
    )

    return ACTIVE_PILOT_TAXONOMY_ORDER, build_training_prompt, parse_prediction_json


def _run_ffmpeg(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        message = stderr if stderr else stdout
        raise RuntimeError(message if message else f"ffmpeg failed with code {exc.returncode}") from exc


def trim_input_video(raw_video_path: str, working_video_path: str, trim_start_sec: float) -> str:
    raw_path = Path(raw_video_path).resolve()
    working_path = Path(working_video_path).resolve()
    working_path.parent.mkdir(parents=True, exist_ok=True)

    if float(trim_start_sec) <= 0.0001:
        if raw_path != working_path:
            shutil.copy2(raw_path, working_path)
        return str(working_path)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{float(trim_start_sec):.3f}",
        "-i", str(raw_path),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(working_path),
    ]
    _run_ffmpeg(cmd)
    return str(working_path)


def export_review_clips(timeline: RallyTimeline, *, working_video_path: str, review_clips_dir: str) -> dict[str, str]:
    review_dir = Path(review_clips_dir).resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    src = str(Path(working_video_path).resolve())

    def _export_one(point) -> tuple[str, str]:
        clip_path = review_dir / f"{point.id}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{float(point.t_start):.3f}",
            "-to", f"{float(point.t_end):.3f}",
            "-i", src,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-threads", "2",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(clip_path),
        ]
        _run_ffmpeg(cmd)
        return point.id, str(clip_path).replace("\\", "/")

    workers = min(len(timeline.points), 8)
    exported: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_export_one, point): point.id for point in timeline.points}
        for future in as_completed(futures):
            point_id, path = future.result()
            exported[point_id] = path
    return exported


class WinnerAdapterPredictor:
    def __init__(self, config: ProductionPipelineConfig):
        self.config = config

        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        taxonomy_order, build_training_prompt, parse_prediction_json = _load_winner_prompt_helpers()
        self._torch = torch
        self._parse_prediction_json = parse_prediction_json
        self.prompt_text = build_training_prompt(list(taxonomy_order))
        self.processor = AutoProcessor.from_pretrained(config.base_model_dir)
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.base_model_dir,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base_model, config.adapter_dir)
        self.model.eval()

    def predict_clip(self, clip_path: str | Path) -> tuple[dict[str, str], str]:
        clip = str(Path(clip_path).resolve())
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": clip},
                    {"type": "text", "text": self.prompt_text},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processor_kwargs: dict[str, Any] = {
            "text": [text],
            "videos": [clip],
            "return_tensors": "pt",
            "fps": float(self.config.fps_sample),
            "min_frames": int(self.config.min_frames),
            "max_frames": int(self.config.max_frames),
            "size": {
                "shortest_edge": int(self.config.size_shortest_edge),
                "longest_edge": int(self.config.size_longest_edge),
            },
        }
        if int(self.config.max_pixels) > 0:
            processor_kwargs["max_pixels"] = int(self.config.max_pixels)

        inputs = self.processor(**processor_kwargs)
        inputs = {key: (value.to(self.model.device) if hasattr(value, "to") else value) for key, value in inputs.items()}

        with self._torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=int(self.config.max_new_tokens), do_sample=False)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return self._parse_prediction_json(output_text), output_text


def _apply_adapter_predictions(
    timeline: RallyTimeline,
    *,
    predictions_jsonl_path: str,
    predictor: WinnerAdapterPredictor,
    review_clips: dict[str, str],
    adapter_dir: str,
) -> RallyTimeline:
    records: list[dict[str, Any]] = []
    for point in timeline.points:
        if not counts_toward_score(point):
            continue

        parsed, raw_output = predictor.predict_clip(review_clips[point.id])
        predicted_winner = str(parsed.get("winner", "")).strip()
        predicted_loser = str(parsed.get("loser", "")).strip()
        predicted_taxonomy = str(parsed.get("taxonomy", "")).strip()
        predicted_last_hitter = str(parsed.get("last_hitter", "")).strip()

        point.winner_model = str(adapter_dir)
        point.winner_end_category = predicted_taxonomy or point.winner_end_category
        point.winner_last_hitter_candidate = predicted_last_hitter if predicted_last_hitter in KNOWN_WINNERS else "unknown"
        point.winner_loser_candidate = predicted_loser if predicted_loser in KNOWN_WINNERS else "unknown"
        point.winner_confidence = 0.0
        point.source = "ai"

        if predicted_winner in KNOWN_WINNERS:
            point.winner = predicted_winner
            point.winner_candidate = predicted_winner
            point.winner_decision = "review"
            point.winner_reason = "adapter_prediction_pending_review"
            if point.winner_loser_candidate not in KNOWN_WINNERS:
                point.winner_loser_candidate = "player_b" if predicted_winner == "player_a" else "player_a"
        else:
            point.winner = "unknown"
            point.winner_candidate = "unknown"
            point.winner_decision = "blocked"
            point.winner_reason = "adapter_prediction_missing"

        records.append(
            {
                "point_id": point.id,
                "clip_path": review_clips[point.id],
                "winner_pred": predicted_winner,
                "loser_pred": predicted_loser,
                "taxonomy_pred": predicted_taxonomy,
                "last_hitter_pred": predicted_last_hitter,
                "raw_output": raw_output,
            }
        )

    output_path = Path(predictions_jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return timeline


def _load_or_raise_job(job_or_path: MatchJob | str | Path) -> MatchJob:
    if isinstance(job_or_path, MatchJob):
        return job_or_path
    return load_match_job(job_or_path)


def load_job_timeline(job_or_path: MatchJob | str | Path) -> RallyTimeline:
    job = _load_or_raise_job(job_or_path)
    return load_rally_timeline(Path(job.artifacts.timeline_json_path))


def run_initial_job_pipeline(
    job_or_path: MatchJob | str | Path,
    *,
    config: ProductionPipelineConfig | None = None,
    stop_check: "Callable[[], bool] | None" = None,
) -> MatchJob:
    from typing import Callable

    def _check_stop() -> None:
        if stop_check and stop_check():
            raise RuntimeError("stopped_by_operator")

    config = config or ProductionPipelineConfig()
    job = _load_or_raise_job(job_or_path)
    raw_video_path = Path(job.raw_video_path)
    if not raw_video_path.exists():
        raise FileNotFoundError(f"Raw video not found: {raw_video_path}")

    job_dir = job.artifacts.job_dir
    _job_log(job_dir, f"Pipeline started — job {job.job_id}")
    _job_log(job_dir, f"Input: {Path(job.raw_video_path).name}  trim_start={job.trim_start_sec}s  best_of={job.best_of}")

    _check_stop()
    update_job_runtime_state(job, status="running", current_step="trim_input", error_message="")
    _job_log(job_dir, "Step 1/5: trim_input — cutting working video with GPU encoder")
    trim_input_video(job.raw_video_path, job.artifacts.working_video_path, job.trim_start_sec)
    _job_log(job_dir, "Step 1/5: trim_input — done")

    _check_stop()
    update_job_runtime_state(job, status="running", current_step="generate_rally_timeline")
    _job_log(job_dir, "Step 2/5: generate_rally_timeline — loading YOLO models and detecting rallies (slowest step)")
    build_rally_timeline = _load_build_rally_timeline()
    timeline = build_rally_timeline(
        job.artifacts.working_video_path,
        config.table_weights_path,
        pose_weights_path=config.pose_weights_path,
        best_of=job.best_of,
    )
    timeline.video_path = str(Path(job.artifacts.working_video_path).resolve()).replace("\\", "/")
    save_rally_timeline(Path(job.artifacts.timeline_json_path), timeline)
    _job_log(job_dir, f"Step 2/5: generate_rally_timeline — done, {len(timeline.points)} rallies detected")

    _check_stop()
    update_job_runtime_state(job, status="running", current_step="export_review_clips", timeline=timeline)
    _job_log(job_dir, f"Step 3/5: export_review_clips — cutting {len(timeline.points)} clips in parallel")
    review_clips = export_review_clips(
        timeline,
        working_video_path=job.artifacts.working_video_path,
        review_clips_dir=job.artifacts.review_clips_dir,
    )
    _job_log(job_dir, f"Step 3/5: export_review_clips — done, {len(review_clips)} clips")

    _check_stop()
    update_job_runtime_state(job, status="running", current_step="predict_winners_with_adapter", timeline=timeline)
    _job_log(job_dir, "Step 4/5: predict_winners_with_adapter — loading Qwen3-VL adapter")
    predictor = WinnerAdapterPredictor(config)
    timeline = _apply_adapter_predictions(
        timeline,
        predictions_jsonl_path=job.artifacts.predictions_jsonl_path,
        predictor=predictor,
        review_clips=review_clips,
        adapter_dir=config.adapter_dir,
    )
    timeline.score_validation = build_score_validation(timeline, expected_scope="any")
    save_rally_timeline(Path(job.artifacts.timeline_json_path), timeline)
    _job_log(job_dir, "Step 4/5: predict_winners_with_adapter — done")

    _check_stop()
    update_job_runtime_state(job, status="running", current_step="render_preview", timeline=timeline)
    _job_log(job_dir, "Step 5/5: render_preview — rendering scoreboard video")
    render_job_preview(job)
    timeline = load_job_timeline(job)
    review_status = build_review_status(timeline)
    next_status = "ready_for_final" if review_status["final_export_ready"] else "needs_review"
    next_step = "preview_ready" if review_status["preview_render_allowed"] else "review_required_no_preview"
    update_job_runtime_state(job, status=next_status, current_step=next_step, timeline=timeline)
    _job_log(job_dir, f"Pipeline complete — status={next_status}")
    return job


def render_job_preview(job_or_path: MatchJob | str | Path) -> MatchJob:
    job = _load_or_raise_job(job_or_path)
    timeline = load_job_timeline(job)
    review_status = build_review_status(timeline)
    if not review_status["preview_render_allowed"]:
        update_job_runtime_state(job, status="needs_review", current_step="preview_skipped_no_known_winner", timeline=timeline)
        return job
    render_scoreboard_video(
        input_video_path=job.artifacts.working_video_path,
        timeline=timeline,
        output_video_path=job.artifacts.preview_video_path,
        player_a_name=job.player_a_name,
        player_b_name=job.player_b_name,
    )
    update_job_runtime_state(job, status=job.status, current_step="preview_ready", timeline=timeline)
    return job


def export_job_final_video(job_or_path: MatchJob | str | Path) -> MatchJob:
    job = _load_or_raise_job(job_or_path)
    timeline = load_job_timeline(job)
    review_status = build_review_status(timeline)
    if not review_status["final_export_ready"]:
        raise RuntimeError("Final export is blocked until all scoring rallies have resolved winners.")

    update_job_runtime_state(job, status="running", current_step="final_export", timeline=timeline)
    render_scoreboard_video(
        input_video_path=job.artifacts.working_video_path,
        timeline=timeline,
        output_video_path=job.artifacts.final_video_path,
        player_a_name=job.player_a_name,
        player_b_name=job.player_b_name,
    )
    update_job_runtime_state(job, status="completed", current_step="final_export_complete", timeline=timeline)
    return job


def review_job_point(
    job_or_path: MatchJob | str | Path,
    *,
    point_id: str,
    action: str,
    winner: str | None = None,
    reviewer: str = "operator",
    note: str = "",
) -> MatchJob:
    job = _load_or_raise_job(job_or_path)
    timeline = load_job_timeline(job)

    if action == "keep":
        accept_point_prediction(timeline, point_id=point_id, reviewer=reviewer, note=note)
    elif action == "set_winner":
        if winner not in KNOWN_WINNERS:
            raise ValueError("winner must be player_a or player_b for set_winner action")
        apply_point_review(timeline, point_id=point_id, winner=winner, reviewer=reviewer, note=note)
    elif action == "mark_let":
        apply_point_no_score(timeline, point_id=point_id, reviewer=reviewer, note=note)
    else:
        raise ValueError(f"Unsupported review action: {action}")

    timeline.score_validation = build_score_validation(timeline, expected_scope="any")
    save_rally_timeline(Path(job.artifacts.timeline_json_path), timeline)
    next_status = "ready_for_final" if build_review_status(timeline)["final_export_ready"] else "needs_review"
    update_job_runtime_state(job, status=next_status, current_step="review_updated", timeline=timeline)
    return job
