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
from backend.production_defaults import PRODUCTION_RALLY_DEFAULTS, PRODUCTION_WINNER_DEFAULTS
from backend.production_jobs import (
    KNOWN_WINNERS,
    MatchJob,
    accept_point_prediction,
    apply_point_no_score,
    apply_point_review,
    build_review_status,
    load_match_job,
    near_player_for_rally,
    save_match_job,
    update_job_runtime_state,
)
from backend.rally_timeline_contract import RallyTimeline, counts_toward_score, load_rally_timeline, save_rally_timeline
from backend.rendering import render_scoreboard_video
from backend.score_validation import build_score_validation
from backend.set_boundary import apply_set_numbers


SCRIPTS_DIR = PROJECT_ROOT / "scripts"


LOGS_DIR = PROJECT_ROOT / "logs"


def _job_log(job_dir: str | Path, message: str) -> None:
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
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
    table_weights_path: str = PRODUCTION_RALLY_DEFAULTS.table_weights_path
    pose_weights_path: str = PRODUCTION_RALLY_DEFAULTS.pose_weights_path
    rally_stride: int = PRODUCTION_RALLY_DEFAULTS.stride
    rally_mode: str = PRODUCTION_RALLY_DEFAULTS.mode
    rally_player_margin_px: int = PRODUCTION_RALLY_DEFAULTS.player_margin_px
    rally_player_fuse_gain: float = PRODUCTION_RALLY_DEFAULTS.player_fuse_gain
    rally_player_signal_source: str = PRODUCTION_RALLY_DEFAULTS.player_signal_source
    rally_ball_fuse_gain: float = PRODUCTION_RALLY_DEFAULTS.ball_fuse_gain
    rally_ball_signal_source: str = PRODUCTION_RALLY_DEFAULTS.ball_signal_source
    base_model_dir: str = PRODUCTION_WINNER_DEFAULTS.base_model_dir
    adapter_dir: str = PRODUCTION_WINNER_DEFAULTS.adapter_dir
    fps_sample: float = PRODUCTION_WINNER_DEFAULTS.fps_sample
    min_frames: int = PRODUCTION_WINNER_DEFAULTS.min_frames
    max_frames: int = PRODUCTION_WINNER_DEFAULTS.max_frames
    size_shortest_edge: int = PRODUCTION_WINNER_DEFAULTS.size_shortest_edge
    size_longest_edge: int = PRODUCTION_WINNER_DEFAULTS.size_longest_edge
    max_pixels: int = PRODUCTION_WINNER_DEFAULTS.max_pixels
    max_new_tokens: int = PRODUCTION_WINNER_DEFAULTS.max_new_tokens


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

    if not timeline.points:
        return {}
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
    best_of: int = 5,
    player_a_starts_near: bool = True,
) -> RallyTimeline:
    """Run adapter inference on each scoring rally and write winner predictions.

    The adapter was trained on Set-1 data where player_a = NEAR side.
    Its output 'player_a' means 'the NEAR-side player won', 'player_b' means FAR won.
    We remap those NEAR/FAR-relative labels to the actual player identity using
    near_player_for_rally(), which accounts for side-swaps between sets and the
    mid-deciding-set swap at score 5.
    """
    records: list[dict[str, Any]] = []

    # Running match state — updated after each scoring point so we can compute
    # set_number and in-set scores for the NEXT rally's near_player_for_rally() call.
    r_score_a, r_score_b = 0, 0
    r_sets_a, r_sets_b = 0, 0
    r_set_number = 1

    for point in timeline.points:
        if not counts_toward_score(point):
            continue

        parsed, raw_output = predictor.predict_clip(review_clips[point.id])
        # Model outputs NEAR/FAR-relative labels (trained with player_a = NEAR, Set 1)
        raw_winner = str(parsed.get("winner", "")).strip()
        raw_loser = str(parsed.get("loser", "")).strip()
        predicted_taxonomy = str(parsed.get("taxonomy", "")).strip()
        predicted_last_hitter = str(parsed.get("last_hitter", "")).strip()

        # Remap NEAR/FAR-relative prediction to actual player identity
        # raw_winner == "player_a" → NEAR side won; "player_b" → FAR side won
        near_player = near_player_for_rally(r_set_number, r_score_a, r_score_b, best_of, player_a_starts_near)
        far_player = "player_b" if near_player == "player_a" else "player_a"

        if raw_winner == "player_a":   # model: NEAR won
            predicted_winner = near_player
            predicted_loser = far_player
        elif raw_winner == "player_b":  # model: FAR won
            predicted_winner = far_player
            predicted_loser = near_player
        else:
            predicted_winner = ""
            predicted_loser = ""

        # Remap last_hitter the same way
        if predicted_last_hitter == "player_a":
            predicted_last_hitter = near_player
        elif predicted_last_hitter == "player_b":
            predicted_last_hitter = far_player
        else:
            predicted_last_hitter = "unknown"

        # Remap raw_loser (if model provided one separately)
        if raw_loser == "player_a":
            raw_loser = near_player
        elif raw_loser == "player_b":
            raw_loser = far_player
        else:
            raw_loser = predicted_loser  # fall back to the derived value

        point.winner_model = str(adapter_dir)
        point.winner_end_category = predicted_taxonomy or point.winner_end_category
        point.winner_last_hitter_candidate = predicted_last_hitter if predicted_last_hitter in KNOWN_WINNERS else "unknown"
        point.winner_loser_candidate = raw_loser if raw_loser in KNOWN_WINNERS else "unknown"
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
                "raw_winner_pred": raw_winner,
                "winner_pred": predicted_winner,
                "loser_pred": predicted_loser,
                "taxonomy_pred": predicted_taxonomy,
                "last_hitter_pred": predicted_last_hitter,
                "set_number_at_pred": r_set_number,
                "near_player_at_pred": near_player,
                "raw_output": raw_output,
            }
        )

        # Advance running match state using the remapped prediction
        if predicted_winner in KNOWN_WINNERS:
            if predicted_winner == "player_a":
                r_score_a += 1
            else:
                r_score_b += 1
            # Detect set end (standard table tennis: 11+ points, diff >= 2)
            sets_needed = (best_of + 1) // 2
            a_wins_set = r_score_a >= 11 and (r_score_a - r_score_b) >= 2
            b_wins_set = r_score_b >= 11 and (r_score_b - r_score_a) >= 2
            if a_wins_set:
                r_sets_a += 1
                r_score_a, r_score_b = 0, 0
                if r_sets_a < sets_needed:
                    r_set_number += 1
            elif b_wins_set:
                r_sets_b += 1
                r_score_a, r_score_b = 0, 0
                if r_sets_b < sets_needed:
                    r_set_number += 1

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
    _job_log(
        job_dir,
        "Production rally config - "
        f"weights={config.table_weights_path} pose={config.pose_weights_path} "
        f"mode={config.rally_mode} stride={config.rally_stride} "
        f"player_signal={config.rally_player_signal_source} ball_signal={config.rally_ball_signal_source}",
    )
    _job_log(
        job_dir,
        "Production winner config - "
        f"base={config.base_model_dir} adapter={config.adapter_dir} "
        f"fps={config.fps_sample} frames={config.min_frames}-{config.max_frames} "
        f"shortest_edge={config.size_shortest_edge} max_pixels={config.max_pixels}",
    )

    _check_stop()
    update_job_runtime_state(job, status="running", current_step="trim_input", error_message="")
    _job_log(job_dir, "Step 1/4: trim_input — cutting working video with GPU encoder")
    trim_input_video(job.raw_video_path, job.artifacts.working_video_path, job.trim_start_sec)
    _job_log(job_dir, "Step 1/4: trim_input — done")

    _check_stop()
    update_job_runtime_state(job, status="running", current_step="generate_rally_timeline")
    _job_log(job_dir, "Step 2/4: generate_rally_timeline — loading YOLO models and detecting rallies (slowest step)")
    build_rally_timeline = _load_build_rally_timeline()
    timeline = build_rally_timeline(
        job.artifacts.working_video_path,
        config.table_weights_path,
        pose_weights_path=config.pose_weights_path,
        best_of=job.best_of,
        stride=config.rally_stride,
        mode=config.rally_mode,
        player_margin_px=config.rally_player_margin_px,
        player_fuse_gain=config.rally_player_fuse_gain,
        player_signal_source=config.rally_player_signal_source,
        ball_fuse_gain=config.rally_ball_fuse_gain,
        ball_signal_source=config.rally_ball_signal_source,
        log_fn=lambda msg: _job_log(job_dir, msg),
    )
    timeline.video_path = str(Path(job.artifacts.working_video_path).resolve()).replace("\\", "/")
    save_rally_timeline(Path(job.artifacts.timeline_json_path), timeline)
    _job_log(job_dir, f"Step 2/4: generate_rally_timeline — done, {len(timeline.points)} rallies detected")
    if not timeline.points:
        raise RuntimeError(
            "No rallies detected in this video. "
            "Check that the video contains table tennis gameplay and the table is clearly visible."
        )

    _check_stop()
    update_job_runtime_state(job, status="running", current_step="export_review_clips", timeline=timeline)
    _job_log(job_dir, f"Step 3/4: export_review_clips — cutting {len(timeline.points)} clips in parallel")
    review_clips = export_review_clips(
        timeline,
        working_video_path=job.artifacts.working_video_path,
        review_clips_dir=job.artifacts.review_clips_dir,
    )
    _job_log(job_dir, f"Step 3/4: export_review_clips — done, {len(review_clips)} clips")

    _check_stop()
    update_job_runtime_state(job, status="running", current_step="predict_winners_with_adapter", timeline=timeline)
    _job_log(job_dir, "Step 4/4: predict_winners_with_adapter — loading Qwen3-VL adapter")
    predictor = WinnerAdapterPredictor(config)
    timeline = _apply_adapter_predictions(
        timeline,
        predictions_jsonl_path=job.artifacts.predictions_jsonl_path,
        predictor=predictor,
        review_clips=review_clips,
        adapter_dir=config.adapter_dir,
        best_of=job.best_of,
        player_a_starts_near=job.player_a_starts_near,
    )
    apply_set_numbers(timeline, best_of=job.best_of)
    timeline.score_validation = build_score_validation(timeline, expected_scope="any")
    save_rally_timeline(Path(job.artifacts.timeline_json_path), timeline)
    _job_log(job_dir, "Step 4/4: predict_winners_with_adapter — done")

    review_status = build_review_status(timeline)
    next_status = "ready_for_final" if review_status["final_export_ready"] else "needs_review"
    update_job_runtime_state(job, status=next_status, current_step="ai_ready", timeline=timeline)
    _job_log(job_dir, f"Pipeline complete — {len(timeline.points)} rallies ready for review, status={next_status}")
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
    job_dir = job.artifacts.job_dir
    update_job_runtime_state(job, status="running", current_step="final_export", timeline=timeline)
    _job_log(job_dir, f"Final export started — {job.player_a_name} vs {job.player_b_name}")
    _job_log(job_dir, f"Rallies in timeline: {len(timeline.points)}")
    _job_log(job_dir, f"Output: {Path(job.artifacts.final_video_path).name}")
    _job_log(job_dir, "Rendering scoreboard overlay onto full match video (this takes a few minutes)...")
    render_scoreboard_video(
        input_video_path=job.artifacts.working_video_path,
        timeline=timeline,
        output_video_path=job.artifacts.final_video_path,
        player_a_name=job.player_a_name,
        player_b_name=job.player_b_name,
        tournament_name=job.tournament_name,
        round_name=job.round_name,
    )
    _job_log(job_dir, "Render + audio merge complete.")
    _job_log(job_dir, f"Final export complete → {job.artifacts.final_video_path}")
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
