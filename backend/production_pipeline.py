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
from backend.set_boundary import apply_set_numbers, populate_player_positions
from backend.step3_rally_start_review import Step3PlayerContext, build_step3_1_rally_start_review


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


def _require_nvenc_gpu() -> None:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi",
        "-i", "testsrc2=size=640x360:rate=1:duration=1",
        "-frames:v", "1",
        "-c:v", "h264_nvenc",
        "-preset", "p1",
        "-f", "null",
        "-",
    ]
    try:
        _run_ffmpeg(cmd)
    except RuntimeError as exc:
        raise RuntimeError(
            "GPU required: FFmpeg h264_nvenc is unavailable. "
            "Check the NVIDIA driver, RTX 5060 Ti visibility, and FFmpeg NVENC support. "
            f"Original error: {exc}"
        ) from exc


def trim_input_video(raw_video_path: str, working_video_path: str, trim_start_sec: float) -> str:
    raw_path = Path(raw_video_path).resolve()
    working_path = Path(working_video_path).resolve()
    working_path.parent.mkdir(parents=True, exist_ok=True)
    _require_nvenc_gpu()

    if float(trim_start_sec) <= 0.0001:
        if raw_path != working_path:
            shutil.copy2(raw_path, working_path)
        return str(working_path)

    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-hwaccel_output_format", "cuda",
        "-ss", f"{float(trim_start_sec):.3f}",
        "-i", str(raw_path),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "h264_nvenc",
        "-preset", "p1",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(working_path),
    ]
    _run_ffmpeg(cmd)
    return str(working_path)


def export_review_clips(timeline: RallyTimeline, *, working_video_path: str, review_clips_dir: str) -> dict[str, str]:
    review_dir = Path(review_clips_dir).resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    src = str(Path(working_video_path).resolve())
    _require_nvenc_gpu()

    def _export_one(point) -> tuple[str, str]:
        clip_path = review_dir / f"{point.id}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-ss", f"{float(point.t_start):.3f}",
            "-to", f"{float(point.t_end):.3f}",
            "-i", src,
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "h264_nvenc",
            "-preset", "p1",
            "-c:a", "copy",
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
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required: torch.cuda.is_available() returned False for winner prediction.")
        torch.cuda.set_device(0)
        self._torch = torch
        self._parse_prediction_json = parse_prediction_json
        self.prompt_text = build_training_prompt(list(taxonomy_order))
        self.processor = AutoProcessor.from_pretrained(config.base_model_dir)
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.base_model_dir,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
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
    _job_log(job_dir, "Step 1/5: trim_input — cutting working video with GPU encoder")
    trim_input_video(job.raw_video_path, job.artifacts.working_video_path, job.trim_start_sec)
    _job_log(job_dir, "Step 1/5: trim_input — done")

    # Step 2: player identification — runs BEFORE rally detection so player names
    # are available for all downstream steps.  Also detects the table ROI once
    # so Step 3 can reuse it (the table-ROI detector is run here even when
    # identification itself is skipped, because Step 3 always needs the ROI).
    _check_stop()
    update_job_runtime_state(job, status="running", current_step="player_identification")
    _names_already_set = bool(job.player_a_name.strip() and job.player_b_name.strip())
    table_roi = None   # shared across Step 2 → Step 3 to avoid duplicate detection

    if _names_already_set:
        _job_log(
            job_dir,
            f"Step 2/5: identify_players — names already provided ({job.player_a_name!r}, {job.player_b_name!r}); "
            "skipping face identification but still detecting table ROI for Step 3",
        )
        try:
            from backend.player_identification import detect_table_roi_and_player_zone
            table_roi, _zone = detect_table_roi_and_player_zone(
                job.artifacts.working_video_path, config.table_weights_path,
            )
            if table_roi is not None:
                _job_log(
                    job_dir,
                    f"Step 2/5: identify_players — table ROI: x={table_roi.x} y={table_roi.y} "
                    f"w={table_roi.w} h={table_roi.h}",
                )
            else:
                _job_log(job_dir, "Step 2/5: identify_players — table ROI detection failed")
        except Exception as exc:
            _job_log(job_dir, f"Step 2/5: identify_players — table ROI detection FAILED: {exc}")
    else:
        _job_log(job_dir, "Step 2/5: identify_players — scanning face DB for known players")
        try:
            from backend.player_identity import FaceDB
            from backend.player_identification import quick_identify_players_standalone

            face_db_path = PROJECT_ROOT / "data" / "players" / "faces.json"
            face_db = FaceDB(face_db_path)
            id_result = quick_identify_players_standalone(
                job.artifacts.working_video_path,
                config.pose_weights_path,
                face_db,
                table_weights_path=config.table_weights_path,
                log_fn=lambda msg: _job_log(job_dir, msg),
            )
            table_roi = id_result.table_roi
            if len(face_db) == 0:
                _job_log(job_dir, "Step 2/5: identify_players — face DB is empty, no names resolved")
            else:
                resolved_near = id_result.near_name
                resolved_far = id_result.far_name
                if resolved_near is not None or resolved_far is not None:
                    if job.player_a_starts_near:
                        if resolved_near is not None:
                            job.player_a_name = resolved_near
                        if resolved_far is not None:
                            job.player_b_name = resolved_far
                    else:
                        if resolved_far is not None:
                            job.player_a_name = resolved_far
                        if resolved_near is not None:
                            job.player_b_name = resolved_near
                    save_match_job(job)
                    _job_log(
                        job_dir,
                        f"Step 2/5: identify_players — NEAR={resolved_near!r} FAR={resolved_far!r}"
                        f" → player_a={job.player_a_name!r} player_b={job.player_b_name!r}"
                        f" (status={id_result.status})",
                    )
                else:
                    _job_log(job_dir, f"Step 2/5: identify_players — no faces matched (status={id_result.status}), keeping user names")
        except Exception as exc:
            _job_log(job_dir, f"Step 2/5: identify_players — FAILED: {exc} — keeping user-provided names")

    _check_stop()
    update_job_runtime_state(job, status="running", current_step="generate_rally_timeline")
    _job_log(job_dir, "Step 3/5: detect_rallies — loading YOLO models and detecting rallies (slowest step)")
    build_rally_timeline = _load_build_rally_timeline()
    if table_roi is not None:
        _job_log(job_dir, "Step 3/5: detect_rallies — reusing table ROI from Step 2 (skipping re-detection)")
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
        table_roi=table_roi,
        log_fn=lambda msg: _job_log(job_dir, msg),
    )
    timeline.video_path = str(Path(job.artifacts.working_video_path).resolve()).replace("\\", "/")
    _job_log(job_dir, f"Step 3/5: detect_rallies — done, {len(timeline.points)} rallies detected")

    _check_stop()
    _job_log(job_dir, "Step 3/5: detect_rallies (sampling player positions) — sampling YOLO X positions for set-boundary Signal 3")
    try:
        populate_player_positions(
            timeline,
            job.artifacts.working_video_path,
            config.pose_weights_path,
        )
        n_pos = sum(1 for p in timeline.points if p.player_a_mean_x is not None)
        _job_log(job_dir, f"Step 3/5: detect_rallies (sampling player positions) — done, {n_pos}/{len(timeline.points)} rallies have position data")
    except Exception as exc:
        _job_log(job_dir, f"Step 3/5: detect_rallies (sampling player positions) — FAILED: {exc} — set boundary will use Signal 1+2 only")

    save_rally_timeline(Path(job.artifacts.timeline_json_path), timeline)
    if not timeline.points:
        raise RuntimeError(
            "No rallies detected in this video. "
            "Check that the video contains table tennis gameplay and the table is clearly visible."
        )

    _check_stop()
    update_job_runtime_state(job, status="running", current_step="export_review_clips", timeline=timeline)
    _job_log(job_dir, f"Step 4/5: export_clips — cutting {len(timeline.points)} clips in parallel")
    review_clips = export_review_clips(
        timeline,
        working_video_path=job.artifacts.working_video_path,
        review_clips_dir=job.artifacts.review_clips_dir,
    )
    _job_log(job_dir, f"Step 4/5: export_clips — done, {len(review_clips)} clips")

    _check_stop()
    update_job_runtime_state(job, status="running", current_step="predict_winners_with_adapter", timeline=timeline)
    _job_log(job_dir, "Step 5/5: predict_winners — loading Qwen3-VL adapter")
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
    _job_log(job_dir, "Step 5/5: predict_winners — done")

    review_status = build_review_status(timeline)
    next_status = "ready_for_final" if review_status["final_export_ready"] else "needs_review"
    update_job_runtime_state(job, status=next_status, current_step="ai_ready", timeline=timeline)
    _job_log(job_dir, f"Pipeline complete — {len(timeline.points)} rallies ready for review, status={next_status}")
    return job


# ---------------------------------------------------------------------------
# Staged pipeline — runs one stage at a time, pauses for operator confirmation
# between stages.  The GUI shows output + "Next" button at each pause point.
#
# Stage flow:
#   stage_trim_and_identify → status "awaiting_confirmation" / step "confirm_players"
#   stage_detect_sets        → status "awaiting_confirmation" / step "confirm_sets"
#   stage_detect_rallies     → status "awaiting_confirmation" / step "confirm_rallies"
#   stage_predict_winners    → status "needs_review"
# ---------------------------------------------------------------------------

def run_pipeline_stage_trim_and_detect_sets(
    job_or_path: MatchJob | str | Path,
    *,
    config: ProductionPipelineConfig | None = None,
) -> MatchJob:
    """Stage 1 + 3.1: trim video, detect table ROI, detect side swaps.

    Player names are already confirmed by the operator in the setup form
    BEFORE this stage runs — no confirm_players pause needed.

    Pauses at confirm_sets for operator to verify set count + swap times.
    """
    config = config or ProductionPipelineConfig()
    job = _load_or_raise_job(job_or_path)
    job_dir = job.artifacts.job_dir

    # Step 1: trim
    update_job_runtime_state(job, status="running", current_step="trim_input", error_message="")
    _job_log(job_dir, "Step 1/5: trim_input — cutting working video with GPU encoder")
    trim_input_video(job.raw_video_path, job.artifacts.working_video_path, job.trim_start_sec)
    _job_log(job_dir, "Step 1/5: trim_input — done")
    _job_log(job_dir, "Legacy trim+detect entrypoint — delegating Step 3.1 to trusted Step 2 side-swap detector")
    return run_pipeline_stage_detect_sets(job, config=config)

    # Recover or detect table ROI.  If Step 2 (identification scan) already
    # detected it, the ROI is stored in job.timeline_summary["table_roi"] —
    # reuse it instead of running YOLOv8x-table again.
    update_job_runtime_state(job, status="running", current_step="detect_sets")
    table_roi = None
    roi_data = job.timeline_summary.get("table_roi")
    if roi_data and roi_data.get("w", 0) > 0:
        from backend.ai_table_roi import TableROI
        table_roi = TableROI(
            x=int(roi_data["x"]), y=int(roi_data["y"]),
            w=int(roi_data["w"]), h=int(roi_data["h"]),
            confidence=float(roi_data.get("confidence", 1.0)),
        )
        _job_log(job_dir, f"Step 3.1: detect_sets — reusing table ROI from Step 2: "
                 f"x={table_roi.x} y={table_roi.y} w={table_roi.w} h={table_roi.h}")
    else:
        try:
            from backend.player_identification import detect_table_roi_and_player_zone
            _job_log(job_dir, "Step 3.1: detect_sets — detecting table ROI (not cached from Step 2)")
            table_roi, _zone = detect_table_roi_and_player_zone(
                job.artifacts.working_video_path, config.table_weights_path,
            )
            if table_roi is not None:
                _job_log(job_dir, f"Step 3.1: detect_sets — table ROI: x={table_roi.x} y={table_roi.y} w={table_roi.w} h={table_roi.h}")
                job.timeline_summary["table_roi"] = {
                    "x": table_roi.x, "y": table_roi.y,
                    "w": table_roi.w, "h": table_roi.h,
                    "confidence": table_roi.confidence,
                }
            else:
                _job_log(job_dir, "Step 3.1: detect_sets — table ROI detection FAILED")
        except Exception as exc:
            _job_log(job_dir, f"Step 3.1: detect_sets — table ROI FAILED: {exc}")

    # Now run side-swap detection (reuses code from detect_side_swap.py)
    _job_log(job_dir, "Step 3.1: detect_sets — detecting side swaps to find set boundaries")

    import sys as _sys
    if str(SCRIPTS_DIR) not in _sys.path:
        _sys.path.append(str(SCRIPTS_DIR))
    from detect_side_swap import (
        sample_positions, smoothed_side, baseline_state,
        find_swap, refine_swap_to_transition_window,
        classify_side, SIDE_L, SIDE_R,
    )
    from backend.player_identity import FaceDB, FaceEmbedder, face_similarity
    from backend.player_identification import (
        _detect_bodies_and_faces, _try_embed_face, DEFAULT_MATCH_THRESHOLD,
    )

    video_path = job.artifacts.working_video_path

    # Derive player zone from table ROI
    if table_roi is not None and table_roi.w > 0:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        bw, bh = float(table_roi.w), float(table_roi.h)
        table_center_x = table_roi.x + table_roi.w / 2.0
        player_zone = (
            max(0.0, table_roi.x - bw * 0.40),
            max(0.0, table_roi.y - bh * 1.00),
            min(float(frame_w), table_roi.x + table_roi.w + bw * 0.40),
            min(float(frame_h), table_roi.y + table_roi.h + bh * 1.00),
        )

        face_db = FaceDB(PROJECT_ROOT / "data" / "players" / "faces.json")
        face_model_path = PROJECT_ROOT / "data" / "models" / "face" / "w600k_r50.onnx"
        embedder = FaceEmbedder(face_model_path)
        from ultralytics import YOLO
        yolo = YOLO(str(config.pose_weights_path))

        _job_log(job_dir, "Step 3.1: detect_sets — sampling player positions")
        records = sample_positions(
            str(video_path), yolo, embedder, face_db, player_zone, table_center_x,
            sample_step=2.0, match_threshold=DEFAULT_MATCH_THRESHOLD,
        )

        from collections import Counter
        identity_counts = Counter(r["identity"] for r in records if r["identity"])
        top2 = identity_counts.most_common(2)

        swaps_info: list[dict] = []
        n_sets = 1
        if len(top2) >= 2:
            name_a, _ = top2[0]
            name_b, _ = top2[1]
            tl_a = [(r["t"], r["side"]) for r in records if r["identity"] == name_a]
            tl_b = [(r["t"], r["side"]) for r in records if r["identity"] == name_b]
            init_a = baseline_state(tl_a, 10, 60)
            init_b = baseline_state(tl_b, 10, 60)
            duration = max(r["t"] for r in records) if records else 0.0

            if init_a is not None and init_b is not None and init_a != init_b:
                cur_a, cur_b = init_a, init_b
                cursor = 60.0
                while cursor <= duration:
                    result = find_swap(tl_a, tl_b, cursor, duration, 2.0, cur_a, cur_b, 60.0, 15.0)
                    if result is None:
                        break
                    t_swap, mode = result
                    fl_a = SIDE_R if cur_a == SIDE_L else SIDE_L
                    fl_b = SIDE_R if cur_b == SIDE_L else SIDE_L
                    t_bs, t_be = refine_swap_to_transition_window(
                        str(video_path), yolo, player_zone, table_center_x, t_swap,
                        table_roi=table_roi, search_before=90.0, search_after=10.0,
                    )
                    cutoff = t_be if t_be else t_swap
                    swaps_info.append({"t_swap": float(t_swap), "t_break_start": float(t_bs) if t_bs else None,
                                       "t_break_end": float(t_be) if t_be else None, "t_cutoff": float(cutoff), "mode": mode})
                    _job_log(job_dir, f"Step 3.1: detect_sets — swap: break ends at {cutoff:.1f}s (mode={mode})")
                    cur_a, cur_b = fl_a, fl_b
                    cursor = t_swap + 62.0
            else:
                _job_log(job_dir, "Step 3.1: detect_sets — baseline sides ambiguous, assuming 1 set")
        else:
            _job_log(job_dir, f"Step 3.1: detect_sets — only {len(top2)} player(s) identified, assuming 1 set")

        n_sets = len(swaps_info) + 1
        job.timeline_summary["detected_sets"] = {
            "n_sets": n_sets, "swaps": swaps_info, "duration": float(duration),
        }
    else:
        _job_log(job_dir, "Step 3.1: detect_sets — no table ROI, assuming 1 set")
        job.timeline_summary["detected_sets"] = {"n_sets": 1, "swaps": [], "duration": 0.0}
        n_sets = 1

    _job_log(job_dir, f"Step 3.1: detect_sets — {n_sets} set(s) detected")

    # Pause for operator confirmation
    update_job_runtime_state(job, status="awaiting_confirmation", current_step="confirm_sets")
    _job_log(job_dir, "Paused — waiting for operator to confirm set count and swap times")
    save_match_job(job)
    return job


def _run_pipeline_stage_detect_sets_side_swap_v2(
    job_or_path: MatchJob | str | Path,
    *,
    config: ProductionPipelineConfig | None = None,
) -> MatchJob:
    """Stage 3.1: detect side swaps to determine set count and boundaries.
    Pauses for operator confirmation."""
    config = config or ProductionPipelineConfig()
    job = _load_or_raise_job(job_or_path)
    job_dir = job.artifacts.job_dir

    update_job_runtime_state(job, status="running", current_step="detect_sets")
    _job_log(job_dir, "Step 3.1: detect_sets — detecting side swaps to find set boundaries")

    # Import swap detection functions
    import sys as _sys
    if str(SCRIPTS_DIR) not in _sys.path:
        _sys.path.append(str(SCRIPTS_DIR))
    from detect_side_swap import (
        detect_table_break_candidates,
        detect_rally_anchor_side_swaps,
        dominant_side_in_range,
        infer_opposite_side,
        sample_positions,
        validate_known_player_swaps,
    )
    from backend.player_identity import FaceDB, FaceEmbedder
    from backend.player_identification import detect_table_roi_and_player_zone, DEFAULT_MATCH_THRESHOLD

    video_path = job.artifacts.working_video_path
    player_a_name = str(job.player_a_name).strip()
    player_b_name = str(job.player_b_name).strip()
    if not player_a_name or not player_b_name or player_a_name.lower() == "unknown" or player_b_name.lower() == "unknown":
        _job_log(job_dir, "Step 3.1: detect_sets — Step 2 player names are incomplete; pausing instead of guessing")
        job.timeline_summary["detected_sets"] = {
            "n_sets": 1,
            "swaps": [],
            "note": "step2_player_names_incomplete",
            "step2_ground_truth_required": True,
        }
        update_job_runtime_state(job, status="awaiting_confirmation", current_step="confirm_sets")
        save_match_job(job)
        return job

    # Recover table ROI from job metadata (saved by stage 1+2)
    table_roi = None
    roi_data = job.timeline_summary.get("table_roi")
    if roi_data:
        from backend.ai_table_roi import TableROI
        table_roi = TableROI(
            x=int(roi_data["x"]), y=int(roi_data["y"]),
            w=int(roi_data["w"]), h=int(roi_data["h"]),
            confidence=float(roi_data.get("confidence", 1.0)),
        )

    if table_roi is None:
        _job_log(job_dir, "Step 3.1: detect_sets — detecting table ROI (not cached)")
        table_roi, _zone = detect_table_roi_and_player_zone(video_path, config.table_weights_path)

    if table_roi is None or table_roi.w <= 0:
        _job_log(job_dir, "Step 3.1: detect_sets — FAILED: table ROI not detected")
        update_job_runtime_state(job, status="awaiting_confirmation", current_step="confirm_sets",
                                 error_message="Table ROI detection failed")
        save_match_job(job)
        return job

    table_center_x = table_roi.x + table_roi.w / 2.0
    _job_log(job_dir, f"Step 3.1: detect_sets — table center x={table_center_x:.0f}")

    # Reuse the Step 2 player zone if available; it was tuned to exclude
    # adjacent-table players.  Derive the same style of zone only as fallback.
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = float(n_frames / fps) if fps > 0 else 0.0
    zone_data = job.timeline_summary.get("player_zone")
    if isinstance(zone_data, dict) and {"x1", "y1", "x2", "y2"}.issubset(zone_data.keys()):
        player_zone = (
            float(zone_data["x1"]),
            float(zone_data["y1"]),
            float(zone_data["x2"]),
            float(zone_data["y2"]),
        )
        _job_log(job_dir, "Step 3.1: detect_sets — reusing Step 2 player zone")
    else:
        bw, bh = float(table_roi.w), float(table_roi.h)
        player_zone = (
            max(0.0, table_roi.x - bw * 0.30),
            max(0.0, table_roi.y - bh * 1.10),
            min(float(frame_w), table_roi.x + table_roi.w + bw * 0.30),
            min(float(frame_h), table_roi.y + table_roi.h + bh * 1.10),
        )
        _job_log(job_dir, "Step 3.1: detect_sets — derived fallback player zone from Step 2 table ROI")

    existing_ground_truth = job.timeline_summary.get("step2_ground_truth", {})
    if not isinstance(existing_ground_truth, dict):
        existing_ground_truth = {}
    job.timeline_summary["step2_ground_truth"] = {
        "source": existing_ground_truth.get("source", "identify_players"),
        "scan_id": job.timeline_summary.get("identify_scan_id", ""),
        "player_a": {"name": player_a_name, "initial_role": "near", "starts_near": True},
        "player_b": {"name": player_b_name, "initial_role": "far", "starts_near": False},
        "trusted": True,
    }

    break_candidates: list[dict] = []

    # Build rough full-video rally anchors using the same well-debugged detector
    # that Step 3.2 uses per set.  These anchors are only for side-swap search.
    try:
        rough_timeline_path = Path(job_dir) / "side_swap_rally_proposals.json"
        if rough_timeline_path.exists():
            _job_log(job_dir, "Step 3.1: detect_sets — reusing cached rough full-video rally anchors")
            rough_timeline = load_rally_timeline(rough_timeline_path)
        else:
            build_rally_timeline = _load_build_rally_timeline()
            _job_log(job_dir, "Step 3.1: detect_sets — building rough full-video rally anchors")
            rough_timeline = build_rally_timeline(
                str(video_path),
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
                table_roi=table_roi,
                log_fn=lambda msg: _job_log(job_dir, msg),
            )
            save_rally_timeline(rough_timeline_path, rough_timeline)
        rough_points = sorted(list(rough_timeline.points), key=lambda point: float(point.t_start))
        rough_scoring = sum(1 for point in rough_points if counts_toward_score(point))
        rough_lets = len(rough_points) - rough_scoring
        if rough_points:
            duration = max(duration, max(float(point.t_end) for point in rough_points))
        rough_summary = {
            "path": str(rough_timeline_path).replace("\\", "/"),
            "total": len(rough_points),
            "scoring": rough_scoring,
            "lets": rough_lets,
        }
        _job_log(
            job_dir,
            f"Step 3.1: detect_sets — rough anchors: {rough_scoring} scoring + {rough_lets} LETs = {len(rough_points)} total",
        )
    except Exception as exc:
        _job_log(job_dir, f"Step 3.1: detect_sets — rough rally anchor detection FAILED: {exc}")
        job.timeline_summary["detected_sets"] = {
            "n_sets": 1,
            "swaps": [],
            "note": "rough_rally_anchor_detection_failed",
            "error": str(exc),
            "duration": duration,
            "algorithm": "rally_anchor_side_scan_v2",
        }
        update_job_runtime_state(job, status="awaiting_confirmation", current_step="confirm_sets")
        save_match_job(job)
        return job

    # Load models
    _job_log(job_dir, "Step 3.1: detect_sets — loading models for swap detection")
    face_db = FaceDB(PROJECT_ROOT / "data" / "players" / "faces.json")
    face_model_path = PROJECT_ROOT / "data" / "models" / "face" / "w600k_r50.onnx"
    embedder = FaceEmbedder(face_model_path)
    db_names = {rec.name for rec in face_db.records}
    missing_db = [name for name in [player_a_name, player_b_name] if name not in db_names]
    if missing_db:
        _job_log(job_dir, f"Step 3.1: detect_sets — Step 2 player(s) missing from FaceDB: {missing_db}; pausing without guessing")
        job.timeline_summary["detected_sets"] = {
            "n_sets": 1,
            "swaps": [],
            "note": "step2_player_missing_from_face_db",
            "missing_players": missing_db,
            "break_candidates": break_candidates,
            "duration": duration,
        }
        update_job_runtime_state(job, status="awaiting_confirmation", current_step="confirm_sets")
        save_match_job(job)
        return job
    from ultralytics import YOLO
    yolo = YOLO(str(config.pose_weights_path))

    # Sample positions
    _job_log(job_dir, f"Step 3.1: detect_sets — sampling trusted Step 2 players: {player_a_name} vs {player_b_name}")
    records = sample_positions(
        str(video_path), yolo, embedder, face_db, player_zone, table_center_x,
        sample_step=2.0, match_threshold=DEFAULT_MATCH_THRESHOLD,
    )

    from collections import Counter
    identity_counts = Counter(r["identity"] for r in records if r["identity"])
    _job_log(
        job_dir,
        "Step 3.1: detect_sets — trusted identity samples: "
        f"{player_a_name}={identity_counts.get(player_a_name, 0)}, "
        f"{player_b_name}={identity_counts.get(player_b_name, 0)}",
    )

    tl_a = [(r["t"], r["side"]) for r in records if r["identity"] == player_a_name]
    tl_b = [(r["t"], r["side"]) for r in records if r["identity"] == player_b_name]

    init_source = "early_window_10_60"
    init_anchor_window = None
    if rough_points:
        first_anchor = rough_points[0]
        first_start = float(first_anchor.active_start if first_anchor.active_start is not None else first_anchor.t_start)
        first_end = float(first_anchor.active_end if first_anchor.active_end is not None else first_anchor.t_end)
        init_anchor_window = {
            "point_id": str(first_anchor.id),
            "lo": max(0.0, first_start - 3.0),
            "hi": max(first_end, first_start + 6.0),
        }
        init_a_ev = dominant_side_in_range(
            tl_a,
            float(init_anchor_window["lo"]),
            float(init_anchor_window["hi"]),
            min_samples=1,
            min_majority_frac=0.55,
        )
        init_b_ev = dominant_side_in_range(
            tl_b,
            float(init_anchor_window["lo"]),
            float(init_anchor_window["hi"]),
            min_samples=1,
            min_majority_frac=0.55,
        )
        init_source = "rally1_anchor_window"
    else:
        init_a_ev = {"side": None, "samples": 0, "majority_frac": 0.0}
        init_b_ev = {"side": None, "samples": 0, "majority_frac": 0.0}

    fallback_a_ev = dominant_side_in_range(tl_a, 10.0, 60.0, min_samples=1)
    fallback_b_ev = dominant_side_in_range(tl_b, 10.0, 60.0, min_samples=1)
    if init_a_ev["side"] is None and init_b_ev["side"] is None:
        init_a_ev = fallback_a_ev
        init_b_ev = fallback_b_ev
        init_source = "early_window_10_60"
    elif init_a_ev["side"] is None and fallback_a_ev["side"] is not None:
        init_a_ev = fallback_a_ev
        init_source = f"{init_source}+player_a_early_window_10_60"
    elif init_b_ev["side"] is None and fallback_b_ev["side"] is not None:
        init_b_ev = fallback_b_ev
        init_source = f"{init_source}+player_b_early_window_10_60"

    init_a = init_a_ev["side"]
    init_b = init_b_ev["side"]
    if init_a is not None and init_b is None:
        init_b = infer_opposite_side(init_a)
    elif init_b is not None and init_a is None:
        init_a = infer_opposite_side(init_b)

    if init_a is None or init_b is None or init_a == init_b:
        _job_log(job_dir, "Step 3.1: detect_sets — cannot infer initial L/R sides for trusted Step 2 players; pausing")
        job.timeline_summary["detected_sets"] = {
            "n_sets": 1,
            "swaps": [],
            "note": "trusted_players_initial_side_ambiguous",
            "break_candidates": break_candidates,
            "rough_rallies": rough_summary,
            "side_samples": {
                "player_a": {"name": player_a_name, "count": len(tl_a), "initial": init_a_ev},
                "player_b": {"name": player_b_name, "count": len(tl_b), "initial": init_b_ev},
                "initial_source": init_source,
                "rally1_window": init_anchor_window,
                "fallback_10_60": {"player_a": fallback_a_ev, "player_b": fallback_b_ev},
            },
            "duration": duration,
            "algorithm": "rally_anchor_side_scan_v2",
        }
        update_job_runtime_state(job, status="awaiting_confirmation", current_step="confirm_sets")
        save_match_job(job)
        return job

    _job_log(
        job_dir,
        f"Step 3.1: detect_sets — initial trusted sides: "
        f"{player_a_name}={init_a}, {player_b_name}={init_b}",
    )

    side_scan = detect_rally_anchor_side_swaps(
        rough_points,
        tl_a,
        tl_b,
        init_a=str(init_a),
        init_b=str(init_b),
        player_a_name=player_a_name,
        player_b_name=player_b_name,
        best_of=job.best_of,
        log_fn=lambda msg: _job_log(job_dir, msg),
    )
    swaps_info = list(side_scan.get("swaps", []))
    mid_set_swaps = list(side_scan.get("mid_set_swaps", []))

    first_primary_cutoff = min((float(s.get("t_cutoff", 0.0)) for s in swaps_info), default=float("inf"))
    long_rough_anchors: list[dict] = []
    for point in rough_points:
        point_start = float(point.t_start)
        point_end = float(point.t_end)
        point_duration = point_end - point_start
        if point_start >= first_primary_cutoff:
            continue
        if point_duration >= 18.0:
            long_rough_anchors.append(
                {
                    "id": str(point.id),
                    "t_start": point_start,
                    "t_end": point_end,
                    "duration": point_duration,
                    "flags": list(point.flags),
                }
            )

    repair_swaps: list[dict] = []
    repair_diagnostics: dict[str, Any] = {
        "needed": bool(long_rough_anchors) or not swaps_info,
        "used": False,
        "reason": "long_rough_anchor_before_primary_swap" if long_rough_anchors else (
            "no_primary_rally_anchor_swap" if not swaps_info else ""
        ),
        "long_rough_anchors": long_rough_anchors,
    }

    if repair_diagnostics["needed"]:
        # Table-motion break windows are a repair signal only. They do not decide
        # a set boundary unless trusted Step 2 identities also flip sides.
        try:
            _job_log(job_dir, "Step 3.1: detect_sets — collecting table-motion break repair diagnostics")
            break_candidates = detect_table_break_candidates(
                str(video_path),
                table_roi,
                dense_step=0.5,
                min_break_sec=8.0,
                min_start_sec=20.0,
                resume_search_sec=45.0,
                log_fn=lambda msg: _job_log(job_dir, msg),
            )
        except Exception as exc:
            _job_log(job_dir, f"Step 3.1: detect_sets — table-motion diagnostics FAILED: {exc}")
            repair_diagnostics["error"] = str(exc)

        if break_candidates:
            repair_swaps = validate_known_player_swaps(
                break_candidates,
                tl_a,
                tl_b,
                init_a=str(init_a),
                init_b=str(init_b),
                player_a_name=player_a_name,
                player_b_name=player_b_name,
                refine_start_with_last_old=False,
                log_fn=lambda msg: _job_log(job_dir, msg),
            )
            for swap in repair_swaps:
                rough_scoring_before = sum(
                    1
                    for point in rough_points
                    if counts_toward_score(point) and float(point.t_end) <= float(swap.get("t_break_start", 0.0))
                )
                swap["mode"] = "break-repair-rally-anchor"
                swap["source"] = "table_break_identity_repair"
                swap["kind"] = "set_boundary"
                swap["repair"] = {
                    "reason": repair_diagnostics["reason"],
                    "rough_scoring_before_break": rough_scoring_before,
                    "long_rough_anchors_before_primary": long_rough_anchors,
                }

    first_repair_cutoff = min((float(s.get("t_cutoff", 0.0)) for s in repair_swaps), default=float("inf"))
    if repair_swaps and (not swaps_info or first_repair_cutoff + 10.0 < first_primary_cutoff):
        _job_log(
            job_dir,
            "Step 3.1: detect_sets — using identity-confirmed break repair "
            f"({len(repair_swaps)} swap(s)) instead of primary rally-anchor result",
        )
        swaps_info = repair_swaps
        repair_diagnostics["used"] = True
        repair_diagnostics["repair_swap_count"] = len(repair_swaps)
    elif repair_swaps:
        repair_diagnostics["repair_swap_count"] = len(repair_swaps)

    n_sets = len(swaps_info) + 1
    _job_log(job_dir, f"Step 3.1: detect_sets — {n_sets} set(s) detected, {len(swaps_info)} swap(s)")

    # Save results for GUI display + next stage
    detected_sets = {
        "n_sets": n_sets,
        "swaps": swaps_info,
        "mid_set_swaps": mid_set_swaps,
        "duration": float(duration),
        "break_candidates": break_candidates,
        "rough_rallies": rough_summary,
        "algorithm": "rally_anchor_side_scan_v2",
        "side_scan": {
            "checked_anchor_count": side_scan.get("checked_anchor_count", 0),
            "checked_anchors": side_scan.get("checked_anchors", []),
            "total_anchor_count": side_scan.get("total_anchor_count", len(rough_points)),
            "scoring_anchor_count": side_scan.get("scoring_anchor_count", rough_summary.get("scoring", 0)),
            "repair": repair_diagnostics,
        },
        "side_samples": {
            "player_a": {"name": player_a_name, "count": len(tl_a), "initial_side": init_a, "initial": init_a_ev},
            "player_b": {"name": player_b_name, "count": len(tl_b), "initial_side": init_b, "initial": init_b_ev},
            "initial_source": init_source,
            "rally1_window": init_anchor_window,
            "fallback_10_60": {"player_a": fallback_a_ev, "player_b": fallback_b_ev},
        },
    }
    if not swaps_info:
        detected_sets["note"] = "no_rally_anchor_side_swap_detected"
    job.timeline_summary["detected_sets"] = detected_sets
    update_job_runtime_state(job, status="awaiting_confirmation", current_step="confirm_sets")
    _job_log(job_dir, "Paused — waiting for operator to confirm set count and swap times")
    save_match_job(job)
    return job


def run_pipeline_stage_detect_sets(
    job_or_path: MatchJob | str | Path,
    *,
    config: ProductionPipelineConfig | None = None,
) -> MatchJob:
    """Step 3.1: detect total rally/LET starts for full-input review.

    This stage deliberately does not split sets or detect side swaps. The goal
    is to let the operator verify total start-times first.
    """
    config = config or ProductionPipelineConfig()
    job = _load_or_raise_job(job_or_path)
    job_dir = Path(job.artifacts.job_dir)

    update_job_runtime_state(job, status="running", current_step="detect_total_rallies")
    _job_log(job_dir, "Step 3.1: total rally start detection — detecting all rally/LET starts")

    video_path = Path(job.artifacts.working_video_path)

    table_roi = None
    roi_data = job.timeline_summary.get("table_roi")
    if roi_data:
        from backend.ai_table_roi import TableROI
        table_roi = TableROI(
            x=int(roi_data["x"]),
            y=int(roi_data["y"]),
            w=int(roi_data["w"]),
            h=int(roi_data["h"]),
            confidence=float(roi_data.get("confidence", 1.0)),
        )

    if table_roi is None:
        from backend.player_identification import detect_table_roi_and_player_zone
        _job_log(job_dir, "Step 3.1: total rally start detection — detecting table ROI (not cached)")
        table_roi, _zone = detect_table_roi_and_player_zone(video_path, config.table_weights_path)

    if table_roi is None or table_roi.w <= 0:
        _job_log(job_dir, "Step 3.1: total rally start detection — FAILED: table ROI not detected")
        update_job_runtime_state(
            job,
            status="awaiting_confirmation",
            current_step="confirm_total_rallies",
            error_message="Table ROI detection failed",
        )
        save_match_job(job)
        return job

    timeline_path = job_dir / "step3_1_total_rally_timeline.json"
    legacy_cache_path = job_dir / "side_swap_rally_proposals.json"
    events_json_path = job_dir / "step3_1_rally_start_events.json"
    frame_dir = job_dir / "step3_1_rally_start_frames"
    player_context = Step3PlayerContext(
        player_a_name=job.player_a_name,
        player_b_name=job.player_b_name,
        player_a_starts_near=job.player_a_starts_near,
    )
    try:
        result = build_step3_1_rally_start_review(
            video_path=video_path,
            timeline_path=timeline_path,
            events_json_path=events_json_path,
            frame_dir=frame_dir,
            table_roi=table_roi,
            table_weights_path=config.table_weights_path,
            pose_weights_path=config.pose_weights_path,
            best_of=job.best_of,
            stride=config.rally_stride,
            mode=config.rally_mode,
            player_margin_px=config.rally_player_margin_px,
            player_fuse_gain=config.rally_player_fuse_gain,
            player_signal_source=config.rally_player_signal_source,
            ball_fuse_gain=config.rally_ball_fuse_gain,
            ball_signal_source=config.rally_ball_signal_source,
            player_context=player_context,
            legacy_cache_path=legacy_cache_path,
            log_fn=lambda msg: _job_log(job_dir, msg),
        )
    except Exception as exc:
        _job_log(job_dir, f"Step 3.1: total rally start detection — FAILED: {exc}")
        job.timeline_summary["detected_total_rallies"] = {
            "algorithm": "total_rally_start_time_review_v2",
            "error": str(exc),
        }
        update_job_runtime_state(job, status="awaiting_confirmation", current_step="confirm_total_rallies")
        save_match_job(job)
        return job

    events = result.events
    summary = result.summary
    scoring_count = int(summary.get("scoring", 0))
    let_count = int(summary.get("lets", 0))
    needs_review_count = int(summary.get("needs_review", 0))
    _job_log(
        job_dir,
        f"Step 3.1: total rally start detection — detected {len(events)} total "
        f"({scoring_count} scoring + {let_count} LET/non-scoring + "
        f"{needs_review_count} needs-review)",
    )
    first_server = summary.get("first_server") or {}
    if first_server:
        _job_log(
            job_dir,
            "Step 3.1: total rally start detection — first server: "
            f"{first_server.get('server_player_name', 'unknown')} "
            f"(role={first_server.get('starter_role', '-')}, t={float(first_server.get('t_start', 0.0)):.3f}s)",
        )
    _job_log(job_dir, f"Step 3.1: total rally start detection — exported start frames: {summary['start_frames_dir']}")

    job.timeline_summary["detected_total_rallies"] = summary
    job.timeline_summary["detected_sets"] = {
        "n_sets": 1,
        "swaps": [],
        "duration": float(max((float(event["t_end"]) for event in events), default=0.0)),
        "note": "step3_1_total_rally_review_only",
        "algorithm": "total_rally_start_time_review_v2",
    }
    update_job_runtime_state(job, status="awaiting_confirmation", current_step="confirm_total_rallies")
    _job_log(job_dir, "Paused — waiting for operator to review total rally start-time frames")
    save_match_job(job)
    return job


def run_pipeline_stage_detect_rallies(
    job_or_path: MatchJob | str | Path,
    *,
    config: ProductionPipelineConfig | None = None,
) -> MatchJob:
    """Stage 3.2: detect rallies per set using confirmed set boundaries.
    Cuts per-set clips, runs rally detection on each, pauses for confirmation."""
    config = config or ProductionPipelineConfig()
    job = _load_or_raise_job(job_or_path)
    job_dir = job.artifacts.job_dir

    update_job_runtime_state(job, status="running", current_step="detect_rallies")

    video_path = job.artifacts.working_video_path
    sets_data = job.timeline_summary.get("detected_sets", {})
    n_sets = sets_data.get("n_sets", 1)
    swaps = sets_data.get("swaps", [])
    duration = sets_data.get("duration", 0.0)

    # Recover table ROI
    table_roi = None
    roi_data = job.timeline_summary.get("table_roi")
    if roi_data:
        from backend.ai_table_roi import TableROI
        table_roi = TableROI(x=int(roi_data["x"]), y=int(roi_data["y"]),
                             w=int(roi_data["w"]), h=int(roi_data["h"]),
                             confidence=float(roi_data.get("confidence", 1.0)))

    # Build set boundaries
    cutoffs = [s["t_cutoff"] for s in swaps]
    boundaries = [0.0] + cutoffs + [duration + 10]

    build_rally_timeline = _load_build_rally_timeline()
    all_points = []
    per_set_counts: list[dict] = []
    set_clips_dir = Path(job_dir) / "set_clips"
    set_clips_dir.mkdir(parents=True, exist_ok=True)

    for si in range(n_sets):
        t_lo = boundaries[si]
        t_hi = boundaries[si + 1]
        _job_log(job_dir, f"Step 3.2: detect_rallies — Set {si+1}/{n_sets} [{t_lo:.1f}s .. {t_hi:.1f}s]")

        # Cut per-set clip
        clip_path = str(set_clips_dir / f"set{si+1}.mp4")
        cmd = ["ffmpeg", "-y", "-ss", f"{t_lo:.3f}", "-to", f"{t_hi:.3f}",
               "-i", str(video_path), "-c", "copy", clip_path]
        _run_ffmpeg(cmd)

        # Run rally detection on clip with pre-detected table ROI
        timeline = build_rally_timeline(
            clip_path, config.table_weights_path,
            pose_weights_path=config.pose_weights_path,
            best_of=job.best_of, stride=config.rally_stride, mode=config.rally_mode,
            player_margin_px=config.rally_player_margin_px,
            player_fuse_gain=config.rally_player_fuse_gain,
            player_signal_source=config.rally_player_signal_source,
            ball_fuse_gain=config.rally_ball_fuse_gain,
            ball_signal_source=config.rally_ball_signal_source,
            table_roi=table_roi,
            log_fn=lambda msg: _job_log(job_dir, msg),
        )

        # Offset timestamps back to full-video time
        for p in timeline.points:
            p.t_start += t_lo
            p.t_end += t_lo
            if p.active_start is not None:
                p.active_start += t_lo
            if p.active_end is not None:
                p.active_end += t_lo
            p.set_number = si + 1

        n_scoring = sum(1 for p in timeline.points if counts_toward_score(p))
        n_let = len(timeline.points) - n_scoring
        _job_log(job_dir, f"Step 3.2: detect_rallies — Set {si+1}: {n_scoring} scoring + {n_let} LETs = {len(timeline.points)} total")
        per_set_counts.append({"set": si + 1, "scoring": n_scoring, "lets": n_let, "total": len(timeline.points)})
        all_points.extend(timeline.points)

    # Re-number point IDs sequentially across all sets
    for i, p in enumerate(all_points):
        p.id = f"pt_{i+1:04d}"

    # Save merged timeline
    from backend.rally_timeline_contract import RallyTimeline, save_rally_timeline
    merged_timeline = RallyTimeline(
        video_path=str(Path(video_path).resolve()).replace("\\", "/"),
        video_fps=timeline.video_fps if all_points else 30.0,
        best_of=job.best_of,
        created_at=timeline.created_at if all_points else "",
        roi=timeline.roi if all_points else {},
        points=all_points,
        analysis_metadata={"detector_mode": config.rally_mode, "staged_pipeline": True,
                           "per_set_counts": per_set_counts},
    )
    save_rally_timeline(Path(job.artifacts.timeline_json_path), merged_timeline)

    # Save per-set rally counts for GUI display
    job.timeline_summary["per_set_rallies"] = per_set_counts
    job.timeline_summary["total_rallies"] = len(all_points)

    update_job_runtime_state(job, status="awaiting_confirmation", current_step="confirm_rallies", timeline=merged_timeline)
    _job_log(job_dir, f"Paused — waiting for operator to confirm rally counts ({len(all_points)} total across {n_sets} sets)")
    save_match_job(job)
    return job


def run_pipeline_stage_predict(
    job_or_path: MatchJob | str | Path,
    *,
    config: ProductionPipelineConfig | None = None,
    stop_check: "Callable[[], bool] | None" = None,
) -> MatchJob:
    """Stage 4+5: export clips + predict winners.  Runs to completion (no pause)."""
    from typing import Callable

    def _check_stop() -> None:
        if stop_check and stop_check():
            raise RuntimeError("stopped_by_operator")

    config = config or ProductionPipelineConfig()
    job = _load_or_raise_job(job_or_path)
    job_dir = job.artifacts.job_dir
    timeline = load_rally_timeline(Path(job.artifacts.timeline_json_path))

    if not timeline.points:
        raise RuntimeError("No rallies in timeline — cannot predict winners.")

    # Step 4: export clips
    _check_stop()
    update_job_runtime_state(job, status="running", current_step="export_review_clips", timeline=timeline)
    _job_log(job_dir, f"Step 4/5: export_clips — cutting {len(timeline.points)} clips in parallel")
    review_clips = export_review_clips(
        timeline,
        working_video_path=job.artifacts.working_video_path,
        review_clips_dir=job.artifacts.review_clips_dir,
    )
    _job_log(job_dir, f"Step 4/5: export_clips — done, {len(review_clips)} clips")

    # Step 5: predict winners
    _check_stop()
    update_job_runtime_state(job, status="running", current_step="predict_winners_with_adapter", timeline=timeline)
    _job_log(job_dir, "Step 5/5: predict_winners — loading Qwen3-VL adapter")
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
    timeline.score_validation = build_score_validation(timeline, expected_scope="any")
    save_rally_timeline(Path(job.artifacts.timeline_json_path), timeline)
    _job_log(job_dir, "Step 5/5: predict_winners — done")

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
