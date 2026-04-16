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


def run_pipeline_stage_detect_sets(
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
        sample_positions, smoothed_side, baseline_state,
        find_swap, refine_swap_to_transition_window,
        classify_side, SIDE_L, SIDE_R,
    )
    from backend.player_identity import FaceDB, FaceEmbedder, face_similarity
    from backend.player_identification import (
        detect_table_roi_and_player_zone, _detect_bodies_and_faces,
        _try_embed_face, DEFAULT_MATCH_THRESHOLD,
    )

    video_path = job.artifacts.working_video_path

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

    # Derive player zone from table ROI
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    bw, bh = float(table_roi.w), float(table_roi.h)
    player_zone = (
        max(0.0, table_roi.x - bw * 0.40),
        max(0.0, table_roi.y - bh * 1.00),
        min(float(frame_w), table_roi.x + table_roi.w + bw * 0.40),
        min(float(frame_h), table_roi.y + table_roi.h + bh * 1.00),
    )

    # Load models
    _job_log(job_dir, "Step 3.1: detect_sets — loading models for swap detection")
    face_db = FaceDB(PROJECT_ROOT / "data" / "players" / "faces.json")
    face_model_path = PROJECT_ROOT / "data" / "models" / "face" / "w600k_r50.onnx"
    embedder = FaceEmbedder(face_model_path)
    from ultralytics import YOLO
    yolo = YOLO(str(config.pose_weights_path))

    # Sample positions
    _job_log(job_dir, "Step 3.1: detect_sets — sampling player positions across video")
    records = sample_positions(
        str(video_path), yolo, embedder, face_db, player_zone, table_center_x,
        sample_step=2.0, match_threshold=DEFAULT_MATCH_THRESHOLD,
    )

    from collections import Counter
    identity_counts = Counter(r["identity"] for r in records if r["identity"])
    top2 = identity_counts.most_common(2)

    if len(top2) < 2:
        _job_log(job_dir, f"Step 3.1: detect_sets — only {len(top2)} player(s) identified, cannot detect swaps")
        job.timeline_summary["detected_sets"] = {"n_sets": 1, "swaps": [], "note": "insufficient face data"}
        update_job_runtime_state(job, status="awaiting_confirmation", current_step="confirm_sets")
        save_match_job(job)
        return job

    name_a, _ = top2[0]
    name_b, _ = top2[1]
    tl_a = [(r["t"], r["side"]) for r in records if r["identity"] == name_a]
    tl_b = [(r["t"], r["side"]) for r in records if r["identity"] == name_b]
    init_a = baseline_state(tl_a, 10, 60)
    init_b = baseline_state(tl_b, 10, 60)
    duration = max(r["t"] for r in records) if records else 0.0

    if init_a is None or init_b is None or init_a == init_b:
        _job_log(job_dir, "Step 3.1: detect_sets — baseline sides not opposite, assuming 1 set")
        job.timeline_summary["detected_sets"] = {"n_sets": 1, "swaps": [], "note": "baseline ambiguous"}
        update_job_runtime_state(job, status="awaiting_confirmation", current_step="confirm_sets")
        save_match_job(job)
        return job

    # Find all swaps
    swaps_info: list[dict] = []
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
        _job_log(job_dir, f"Step 3.1: detect_sets — swap detected: break ends at {cutoff:.1f}s (mode={mode})")
        cur_a, cur_b = fl_a, fl_b
        cursor = t_swap + 62.0

    n_sets = len(swaps_info) + 1
    _job_log(job_dir, f"Step 3.1: detect_sets — {n_sets} set(s) detected, {len(swaps_info)} swap(s)")

    # Save results for GUI display + next stage
    job.timeline_summary["detected_sets"] = {
        "n_sets": n_sets,
        "swaps": swaps_info,
        "duration": float(duration),
    }
    update_job_runtime_state(job, status="awaiting_confirmation", current_step="confirm_sets")
    _job_log(job_dir, "Paused — waiting for operator to confirm set count and swap times")
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
