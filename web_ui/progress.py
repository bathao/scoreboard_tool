from __future__ import annotations

from datetime import datetime, timezone

from backend.production_jobs import MatchJob


_STEP_ORDER = [
    "created",
    "trim_input",
    "generate_rally_timeline",
    "export_review_clips",
    "predict_winners_with_adapter",
    "ai_ready",
]

# Expected duration in seconds per step (calibrated on match_test.mp4, 2m40s / 14 rallies).
# Used for within-step progress interpolation only.
_STEP_EXPECTED_SEC: dict[str, float] = {
    "trim_input": 8.0,
    "generate_rally_timeline": 310.0,
    "export_review_clips": 25.0,
    "predict_winners_with_adapter": 162.0,
}


def _parse_iso_datetime(raw_value: str) -> datetime | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _elapsed_minutes_label(job: MatchJob | None) -> str:
    if job is None:
        return "-"
    start = _parse_iso_datetime(job.created_at)
    end = _parse_iso_datetime(job.updated_at)
    if start is None:
        return "-"
    if job.status == "running":
        end = datetime.now(timezone.utc)
    elif end is None:
        end = datetime.now(timezone.utc)
    elapsed_sec = max(0, int((end - start).total_seconds()))
    m, s = divmod(elapsed_sec, 60)
    return f"{m} min {s} sec"


def _stage_message(job: MatchJob | None, has_timeline: bool) -> str:
    if job is None:
        return ""
    if job.current_step in {"trim_input", "generate_rally_timeline", "export_review_clips", "predict_winners_with_adapter"}:
        return "Preparing rally clips and AI winner suggestions from the selected match. This page refreshes automatically."
    if job.current_step == "final_export":
        return "Export is running. The final scoreboard video will appear here when ready."
    if has_timeline:
        return "Review the uncertain rallies below, then press Export to render the final scoreboard video."
    return "Create Match Job starts the local pipeline, then this same panel becomes the review workspace."


def _job_progress(job: MatchJob | None, review_status: dict[str, object], has_timeline: bool) -> dict[str, object]:
    if job is None:
        return {
            "percent": 0,
            "label": "Idle",
            "step_label": "Waiting for setup",
            "elapsed_label": "-",
            "rallies_label": "-",
            "resolved_label": "-",
            "pending_label": "-",
        }

    step_map = {
        "created": (4, "Setup ready"),
        "trim_input": (12, "Trimming input video"),
        "player_identification": (18, "Identifying players"),
        "detect_sets": (28, "Detecting sets (side-swap analysis)"),
        "confirm_sets": (30, "Confirm set count"),
        "detect_rallies": (40, "Detecting rallies per set"),
        "confirm_rallies": (50, "Confirm rally counts"),
        "generate_rally_timeline": (40, "Detecting rallies"),
        "export_review_clips": (60, "Cutting rally clips"),
        "predict_winners_with_adapter": (82, "Running winner AI"),
        "ai_ready": (92, "AI pipeline finished — ready for review"),
        "review_updated": (92, "Review in progress"),
        "final_export": (98, "Rendering and exporting final video"),
        "final_export_complete": (100, "Export complete"),
        "failed": (100, "Failed"),
    }
    base_percent, step_label = step_map.get(job.current_step, (8, job.current_step.replace("_", " ").strip() or "Processing"))

    # Within-step interpolation: smooth % increase based on elapsed time vs expected duration.
    if job.status == "running" and job.current_step in _STEP_EXPECTED_SEC and job.step_started_at:
        try:
            step_start_dt = datetime.fromisoformat(job.step_started_at)
            elapsed_sec = max(0.0, (datetime.now(timezone.utc) - step_start_dt).total_seconds())
            expected_sec = _STEP_EXPECTED_SEC[job.current_step]
            # Find the next step's base_percent as the ceiling
            step_idx = _STEP_ORDER.index(job.current_step) if job.current_step in _STEP_ORDER else -1
            next_step = _STEP_ORDER[step_idx + 1] if 0 <= step_idx < len(_STEP_ORDER) - 1 else None
            end_pct = step_map[next_step][0] if next_step and next_step in step_map else base_percent + 10
            # Interpolate, cap at 97% of the range so it never claims done
            ratio = min(0.97, elapsed_sec / max(expected_sec, 1.0))
            base_percent = int(round(base_percent + ratio * (end_pct - base_percent)))
        except Exception:
            pass

    if job.status in {"needs_review", "ready_for_final", "completed"} and has_timeline:
        scoring_points = int(review_status.get("scoring_points", 0) or 0)
        resolved_points = int(review_status.get("resolved_scoring_points", 0) or 0)
        if scoring_points > 0:
            review_percent = resolved_points / max(scoring_points, 1)
            base_percent = max(base_percent, min(99, int(round(92 + (7 * review_percent)))))
    if job.status == "completed":
        base_percent = 100
    label = {
        "running": "Running",
        "needs_review": "Waiting for review",
        "ready_for_final": "Ready for export",
        "completed": "Completed",
        "failed": "Failed",
        "created": "Created",
    }.get(job.status, job.status.replace("_", " ").title())

    scoring_points = int(review_status.get("scoring_points", 0) or 0)
    non_scoring_points = int(review_status.get("non_scoring_points", 0) or 0)
    total_rallies = scoring_points + non_scoring_points
    resolved_points = int(review_status.get("resolved_scoring_points", 0) or 0)
    pending_points = int(review_status.get("unresolved_scoring_points", 0) or 0)

    return {
        "percent": max(0, min(100, int(base_percent))),
        "label": label,
        "step_label": step_label,
        "elapsed_label": _elapsed_minutes_label(job),
        "rallies_label": str(total_rallies) if total_rallies > 0 else "-",
        "resolved_label": str(resolved_points) if total_rallies > 0 else "-",
        "pending_label": str(pending_points) if total_rallies > 0 else "-",
    }
