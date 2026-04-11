from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any

from backend.config import PROJECT_ROOT
from backend.rally_timeline_contract import Correction, RallyTimeline, RallyTimelinePoint, counts_toward_score


JOB_SCHEMA_VERSION = "local_match_job_v1"
DEFAULT_JOBS_ROOT = PROJECT_ROOT / "runtime_jobs"
KNOWN_WINNERS = {"player_a", "player_b"}
LET_FLAGS = {"let_no_score", "rally_label_let"}
VALID_JOB_PURPOSES = {"output_only", "output_and_dataset"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_path_str(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("\\", "/")


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    cleaned = cleaned.strip(".-_")
    return cleaned or "match"


def parse_timecode_to_seconds(raw_value: str) -> float:
    raw = str(raw_value or "").strip()
    if not raw:
        return 0.0
    if ":" not in raw:
        value = float(raw)
        if value < 0:
            raise ValueError("trim_start_sec must be non-negative")
        return value

    parts = [part.strip() for part in raw.split(":")]
    if len(parts) not in {2, 3}:
        raise ValueError(f"Unsupported timecode format: {raw_value}")
    if any(part == "" for part in parts):
        raise ValueError(f"Unsupported timecode format: {raw_value}")

    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Unsupported timecode format: {raw_value}") from exc

    if len(values) == 2:
        minutes, seconds = values
        total = (minutes * 60.0) + seconds
    else:
        hours, minutes, seconds = values
        total = (hours * 3600.0) + (minutes * 60.0) + seconds
    if total < 0:
        raise ValueError("trim_start_sec must be non-negative")
    return total


def format_seconds_mmss(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    minutes = total // 60
    secs = total % 60
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class MatchJobArtifacts:
    job_dir: str
    job_json: str
    raw_video_path: str
    working_video_path: str
    timeline_json_path: str
    preview_video_path: str
    final_video_path: str
    review_clips_dir: str
    predictions_jsonl_path: str

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "MatchJobArtifacts":
        return MatchJobArtifacts(
            job_dir=str(data.get("job_dir", "")),
            job_json=str(data.get("job_json", "")),
            raw_video_path=str(data.get("raw_video_path", "")),
            working_video_path=str(data.get("working_video_path", "")),
            timeline_json_path=str(data.get("timeline_json_path", "")),
            preview_video_path=str(data.get("preview_video_path", "")),
            final_video_path=str(data.get("final_video_path", "")),
            review_clips_dir=str(data.get("review_clips_dir", "")),
            predictions_jsonl_path=str(data.get("predictions_jsonl_path", "")),
        )


@dataclass
class MatchJob:
    schema_version: str
    job_id: str
    created_at: str
    updated_at: str
    status: str
    current_step: str
    error_message: str
    raw_video_path: str
    trim_start_sec: float
    player_a_name: str
    player_b_name: str
    best_of: int
    job_purpose: str
    artifacts: MatchJobArtifacts
    timeline_summary: dict[str, Any] = field(default_factory=dict)
    review_status: dict[str, Any] = field(default_factory=dict)
    step_started_at: str = ""
    tournament_name: str = ""
    round_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = asdict(self.artifacts)
        return payload

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "MatchJob":
        return MatchJob(
            schema_version=str(data.get("schema_version", JOB_SCHEMA_VERSION)),
            job_id=str(data.get("job_id", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            status=str(data.get("status", "created")),
            current_step=str(data.get("current_step", "")),
            error_message=str(data.get("error_message", "")),
            raw_video_path=str(data.get("raw_video_path", "")),
            trim_start_sec=float(data.get("trim_start_sec", 0.0)),
            player_a_name=str(data.get("player_a_name", "Player A")),
            player_b_name=str(data.get("player_b_name", "Player B")),
            best_of=int(data.get("best_of", 5)),
            job_purpose=str(data.get("job_purpose", "output_only")),
            artifacts=MatchJobArtifacts.from_dict(dict(data.get("artifacts", {}))),
            timeline_summary=dict(data.get("timeline_summary", {})),
            review_status=dict(data.get("review_status", {})),
            step_started_at=str(data.get("step_started_at", "")),
            tournament_name=str(data.get("tournament_name", "")),
            round_name=str(data.get("round_name", "")),
        )


def ensure_jobs_root(root: Path | None = None) -> Path:
    jobs_root = Path(root) if root is not None else DEFAULT_JOBS_ROOT
    jobs_root.mkdir(parents=True, exist_ok=True)
    return jobs_root


def make_job_id(raw_video_path: str | Path) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = _safe_slug(Path(raw_video_path).stem)
    return f"{timestamp}__{stem}"


def job_dir_from_id(job_id: str, jobs_root: Path | None = None) -> Path:
    return ensure_jobs_root(jobs_root) / str(job_id)


def job_json_path_from_id(job_id: str, jobs_root: Path | None = None) -> Path:
    return job_dir_from_id(job_id, jobs_root) / "job.json"


def create_match_job(
    *,
    raw_video_path: str,
    player_a_name: str,
    player_b_name: str,
    trim_start_sec: float,
    best_of: int,
    job_purpose: str = "output_only",
    tournament_name: str = "",
    round_name: str = "",
    jobs_root: Path | None = None,
) -> MatchJob:
    if best_of <= 0 or best_of % 2 == 0:
        raise ValueError("best_of must be a positive odd number")
    if job_purpose not in VALID_JOB_PURPOSES:
        raise ValueError(f"job_purpose must be one of {sorted(VALID_JOB_PURPOSES)}")

    raw_path = Path(raw_video_path).resolve()
    job_id = make_job_id(raw_path)
    job_dir = job_dir_from_id(job_id, jobs_root)
    job_dir.mkdir(parents=True, exist_ok=True)

    artifacts = MatchJobArtifacts(
        job_dir=_normalize_path_str(job_dir),
        job_json=_normalize_path_str(job_dir / "job.json"),
        raw_video_path=_normalize_path_str(raw_path),
        working_video_path=_normalize_path_str(job_dir / "working_input.mp4"),
        timeline_json_path=_normalize_path_str(job_dir / "timeline_review.json"),
        preview_video_path=_normalize_path_str(job_dir / "preview_scoreboard.mp4"),
        final_video_path=_normalize_path_str(PROJECT_ROOT / "outputs" / f"{job_id}__final_scoreboard.mp4"),
        review_clips_dir=_normalize_path_str(job_dir / "review_clips"),
        predictions_jsonl_path=_normalize_path_str(job_dir / "winner_predictions.jsonl"),
    )
    now = _utc_now_iso()
    job = MatchJob(
        schema_version=JOB_SCHEMA_VERSION,
        job_id=job_id,
        created_at=now,
        updated_at=now,
        status="created",
        current_step="created",
        error_message="",
        raw_video_path=_normalize_path_str(raw_path),
        trim_start_sec=float(trim_start_sec),
        player_a_name=str(player_a_name).strip() or "Player A",
        player_b_name=str(player_b_name).strip() or "Player B",
        best_of=int(best_of),
        job_purpose=str(job_purpose),
        artifacts=artifacts,
        tournament_name=str(tournament_name).strip(),
        round_name=str(round_name).strip(),
    )
    save_match_job(job)
    return job


def save_match_job(job: MatchJob) -> None:
    job.updated_at = _utc_now_iso()
    target = Path(job.artifacts.job_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(job.to_dict(), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def load_match_job(job_path: str | Path) -> MatchJob:
    path = Path(job_path)
    if path.is_dir():
        path = path / "job.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return MatchJob.from_dict(data)


def list_match_jobs(jobs_root: Path | None = None) -> list[MatchJob]:
    root = ensure_jobs_root(jobs_root)
    jobs: list[MatchJob] = []
    for path in sorted(root.glob("*/job.json"), reverse=True):
        try:
            jobs.append(load_match_job(path))
        except Exception:
            continue
    return jobs


def update_job_runtime_state(
    job: MatchJob,
    *,
    status: str | None = None,
    current_step: str | None = None,
    error_message: str | None = None,
    timeline: RallyTimeline | None = None,
) -> MatchJob:
    if status is not None:
        job.status = str(status)
    if current_step is not None and current_step != job.current_step:
        job.current_step = str(current_step)
        job.step_started_at = _utc_now_iso()
    if error_message is not None:
        job.error_message = str(error_message)
    if timeline is not None:
        job.timeline_summary = timeline.build_summary()
        job.review_status = build_review_status(timeline)
    save_match_job(job)
    return job


def effective_rally_winner(point: RallyTimelinePoint) -> str | None:
    """Return the winner to use for scoring: manual correction > AI candidate > None."""
    if point.winner in KNOWN_WINNERS:
        return point.winner
    if point.winner_candidate in KNOWN_WINNERS:
        return point.winner_candidate
    return None


def point_is_review_resolved(point: RallyTimelinePoint) -> bool:
    if not counts_toward_score(point):
        return True
    return point.winner in KNOWN_WINNERS and point.winner_decision == "auto"


def point_needs_human_input(point: RallyTimelinePoint) -> bool:
    """True only when AI has no prediction — human must input winner manually."""
    if not counts_toward_score(point):
        return False
    return effective_rally_winner(point) is None


def build_review_status(timeline: RallyTimeline) -> dict[str, Any]:
    scoring_points = [point for point in timeline.points if counts_toward_score(point)]
    # blocked: no prediction at all — operator must manually enter winner
    blocked_ids = [point.id for point in scoring_points if point_needs_human_input(point)]
    # review: AI has a candidate but confidence is low — operator must confirm or correct
    review_ids = [point.id for point in scoring_points if point.winner_decision == "review"]
    accepted_ids = [point.id for point in scoring_points if point.winner_decision == "auto" and point.winner in KNOWN_WINNERS]
    ai_predicted_ids = [point.id for point in scoring_points if effective_rally_winner(point) is not None]
    # unresolved = any scoring point not yet operator-confirmed (blocked + pending review)
    unresolved_ids = [point.id for point in scoring_points if not point_is_review_resolved(point)]
    return {
        "total_points": len(timeline.points),
        "scoring_points": len(scoring_points),
        "non_scoring_points": sum(1 for point in timeline.points if not counts_toward_score(point)),
        "preview_known_points": len(ai_predicted_ids),
        "resolved_scoring_points": len(accepted_ids),
        "unresolved_scoring_points": len(unresolved_ids),
        "review_points": len(review_ids),
        "blocked_points": len(blocked_ids),
        "final_export_ready": len(unresolved_ids) == 0,
        "preview_render_allowed": len(ai_predicted_ids) > 0,
        "unresolved_point_ids": unresolved_ids,
    }


def _record_change(changes: dict[str, dict[str, Any]], field_name: str, old_value: Any, new_value: Any) -> None:
    if old_value == new_value:
        return
    changes[field_name] = {"from": old_value, "to": new_value}


def _append_correction(point: RallyTimelinePoint, *, reviewer: str, note: str, changes: dict[str, dict[str, Any]]) -> None:
    point.corrections.append(
        Correction(
            at=_utc_now_iso(),
            by=str(reviewer).strip() or "operator",
            changes=changes,
            note=str(note or "").strip(),
        )
    )


def accept_point_prediction(
    timeline: RallyTimeline,
    *,
    point_id: str,
    reviewer: str,
    note: str = "",
) -> RallyTimelinePoint:
    point = get_point_by_id(timeline, point_id)
    winner_value = point.winner if point.winner in KNOWN_WINNERS else point.winner_candidate
    if winner_value not in KNOWN_WINNERS:
        raise ValueError(f"Point {point_id} has no known AI winner to accept")
    return apply_point_review(
        timeline,
        point_id=point_id,
        winner=winner_value,
        reviewer=reviewer,
        note=note or "operator accepted AI winner",
    )


def apply_point_review(
    timeline: RallyTimeline,
    *,
    point_id: str,
    winner: str,
    reviewer: str,
    note: str = "",
) -> RallyTimelinePoint:
    if winner not in KNOWN_WINNERS:
        raise ValueError(f"Unsupported reviewed winner: {winner}")

    point = get_point_by_id(timeline, point_id)
    old_values = {
        "winner": point.winner,
        "winner_candidate": point.winner_candidate,
        "winner_decision": point.winner_decision,
        "winner_confidence": point.winner_confidence,
        "winner_reason": point.winner_reason,
        "winner_loser_candidate": point.winner_loser_candidate,
        "source": point.source,
    }

    point.winner = winner
    point.winner_candidate = winner
    point.winner_loser_candidate = "player_b" if winner == "player_a" else "player_a"
    point.winner_confidence = 1.0
    point.winner_decision = "auto"
    point.winner_reason = "human_review"
    point.source = "human"

    changes: dict[str, dict[str, Any]] = {}
    _record_change(changes, "winner", old_values["winner"], point.winner)
    _record_change(changes, "winner_candidate", old_values["winner_candidate"], point.winner_candidate)
    _record_change(changes, "winner_decision", old_values["winner_decision"], point.winner_decision)
    _record_change(changes, "winner_confidence", old_values["winner_confidence"], point.winner_confidence)
    _record_change(changes, "winner_reason", old_values["winner_reason"], point.winner_reason)
    _record_change(changes, "winner_loser_candidate", old_values["winner_loser_candidate"], point.winner_loser_candidate)
    _record_change(changes, "source", old_values["source"], point.source)
    _append_correction(point, reviewer=reviewer, note=note or "operator reviewed winner", changes=changes)
    return point


def apply_point_no_score(
    timeline: RallyTimeline,
    *,
    point_id: str,
    reviewer: str,
    note: str = "",
) -> RallyTimelinePoint:
    point = get_point_by_id(timeline, point_id)
    old_values = {
        "winner": point.winner,
        "winner_candidate": point.winner_candidate,
        "winner_decision": point.winner_decision,
        "winner_confidence": point.winner_confidence,
        "winner_reason": point.winner_reason,
        "winner_loser_candidate": point.winner_loser_candidate,
        "winner_last_hitter_candidate": point.winner_last_hitter_candidate,
        "source": point.source,
        "flags": list(point.flags),
    }

    point.winner = "unknown"
    point.winner_candidate = "unknown"
    point.winner_loser_candidate = "unknown"
    point.winner_last_hitter_candidate = "unknown"
    point.winner_confidence = 1.0
    point.winner_decision = "auto"
    point.winner_reason = "human_marked_let"
    point.source = "human"
    for flag in sorted(LET_FLAGS):
        if flag not in point.flags:
            point.flags.append(flag)

    changes: dict[str, dict[str, Any]] = {}
    _record_change(changes, "winner", old_values["winner"], point.winner)
    _record_change(changes, "winner_candidate", old_values["winner_candidate"], point.winner_candidate)
    _record_change(changes, "winner_decision", old_values["winner_decision"], point.winner_decision)
    _record_change(changes, "winner_confidence", old_values["winner_confidence"], point.winner_confidence)
    _record_change(changes, "winner_reason", old_values["winner_reason"], point.winner_reason)
    _record_change(changes, "winner_loser_candidate", old_values["winner_loser_candidate"], point.winner_loser_candidate)
    _record_change(
        changes,
        "winner_last_hitter_candidate",
        old_values["winner_last_hitter_candidate"],
        point.winner_last_hitter_candidate,
    )
    _record_change(changes, "source", old_values["source"], point.source)
    _record_change(changes, "flags", old_values["flags"], list(point.flags))
    _append_correction(point, reviewer=reviewer, note=note or "operator marked let / no score", changes=changes)
    return point


def get_point_by_id(timeline: RallyTimeline, point_id: str) -> RallyTimelinePoint:
    for point in timeline.points:
        if point.id == point_id:
            return point
    raise KeyError(f"Unknown point_id: {point_id}")
