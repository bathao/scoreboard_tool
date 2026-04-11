from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import parse_qs, quote_plus

from backend.engine import ScoreEngine
from backend.models import MatchSnapshot, MatchState, RallyEvent
from backend.production_jobs import (
    MatchJob,
    effective_rally_winner,
    format_seconds_mmss,
    job_json_path_from_id,
    load_match_job,
    point_is_review_resolved,
    point_needs_human_input,
)
from backend.production_pipeline import load_job_timeline
from backend.rally_timeline_contract import counts_toward_score


def _respond_html(start_response: Callable, *, body: bytes, status: str = "200 OK"):
    headers = [
        ("Content-Type", "text/html; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-store"),
    ]
    start_response(status, headers)
    return [body]


def _respond_text(start_response: Callable, text: str, *, status: str = "200 OK"):
    body = text.encode("utf-8")
    headers = [
        ("Content-Type", "text/plain; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-store"),
    ]
    start_response(status, headers)
    return [body]


def _respond_json(start_response: Callable, data: object, *, status: str = "200 OK"):
    import json as _json
    body = _json.dumps(data, ensure_ascii=False).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-store"),
    ]
    start_response(status, headers)
    return [body]


def _redirect(start_response: Callable, location: str):
    start_response("302 Found", [("Location", location)])
    return [b""]


def _iter_file(path: Path, chunk_size: int = 64 * 1024) -> Iterable[bytes]:
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            yield chunk


def _serve_file(start_response: Callable, path: Path):
    if not path.exists() or not path.is_file():
        return _respond_text(start_response, "Not found", status="404 Not Found")
    mime, _ = mimetypes.guess_type(str(path))
    headers = [
        ("Content-Type", mime or "application/octet-stream"),
        ("Content-Length", str(path.stat().st_size)),
        ("Cache-Control", "no-store"),
    ]
    start_response("200 OK", headers)
    return _iter_file(path)


def _read_form(environ: dict) -> dict[str, str]:
    length_raw = environ.get("CONTENT_LENGTH", "0") or "0"
    try:
        length = int(length_raw)
    except ValueError:
        length = 0
    body = environ["wsgi.input"].read(length) if length > 0 else b""
    parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    return {key: values[-1] for key, values in parsed.items()}


def _query_params(environ: dict) -> dict[str, str]:
    parsed = parse_qs(str(environ.get("QUERY_STRING", "")), keep_blank_values=True)
    return {key: values[-1] for key, values in parsed.items()}


def _message_context(environ: dict) -> dict[str, object]:
    params = _query_params(environ)
    return {
        "message": params.get("message", ""),
        "error": params.get("kind", "") == "error",
    }


def _job_display_row(job: MatchJob) -> dict[str, object]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "player_a_name": job.player_a_name,
        "player_b_name": job.player_b_name,
        "trim_start_label": format_seconds_mmss(job.trim_start_sec),
        "updated_at": job.updated_at,
        "raw_video_path": job.raw_video_path,
    }


def _winner_label(value: str, job: MatchJob) -> str:
    if value == "player_a":
        return f"{job.player_a_name} (Near)"
    if value == "player_b":
        return f"{job.player_b_name} (Far)"
    return "Unknown"


def _review_status_label(point) -> str:
    if not counts_toward_score(point):
        return "LET / Hong"
    if point_is_review_resolved(point):
        return "Human confirmed"
    if point.winner_decision == "blocked" or point.winner not in {"player_a", "player_b"}:
        return "AI missing winner"
    return "Waiting for operator feedback"


def _review_prompt(point, job: MatchJob) -> str:
    ai_label = _winner_label(point.winner_candidate, job)
    if point.winner_candidate in {"player_a", "player_b"}:
        return f"AI says the winner is {ai_label}. Confirm if correct, or choose the real winner."
    return "AI could not lock the winner for this rally. Choose the real winner to continue."


def _normalize_review_filter(filter_name: str) -> str:
    value = str(filter_name or "pending").strip().lower()
    if value in {"pending", "all"}:
        return value
    return "pending"


def _review_point_rows(job: MatchJob, *, filter_name: str = "pending"):
    timeline = load_job_timeline(job)
    active_filter = _normalize_review_filter(filter_name)
    rows = []
    for point in timeline.points:
        resolved = point_is_review_resolved(point)
        is_non_scoring = not counts_toward_score(point)
        needs_input = point_needs_human_input(point)
        eff_winner = effective_rally_winner(point) if not is_non_scoring else None
        manually_corrected = resolved and bool(point.corrections)
        # pending = any scoring point not yet operator-confirmed (no prediction OR AI predicted but awaiting confirm)
        needs_operator_action = not is_non_scoring and not resolved
        if active_filter == "pending" and not needs_operator_action:
            continue
        rows.append(
            {
                "id": point.id,
                "time_range": f"{point.t_start:.2f}s -> {point.t_end:.2f}s",
                "current_winner_label": _winner_label(point.winner, job),
                "ai_winner_label": _winner_label(point.winner_candidate, job),
                "effective_winner_label": _winner_label(eff_winner, job) if eff_winner else "Unknown",
                "category": point.winner_end_category or "-",
                "source": point.source,
                "decision": point.winner_decision or "",
                "resolved": resolved,
                "needs_input": needs_input,
                "manually_corrected": manually_corrected,
                "can_keep": point.winner in {"player_a", "player_b"} or point.winner_candidate in {"player_a", "player_b"},
                "last_note": point.corrections[-1].note if point.corrections else "",
                "review_status_label": _review_status_label(point),
                "review_prompt": _review_prompt(point, job),
                "clip_src": f"/jobs/{job.job_id}/clips/{point.id}.mp4?ts={job.updated_at}",
                "play_label": f"Playing rally {point.id}",
                "is_non_scoring": is_non_scoring,
                "status_class": (
                    "let" if is_non_scoring
                    else "pending" if needs_input
                    else ("resolved-a" if eff_winner == "player_a" else "resolved-b")
                ),
                "timeline_status_label": (
                    "LET / Hong" if is_non_scoring
                    else "?" if needs_input
                    else (job.player_a_name if eff_winner == "player_a" else job.player_b_name)
                ),
            }
        )
    rows.sort(key=lambda item: item["id"])
    return timeline, rows, active_filter


def _load_selected_job(query: dict[str, str], jobs_root: Path | None = None) -> MatchJob | None:
    job_id = str(query.get("job_id", "")).strip()
    if job_id:
        try:
            return load_match_job(job_json_path_from_id(job_id, jobs_root))
        except Exception:
            return None
    return None


def _resolve_video_file(raw_path: str) -> Path | None:
    candidate = str(raw_path or "").strip()
    if not candidate:
        return None
    path = Path(candidate).expanduser().resolve()
    if not path.exists() or not path.is_file():
        return None
    if path.suffix.lower() not in set(_browser_video_extensions()):
        return None
    return path


def _job_source_video_href(job: MatchJob) -> str:
    return f"/jobs/{job.job_id}/source.mp4?ts={job.updated_at}"


def _job_final_video_href(job: MatchJob) -> str | None:
    if Path(job.artifacts.final_video_path).exists():
        return f"/jobs/{job.job_id}/final.mp4?ts={job.updated_at}"
    return None


def _initial_match_snapshot() -> MatchSnapshot:
    return MatchSnapshot(
        timestamp=0.0,
        set_number=1,
        score_a=0,
        score_b=0,
        sets_a=0,
        sets_b=0,
        is_finished=False,
        winner=None,
    )


def _timeline_score_maps(
    job: MatchJob, timeline
) -> tuple[dict[str, MatchSnapshot], MatchSnapshot, list[tuple[int, int]]]:
    """Returns (score_before_map, final_snapshot, set_scores).

    score_before_map: score state immediately before each rally (keyed by point id).
    final_snapshot: score state after all rallies.
    set_scores: list of (score_a, score_b) for each COMPLETED set, in order.
    Recomputes from scratch every call, so corrections propagate automatically.
    """
    from backend.production_jobs import effective_rally_winner
    engine = ScoreEngine(MatchState(best_of=job.best_of))
    current = _initial_match_snapshot()
    score_before_map: dict[str, MatchSnapshot] = {}
    set_scores: list[tuple[int, int]] = []
    for point in timeline.points:
        score_before_map[point.id] = current
        w = effective_rally_winner(point) if counts_toward_score(point) else None
        if w:
            prev = current
            current = engine.process_event(RallyEvent(winner=w, timestamp=float(point.t_end)))
            # Detect set completion: set number increased or match just finished
            if current.set_number > prev.set_number or (current.is_finished and not prev.is_finished):
                final_a = prev.score_a + (1 if w == "player_a" else 0)
                final_b = prev.score_b + (1 if w == "player_b" else 0)
                set_scores.append((final_a, final_b))
    return score_before_map, current, set_scores


def _point_status_text(row: dict[str, object], job: MatchJob) -> str:
    if bool(row.get("is_non_scoring")):
        return "LET / Hong"
    if bool(row.get("resolved")):
        return _winner_label("player_a" if row.get("current_winner_label") == f"{job.player_a_name} (Near)" else "player_b", job)
    return "Chua duyet"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _raw_matches_root() -> Path:
    repo_root = _repo_root()
    candidates = [
        repo_root / "inputs" / "raw_matches",
        repo_root / "input" / "raw_matches",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return candidates[0]
    return max(existing, key=_count_browseable_videos)


def _count_browseable_videos(root: Path) -> int:
    video_extensions = set(_browser_video_extensions())
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in video_extensions:
            total += 1
    return total


def _browser_video_extensions() -> tuple[str, ...]:
    return (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")


def _resolve_browse_dir(root: Path, requested_relative: str) -> Path:
    root_resolved = root.resolve()
    relative_path = requested_relative.strip().replace("\\", "/").strip("/")
    target = (root_resolved / relative_path).resolve() if relative_path else root_resolved
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Browse path must stay inside {root_resolved}") from exc
    if not target.exists():
        raise ValueError(f"Browse folder not found: {target}")
    if not target.is_dir():
        raise ValueError(f"Browse target is not a folder: {target}")
    return target


def _browse_raw_video_context(root: Path, current_dir: Path) -> dict[str, object]:
    root_resolved = root.resolve()
    current_resolved = current_dir.resolve()
    current_relative = current_resolved.relative_to(root_resolved)
    current_dir_label = "." if str(current_relative) == "." else current_relative.as_posix()
    parent_href = None
    if current_resolved != root_resolved:
        parent_relative = current_resolved.parent.relative_to(root_resolved)
        parent_href = f"/browse/raw-video?path={quote_plus(parent_relative.as_posix() if str(parent_relative) != '.' else '')}"

    entries: list[dict[str, object]] = []
    video_extensions = set(_browser_video_extensions())
    for child in sorted(current_resolved.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        child_relative = child.relative_to(root_resolved)
        child_relative_label = child_relative.as_posix()
        if child.is_dir():
            entries.append(
                {
                    "name": child.name,
                    "kind_label": "Folder",
                    "display_path": child_relative_label,
                    "open_href": f"/browse/raw-video?path={quote_plus(child_relative_label)}",
                    "absolute_path": str(child),
                    "is_dir": True,
                }
            )
            continue
        if child.suffix.lower() not in video_extensions:
            continue
        entries.append(
            {
                "name": child.name,
                "kind_label": "Raw Video",
                "display_path": child_relative_label,
                "open_href": "",
                "absolute_path": str(child),
                "is_dir": False,
            }
        )
    return {
        "entries": entries,
        "current_dir_label": current_dir_label,
        "parent_href": parent_href,
        "raw_matches_root": str(root_resolved),
        "supported_extensions": ", ".join(_browser_video_extensions()),
    }
