from __future__ import annotations

import re
from html import escape
from pathlib import Path
from urllib.parse import quote_plus

from backend.production_jobs import (
    create_match_job,
    job_json_path_from_id,
    load_match_job,
    parse_timecode_to_seconds,
)
from backend.production_pipeline import (
    ProductionPipelineConfig,
    export_job_final_video,
    render_job_preview,
    review_job_point,
    run_initial_job_pipeline,
)

from web_ui.helpers import (
    _browse_raw_video_context,
    _initial_match_snapshot,
    _job_final_video_href,
    _job_source_video_href,
    _load_selected_job,
    _message_context,
    _normalize_review_filter,
    _query_params,
    _raw_matches_root,
    _read_form,
    _redirect,
    _resolve_browse_dir,
    _resolve_video_file,
    _respond_html,
    _respond_json,
    _respond_text,
    _review_point_rows,
    _serve_file,
    _timeline_score_before_map,
)
from web_ui.progress import _job_progress, _stage_message
from web_ui.runner import JobTaskRunner, ThreadingWSGIServer, _cleanup_old_logs, _start_heartbeat_watcher
from web_ui.templates import _render_template

from backend.production_jobs import build_review_status, format_seconds_mmss


def _index_page_context(query: dict[str, str], jobs_root: Path | None, raw_matches_root: Path) -> dict[str, object]:
    current_job = _load_selected_job(query, jobs_root)
    raw_video_value = str(query.get("raw_video_path", "")).strip()
    if current_job is not None:
        raw_video_value = current_job.raw_video_path

    raw_video_path = _resolve_video_file(raw_video_value)
    has_timeline = False
    review_status: dict[str, object] = {
        "final_export_ready": False,
        "unresolved_scoring_points": 0,
        "scoring_points": 0,
    }
    points: list[dict[str, object]] = []
    current_point: dict[str, object] | None = None
    active_filter = _normalize_review_filter(query.get("review_filter", "pending"))
    full_match_src = ""
    full_match_label = ""
    final_video_src = None
    final_video_label = "Playing exported scoreboard video"
    main_video_src = ""
    main_now_playing = ""
    scoreboard = _initial_match_snapshot()
    screen_mode = "setup"
    current_point_id = ""
    current_point_index = 0

    if current_job is not None:
        has_timeline = Path(current_job.artifacts.timeline_json_path).exists()
        if has_timeline:
            screen_mode = "review"
            timeline, all_points, active_filter = _review_point_rows(current_job, filter_name="all")
            pending_points = [row for row in all_points if not bool(row.get("resolved"))]
            current_point_id = str(query.get("current_point", "")).strip()
            if current_point_id:
                current_point = next((row for row in all_points if row["id"] == current_point_id), None)
            if current_point is None:
                current_point = pending_points[0] if pending_points else (all_points[0] if all_points else None)
            if current_point is not None:
                current_point_id = str(current_point["id"])
                current_point_index = next((idx for idx, row in enumerate(all_points) if row["id"] == current_point_id), 0)
            points = pending_points if active_filter == "pending" else all_points
            review_status = build_review_status(timeline)
            score_before_map = _timeline_score_before_map(current_job, timeline)
            if current_point is not None:
                scoreboard = score_before_map.get(current_point_id, scoreboard)
        else:
            review_status = current_job.review_status or review_status

        full_match_src = _job_source_video_href(current_job)
        full_match_label = "Playing trimmed match video"
        final_video_src = _job_final_video_href(current_job)
        if current_point is not None:
            main_video_src = str(current_point["clip_src"])
            main_now_playing = str(current_point["play_label"])
        else:
            main_video_src = full_match_src
            main_now_playing = full_match_label
    elif raw_video_path is not None:
        main_video_src = f"/local-video?path={quote_plus(str(raw_video_path))}"
        main_now_playing = "Playing selected raw video"

    progress = _job_progress(current_job, review_status, has_timeline)

    return {
        "screen_mode": screen_mode,
        "current_job": current_job,
        "raw_video_path_value": raw_video_value,
        "raw_matches_root": str(raw_matches_root),
        "player_a_value": current_job.player_a_name if current_job else "Player A",
        "player_b_value": current_job.player_b_name if current_job else "Player B",
        "trim_start_value": format_seconds_mmss(current_job.trim_start_sec) if current_job else "00:00",
        "best_of_value": current_job.best_of if current_job else 5,
        "has_timeline": has_timeline,
        "review_status": review_status,
        "points": points,
        "current_point": current_point,
        "current_point_id": current_point_id,
        "current_point_index": current_point_index,
        "active_filter": active_filter,
        "scoreboard": scoreboard,
        "progress": progress,
        "stage_message": _stage_message(current_job, has_timeline),
        "main_video_src": main_video_src,
        "main_now_playing": main_now_playing,
        "full_match_src": full_match_src,
        "full_match_label": full_match_label,
        "final_video_src": final_video_src,
        "final_video_label": final_video_label,
    }


def create_local_web_app(
    *,
    config: ProductionPipelineConfig | None = None,
    jobs_root: Path | None = None,
):
    config = config or ProductionPipelineConfig()
    _cleanup_old_logs()
    raw_matches_root = _raw_matches_root()
    raw_matches_root.mkdir(parents=True, exist_ok=True)
    runner = JobTaskRunner(config=config, jobs_root=jobs_root)
    last_beat = _start_heartbeat_watcher(timeout_sec=30.0)

    def app(environ, start_response):
        import time

        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = str(environ.get("PATH_INFO", "/"))
        message_ctx = _message_context(environ)

        if path == "/heartbeat" and method == "POST":
            last_beat[0] = time.monotonic()
            return _respond_text(start_response, "ok")

        if path == "/goodbye" and method == "POST":
            import threading as _threading
            def _delayed_exit():
                time.sleep(0.5)
                import os as _os
                _os._exit(0)
            _threading.Thread(target=_delayed_exit, daemon=True).start()
            return _respond_text(start_response, "bye")

        if path == "/api/job-status" and method == "GET":
            job_id = str(_query_params(environ).get("job_id", "")).strip()
            if not job_id:
                return _respond_json(start_response, {"error": "missing job_id"}, status="400 Bad Request")
            try:
                job = load_match_job(job_json_path_from_id(job_id, jobs_root))
            except Exception:
                return _respond_json(start_response, {"error": "not found"}, status="404 Not Found")
            has_timeline = Path(job.artifacts.timeline_json_path).exists()
            review_status = job.review_status or {}
            progress = _job_progress(job, review_status, has_timeline)
            return _respond_json(start_response, {"status": job.status, "current_step": job.current_step, "progress": progress})

        if path == "/" and method == "GET":
            query = _query_params(environ)
            index_ctx = _index_page_context(query, jobs_root, raw_matches_root)
            current_job = index_ctx.get("current_job")
            body = _render_template(
                "index.html",
                title="Scoreboard Tool Local UI",
                auto_refresh_sec=5 if current_job is not None and current_job.status in {"running", "created"} else None,
                Path=Path,
                **index_ctx,
                **message_ctx,
            )
            return _respond_html(start_response, body=body)

        if path == "/local-video" and method == "GET":
            query = _query_params(environ)
            video_path = _resolve_video_file(query.get("path", ""))
            if video_path is None:
                return _respond_text(start_response, "Video not found", status="404 Not Found")
            return _serve_file(start_response, video_path)

        if path == "/browse/raw-video" and method == "GET":
            query = _query_params(environ)
            requested_path = query.get("path", "")
            try:
                current_dir = _resolve_browse_dir(raw_matches_root, requested_path)
            except Exception as exc:
                return _redirect(start_response, f"/?kind=error&message={quote_plus(str(exc))}")
            body = _render_template(
                "browse_raw_video.html",
                title="Browse Raw Videos",
                auto_refresh_sec=None,
                **_browse_raw_video_context(raw_matches_root, current_dir),
                **message_ctx,
            )
            return _respond_html(start_response, body=body)

        if path == "/jobs" and method == "POST":
            form = _read_form(environ)
            try:
                raw_video_path = str(form.get("raw_video_path", "")).strip()
                if not raw_video_path:
                    raise ValueError("Raw video path is required")
                if not Path(raw_video_path).exists():
                    raise ValueError(f"Raw video not found: {raw_video_path}")
                trim_start_sec = parse_timecode_to_seconds(form.get("trim_start", "0"))
                best_of = int(form.get("best_of", "5"))
                job = create_match_job(
                    raw_video_path=raw_video_path,
                    player_a_name=form.get("player_a_name", "Player A"),
                    player_b_name=form.get("player_b_name", "Player B"),
                    trim_start_sec=trim_start_sec,
                    best_of=best_of,
                    jobs_root=jobs_root,
                )
            except Exception as exc:
                return _redirect(start_response, f"/?kind=error&message={quote_plus(str(exc))}")
            ok, msg = runner.start(
                job.job_id,
                lambda current_job_id: run_initial_job_pipeline(
                    job_json_path_from_id(current_job_id, jobs_root),
                    config=config,
                    stop_check=lambda: runner.is_stop_requested(current_job_id),
                ),
            )
            kind = "info" if ok else "error"
            return _redirect(start_response, f"/?job_id={job.job_id}&kind={kind}&message={quote_plus(msg)}")

        job_match = re.match(r"^/jobs/([^/]+)$", path)
        if job_match and method == "GET":
            job_id = job_match.group(1)
            return _redirect(start_response, f"/?job_id={job_id}")

        stop_match = re.match(r"^/jobs/([^/]+)/stop$", path)
        if stop_match and method == "POST":
            job_id = stop_match.group(1)
            runner.request_stop(job_id)
            from backend.production_pipeline import _job_log, LOGS_DIR
            _job_log(str(LOGS_DIR / job_id), "Stop requested by operator — will stop after current step")
            return _redirect(start_response, f"/?job_id={job_id}&kind=info&message={quote_plus('Stop requested — will stop after current step finishes')}")

        run_match = re.match(r"^/jobs/([^/]+)/run$", path)
        if run_match and method == "POST":
            job_id = run_match.group(1)
            ok, msg = runner.start(
                job_id,
                lambda current_job_id: run_initial_job_pipeline(
                    job_json_path_from_id(current_job_id, jobs_root),
                    config=config,
                ),
            )
            kind = "info" if ok else "error"
            return _redirect(start_response, f"/?job_id={job_id}&kind={kind}&message={quote_plus(msg)}")

        preview_match = re.match(r"^/jobs/([^/]+)/preview$", path)
        if preview_match and method == "POST":
            job_id = preview_match.group(1)
            ok, msg = runner.start(
                job_id,
                lambda current_job_id: render_job_preview(job_json_path_from_id(current_job_id, jobs_root)),
            )
            kind = "info" if ok else "error"
            return _redirect(start_response, f"/?job_id={job_id}&kind={kind}&message={quote_plus(msg)}")

        final_match = re.match(r"^/jobs/([^/]+)/final-export$", path)
        if final_match and method == "POST":
            job_id = final_match.group(1)
            ok, msg = runner.start(
                job_id,
                lambda current_job_id: export_job_final_video(job_json_path_from_id(current_job_id, jobs_root)),
            )
            kind = "info" if ok else "error"
            return _redirect(start_response, f"/?job_id={job_id}&kind={kind}&message={quote_plus(msg)}")

        review_match = re.match(r"^/jobs/([^/]+)/review$", path)
        if review_match and method == "GET":
            job_id = review_match.group(1)
            filter_name = _normalize_review_filter(_query_params(environ).get("filter", "pending"))
            return _redirect(start_response, f"/?job_id={job_id}&review_filter={filter_name}")

        review_point_match = re.match(r"^/jobs/([^/]+)/review/([^/]+)$", path)
        if review_point_match and method == "POST":
            job_id = review_point_match.group(1)
            point_id = review_point_match.group(2)
            form = _read_form(environ)
            filter_name = _normalize_review_filter(form.get("filter", "pending"))
            try:
                job = review_job_point(
                    job_json_path_from_id(job_id, jobs_root),
                    point_id=point_id,
                    action=str(form.get("action", "")),
                    winner=form.get("winner", None),
                    reviewer="local_operator",
                )
            except Exception as exc:
                return _redirect(
                    start_response,
                    f"/?job_id={job_id}&review_filter={quote_plus(filter_name)}&kind=error&message={quote_plus(str(exc))}",
                )
            unresolved_ids = list(job.review_status.get("unresolved_point_ids", [])) if isinstance(job.review_status, dict) else []
            next_point = unresolved_ids[0] if unresolved_ids else point_id
            next_point_query = f"&current_point={quote_plus(next_point)}" if next_point else ""
            return _redirect(
                start_response,
                f"/?job_id={job_id}&review_filter={quote_plus(filter_name)}{next_point_query}&message={quote_plus(f'Updated {point_id}')}",
            )

        source_file_match = re.match(r"^/jobs/([^/]+)/source\.mp4$", path)
        if source_file_match and method == "GET":
            job = load_match_job(job_json_path_from_id(source_file_match.group(1), jobs_root))
            working_path = Path(job.artifacts.working_video_path)
            return _serve_file(start_response, working_path if working_path.exists() else Path(job.raw_video_path))

        preview_file_match = re.match(r"^/jobs/([^/]+)/preview\.mp4$", path)
        if preview_file_match and method == "GET":
            job = load_match_job(job_json_path_from_id(preview_file_match.group(1), jobs_root))
            return _serve_file(start_response, Path(job.artifacts.preview_video_path))

        final_file_match = re.match(r"^/jobs/([^/]+)/final\.mp4$", path)
        if final_file_match and method == "GET":
            job = load_match_job(job_json_path_from_id(final_file_match.group(1), jobs_root))
            return _serve_file(start_response, Path(job.artifacts.final_video_path))

        timeline_file_match = re.match(r"^/jobs/([^/]+)/timeline\.json$", path)
        if timeline_file_match and method == "GET":
            job = load_match_job(job_json_path_from_id(timeline_file_match.group(1), jobs_root))
            return _serve_file(start_response, Path(job.artifacts.timeline_json_path))

        clip_match = re.match(r"^/jobs/([^/]+)/clips/([^/]+)\.mp4$", path)
        if clip_match and method == "GET":
            job = load_match_job(job_json_path_from_id(clip_match.group(1), jobs_root))
            clip_path = Path(job.artifacts.review_clips_dir) / f"{clip_match.group(2)}.mp4"
            return _serve_file(start_response, clip_path)

        log_match = re.match(r"^/jobs/([^/]+)/log$", path)
        if log_match and method == "GET":
            job_id = log_match.group(1)
            from backend.production_pipeline import LOGS_DIR
            log_path = LOGS_DIR / f"{job_id}.log"
            if not log_path.exists():
                return _respond_text(start_response, "(no log yet)")
            lines = log_path.read_text(encoding="utf-8").splitlines()
            tail = "\n".join(lines[-100:])
            return _respond_text(start_response, tail)

        return _respond_text(start_response, f"Unhandled route: {escape(path)}", status="404 Not Found")

    return app
