from __future__ import annotations

from dataclasses import dataclass
from html import escape
import mimetypes
from pathlib import Path
import re
from socketserver import ThreadingMixIn
import threading
from typing import Callable, Iterable
from urllib.parse import parse_qs, quote_plus

from jinja2 import DictLoader, Environment, select_autoescape
from wsgiref.simple_server import WSGIServer

from backend.production_jobs import (
    MatchJob,
    build_review_status,
    create_match_job,
    format_seconds_mmss,
    job_json_path_from_id,
    list_match_jobs,
    load_match_job,
    parse_timecode_to_seconds,
    point_is_review_resolved,
    update_job_runtime_state,
)
from backend.production_pipeline import (
    ProductionPipelineConfig,
    export_job_final_video,
    load_job_timeline,
    render_job_preview,
    review_job_point,
    run_initial_job_pipeline,
)
from backend.rally_timeline_contract import counts_toward_score


TEMPLATES = {
    "base.html": """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{ title }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {% if auto_refresh_sec %}
  <meta http-equiv="refresh" content="{{ auto_refresh_sec }}">
  {% endif %}
  <style>
    :root {
      --bg: #f4f1ea;
      --panel: #fffaf2;
      --ink: #1f1b18;
      --muted: #73685d;
      --line: #d9ccbc;
      --accent: #0b6e4f;
      --accent-2: #d97706;
      --danger: #b42318;
    }
    body {
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #f8f4ed 0%, #efe6d7 100%);
      color: var(--ink);
    }
    .wrap {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      gap: 12px;
    }
    .topbar a {
      color: var(--ink);
      text-decoration: none;
      font-weight: 600;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 12px 28px rgba(80, 60, 30, 0.08);
      margin-bottom: 18px;
    }
    .grid {
      display: grid;
      gap: 16px;
    }
    .grid.two {
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }
    .meta {
      color: var(--muted);
      font-size: 14px;
    }
    .message {
      padding: 12px 14px;
      border-radius: 12px;
      margin-bottom: 16px;
      border: 1px solid var(--line);
      background: #f7f2e9;
    }
    .message.error {
      background: #fff1ef;
      border-color: #f5c2bd;
      color: var(--danger);
    }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      background: #efe5d7;
    }
    .badge.running { background: #e7f0ff; color: #1d4ed8; }
    .badge.needs_review { background: #fff2db; color: #b45309; }
    .badge.ready_for_final { background: #e7f8ef; color: #15803d; }
    .badge.completed { background: #dbf7e7; color: #166534; }
    .badge.failed { background: #ffe3df; color: #b42318; }
    .row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
    }
    .stat {
      background: #f8f2e8;
      border-radius: 12px;
      padding: 12px;
      border: 1px solid var(--line);
    }
    label {
      display: block;
      font-weight: 600;
      margin-bottom: 6px;
    }
    input, select, button, textarea {
      font: inherit;
    }
    input, select, textarea {
      width: 100%;
      box-sizing: border-box;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid #cdbda9;
      background: #fffdf9;
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      background: var(--accent);
      color: white;
      cursor: pointer;
      font-weight: 700;
    }
    button.secondary {
      background: #2f4858;
    }
    button.warn {
      background: var(--accent-2);
    }
    button.danger {
      background: var(--danger);
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }
    video {
      width: 100%;
      max-width: 420px;
      border-radius: 12px;
      background: black;
    }
    .point-card {
      display: grid;
      grid-template-columns: minmax(260px, 380px) 1fr;
      gap: 16px;
      align-items: start;
    }
    .point-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .subtle-link {
      color: #174b6a;
      text-decoration: none;
      font-weight: 600;
    }
    @media (max-width: 900px) {
      .point-card {
        grid-template-columns: 1fr;
      }
      .wrap {
        padding: 14px;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <a href="/">Scoreboard Tool Local UI</a>
      <div class="meta">Single operator · one video at a time</div>
    </div>
    {% if message %}
      <div class="message{% if error %} error{% endif %}">{{ message }}</div>
    {% endif %}
    {% block body %}{% endblock %}
  </div>
</body>
</html>
""",
    "index.html": """
{% extends "base.html" %}
{% block body %}
<div class="grid two">
  <div class="panel">
    <h2>Create Match Job</h2>
    <form method="post" action="/jobs">
      <div style="margin-bottom:12px;">
        <label for="raw_video_path">Raw Video Path</label>
        <input id="raw_video_path" name="raw_video_path" placeholder="C:/videos/match.mp4" required>
      </div>
      <div class="grid two">
        <div>
          <label for="player_a_name">Player A Name (Near)</label>
          <input id="player_a_name" name="player_a_name" value="Player A" required>
        </div>
        <div>
          <label for="player_b_name">Player B Name (Far)</label>
          <input id="player_b_name" name="player_b_name" value="Player B" required>
        </div>
      </div>
      <div class="grid two" style="margin-top:12px;">
        <div>
          <label for="trim_start">Trim Start</label>
          <input id="trim_start" name="trim_start" value="00:00" placeholder="mm:ss or seconds">
        </div>
        <div>
          <label for="best_of">Best Of</label>
          <select id="best_of" name="best_of">
            <option value="3">3</option>
            <option value="5" selected>5</option>
            <option value="7">7</option>
          </select>
        </div>
      </div>
      <div class="point-actions" style="margin-top:16px;">
        <button type="submit">Create Job</button>
      </div>
    </form>
  </div>
  <div class="panel">
    <h2>Current Flow</h2>
    <div class="meta">
      raw video → trim → rally timeline → winner adapter → preview render → review corrections → final export
    </div>
  </div>
</div>

<div class="panel">
  <h2>Jobs</h2>
  {% if jobs %}
    <table>
      <thead>
        <tr>
          <th>Job</th>
          <th>Status</th>
          <th>Players</th>
          <th>Trim</th>
          <th>Updated</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
      {% for job in jobs %}
        <tr>
          <td>
            <div><strong>{{ job.job_id }}</strong></div>
            <div class="meta">{{ job.raw_video_path }}</div>
          </td>
          <td><span class="badge {{ job.status }}">{{ job.status }}</span></td>
          <td>{{ job.player_a_name }} vs {{ job.player_b_name }}</td>
          <td>{{ job.trim_start_label }}</td>
          <td>{{ job.updated_at }}</td>
          <td><a class="subtle-link" href="/jobs/{{ job.job_id }}">Open</a></td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  {% else %}
    <div class="meta">No jobs yet.</div>
  {% endif %}
</div>
{% endblock %}
""",
    "job_detail.html": """
{% extends "base.html" %}
{% block body %}
<div class="panel">
  <div class="row" style="justify-content:space-between;align-items:flex-start;">
    <div>
      <h2>{{ job.job_id }}</h2>
      <div class="meta">{{ job.raw_video_path }}</div>
    </div>
    <span class="badge {{ job.status }}">{{ job.status }}</span>
  </div>
  <div class="grid two" style="margin-top:12px;">
    <div class="stats">
      <div class="stat"><strong>Player A</strong><div>{{ job.player_a_name }} (Near)</div></div>
      <div class="stat"><strong>Player B</strong><div>{{ job.player_b_name }} (Far)</div></div>
      <div class="stat"><strong>Trim Start</strong><div>{{ trim_start_label }}</div></div>
      <div class="stat"><strong>Best Of</strong><div>{{ job.best_of }}</div></div>
    </div>
    <div class="stats">
      <div class="stat"><strong>Current Step</strong><div>{{ job.current_step }}</div></div>
      <div class="stat"><strong>Resolved</strong><div>{{ review_status.resolved_scoring_points }}/{{ review_status.scoring_points }}</div></div>
      <div class="stat"><strong>Needs Review</strong><div>{{ review_status.unresolved_scoring_points }}</div></div>
      <div class="stat"><strong>Blocked</strong><div>{{ review_status.blocked_points }}</div></div>
    </div>
  </div>
  <div class="point-actions">
    <form method="post" action="/jobs/{{ job.job_id }}/run"><button type="submit">Run Pipeline</button></form>
    <form method="post" action="/jobs/{{ job.job_id }}/preview"><button class="secondary" type="submit">Refresh Preview</button></form>
    <form method="post" action="/jobs/{{ job.job_id }}/final-export"><button class="warn" type="submit">Final Export</button></form>
    {% if has_timeline %}
      <a class="subtle-link" href="/jobs/{{ job.job_id }}/review">Open Review</a>
    {% endif %}
    {% if timeline_json_exists %}
      <a class="subtle-link" href="/jobs/{{ job.job_id }}/timeline.json">Timeline JSON</a>
    {% endif %}
  </div>
  {% if job.error_message %}
    <div class="message error" style="margin-top:14px;">{{ job.error_message }}</div>
  {% endif %}
</div>

{% if has_timeline %}
<div class="panel">
  <h2>Timeline Summary</h2>
  <div class="stats">
    <div class="stat"><strong>Total Rallies</strong><div>{{ timeline_summary.total_rallies or 0 }}</div></div>
    <div class="stat"><strong>Known Winners</strong><div>{{ timeline_summary.winner_known_rallies or 0 }}</div></div>
    <div class="stat"><strong>Review</strong><div>{{ timeline_summary.winner_review_rallies or 0 }}</div></div>
    <div class="stat"><strong>Blocked</strong><div>{{ timeline_summary.winner_blocked_rallies or 0 }}</div></div>
  </div>
  <div class="meta" style="margin-top:12px;">
    Preview render is allowed with unresolved reviews. Final export stays blocked until all scoring rallies are confirmed.
  </div>
</div>
{% endif %}

{% if preview_exists %}
<div class="panel">
  <h2>Preview Render</h2>
  <video controls preload="metadata" src="/jobs/{{ job.job_id }}/preview.mp4?ts={{ cache_bust }}"></video>
</div>
{% endif %}

{% if final_exists %}
<div class="panel">
  <h2>Final Export</h2>
  <video controls preload="metadata" src="/jobs/{{ job.job_id }}/final.mp4?ts={{ cache_bust }}"></video>
</div>
{% endif %}
{% endblock %}
""",
    "review.html": """
{% extends "base.html" %}
{% block body %}
<div class="panel">
  <div class="row" style="justify-content:space-between;align-items:flex-start;">
    <div>
      <h2>Review {{ job.job_id }}</h2>
      <div class="meta">Operator only needs to decide who won each rally.</div>
    </div>
    <span class="badge {{ job.status }}">{{ job.status }}</span>
  </div>
  <div class="stats" style="margin-top:12px;">
    <div class="stat"><strong>Resolved</strong><div>{{ review_status.resolved_scoring_points }}</div></div>
    <div class="stat"><strong>Needs Review</strong><div>{{ review_status.unresolved_scoring_points }}</div></div>
    <div class="stat"><strong>Blocked</strong><div>{{ review_status.blocked_points }}</div></div>
    <div class="stat"><strong>Preview Known</strong><div>{{ review_status.preview_known_points }}</div></div>
  </div>
  <div class="point-actions">
    <a class="subtle-link" href="/jobs/{{ job.job_id }}">Back To Job</a>
    <form method="post" action="/jobs/{{ job.job_id }}/preview"><button class="secondary" type="submit">Refresh Preview</button></form>
    <form method="post" action="/jobs/{{ job.job_id }}/final-export"><button class="warn" type="submit">Final Export</button></form>
  </div>
</div>

{% for point in points %}
<div class="panel" id="{{ point.id }}">
  <div class="point-card">
    <div>
      <video controls preload="metadata" src="/jobs/{{ job.job_id }}/clips/{{ point.id }}.mp4?ts={{ cache_bust }}"></video>
    </div>
    <div>
      <div class="row" style="justify-content:space-between;">
        <div>
          <h3 style="margin:0 0 6px 0;">{{ point.id }}</h3>
          <div class="meta">{{ point.time_range }}</div>
        </div>
        <div>
          {% if point.resolved %}
            <span class="badge completed">resolved</span>
          {% elif point.decision == "blocked" %}
            <span class="badge failed">blocked</span>
          {% else %}
            <span class="badge needs_review">review</span>
          {% endif %}
        </div>
      </div>
      <div class="stats" style="margin-top:10px;">
        <div class="stat"><strong>Current Winner</strong><div>{{ point.current_winner_label }}</div></div>
        <div class="stat"><strong>AI Winner</strong><div>{{ point.ai_winner_label }}</div></div>
        <div class="stat"><strong>Category</strong><div>{{ point.category }}</div></div>
        <div class="stat"><strong>Source</strong><div>{{ point.source }}</div></div>
      </div>
      <div class="point-actions">
        {% if point.can_keep %}
        <form method="post" action="/jobs/{{ job.job_id }}/review/{{ point.id }}">
          <input type="hidden" name="action" value="keep">
          <button type="submit">Keep AI Winner</button>
        </form>
        {% endif %}
        <form method="post" action="/jobs/{{ job.job_id }}/review/{{ point.id }}">
          <input type="hidden" name="action" value="set_winner">
          <input type="hidden" name="winner" value="player_a">
          <button class="secondary" type="submit">{{ job.player_a_name }} Wins</button>
        </form>
        <form method="post" action="/jobs/{{ job.job_id }}/review/{{ point.id }}">
          <input type="hidden" name="action" value="set_winner">
          <input type="hidden" name="winner" value="player_b">
          <button class="secondary" type="submit">{{ job.player_b_name }} Wins</button>
        </form>
      </div>
      {% if point.last_note %}
        <div class="meta" style="margin-top:12px;">Last review note: {{ point.last_note }}</div>
      {% endif %}
    </div>
  </div>
</div>
{% endfor %}
{% endblock %}
""",
}


TEMPLATE_ENV = Environment(
    loader=DictLoader(TEMPLATES),
    autoescape=select_autoescape(default=True),
)


def _render_template(template_name: str, **context: object) -> bytes:
    return TEMPLATE_ENV.get_template(template_name).render(**context).encode("utf-8")


def _redirect(start_response: Callable, location: str):
    start_response("302 Found", [("Location", location)])
    return [b""]


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


def _review_point_rows(job: MatchJob):
    timeline = load_job_timeline(job)
    rows = []
    for point in timeline.points:
        if not counts_toward_score(point):
            continue
        rows.append(
            {
                "id": point.id,
                "time_range": f"{point.t_start:.2f}s -> {point.t_end:.2f}s",
                "current_winner_label": _winner_label(point.winner, job),
                "ai_winner_label": _winner_label(point.winner_candidate, job),
                "category": point.winner_end_category or "-",
                "source": point.source,
                "decision": point.winner_decision or "",
                "resolved": point_is_review_resolved(point),
                "can_keep": point.winner in {"player_a", "player_b"} or point.winner_candidate in {"player_a", "player_b"},
                "last_note": point.corrections[-1].note if point.corrections else "",
            }
        )
    rows.sort(key=lambda item: (item["resolved"], item["id"]))
    return timeline, rows


class JobTaskRunner:
    def __init__(self, *, config: ProductionPipelineConfig, jobs_root: Path | None = None):
        self.config = config
        self.jobs_root = jobs_root
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

    def _active_job_id(self) -> str | None:
        for job_id, thread in self._threads.items():
            if thread.is_alive():
                return job_id
        return None

    def start(self, job_id: str, target: Callable[[str], None]) -> tuple[bool, str]:
        with self._lock:
            active_job_id = self._active_job_id()
            if active_job_id and active_job_id != job_id:
                return False, f"Another job is already running: {active_job_id}"

            current = self._threads.get(job_id)
            if current is not None and current.is_alive():
                return False, f"Job {job_id} is already running"

            thread = threading.Thread(target=self._run_safe, args=(job_id, target), daemon=True)
            self._threads[job_id] = thread
            thread.start()
            return True, f"Started background task for {job_id}"

    def _run_safe(self, job_id: str, target: Callable[[str], None]) -> None:
        try:
            target(job_id)
        except Exception as exc:
            job = load_match_job(job_json_path_from_id(job_id, self.jobs_root))
            update_job_runtime_state(job, status="failed", current_step="failed", error_message=str(exc))


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


def create_local_web_app(
    *,
    config: ProductionPipelineConfig | None = None,
    jobs_root: Path | None = None,
):
    config = config or ProductionPipelineConfig()
    runner = JobTaskRunner(config=config, jobs_root=jobs_root)

    def app(environ, start_response):
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = str(environ.get("PATH_INFO", "/"))
        message_ctx = _message_context(environ)

        if path == "/" and method == "GET":
            jobs = [_job_display_row(job) for job in list_match_jobs(jobs_root)]
            body = _render_template(
                "index.html",
                title="Scoreboard Tool Local UI",
                jobs=jobs,
                auto_refresh_sec=None,
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
            return _redirect(start_response, f"/jobs/{job.job_id}?message={quote_plus('Job created')}")

        job_match = re.match(r"^/jobs/([^/]+)$", path)
        if job_match and method == "GET":
            job_id = job_match.group(1)
            try:
                job = load_match_job(job_json_path_from_id(job_id, jobs_root))
            except Exception:
                return _respond_text(start_response, "Job not found", status="404 Not Found")

            has_timeline = Path(job.artifacts.timeline_json_path).exists()
            timeline_summary = job.timeline_summary or {}
            review_status = job.review_status or {}
            if has_timeline and not review_status:
                review_status = build_review_status(load_job_timeline(job))
            body = _render_template(
                "job_detail.html",
                title=f"Job {job.job_id}",
                job=job,
                trim_start_label=format_seconds_mmss(job.trim_start_sec),
                timeline_summary=timeline_summary,
                review_status=review_status,
                has_timeline=has_timeline,
                preview_exists=Path(job.artifacts.preview_video_path).exists(),
                final_exists=Path(job.artifacts.final_video_path).exists(),
                timeline_json_exists=Path(job.artifacts.timeline_json_path).exists(),
                cache_bust=job.updated_at,
                auto_refresh_sec=5 if job.status == "running" else None,
                **message_ctx,
            )
            return _respond_html(start_response, body=body)

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
            return _redirect(start_response, f"/jobs/{job_id}?kind={kind}&message={quote_plus(msg)}")

        preview_match = re.match(r"^/jobs/([^/]+)/preview$", path)
        if preview_match and method == "POST":
            job_id = preview_match.group(1)
            ok, msg = runner.start(
                job_id,
                lambda current_job_id: render_job_preview(job_json_path_from_id(current_job_id, jobs_root)),
            )
            kind = "info" if ok else "error"
            return _redirect(start_response, f"/jobs/{job_id}?kind={kind}&message={quote_plus(msg)}")

        final_match = re.match(r"^/jobs/([^/]+)/final-export$", path)
        if final_match and method == "POST":
            job_id = final_match.group(1)
            ok, msg = runner.start(
                job_id,
                lambda current_job_id: export_job_final_video(job_json_path_from_id(current_job_id, jobs_root)),
            )
            kind = "info" if ok else "error"
            location = f"/jobs/{job_id}?kind={kind}&message={quote_plus(msg)}"
            if path.endswith("/final-export") and environ.get("HTTP_REFERER", "").endswith("/review"):
                location = f"/jobs/{job_id}/review?kind={kind}&message={quote_plus(msg)}"
            return _redirect(start_response, location)

        review_match = re.match(r"^/jobs/([^/]+)/review$", path)
        if review_match and method == "GET":
            job_id = review_match.group(1)
            try:
                job = load_match_job(job_json_path_from_id(job_id, jobs_root))
                timeline, points = _review_point_rows(job)
            except Exception:
                return _respond_text(start_response, "Job not found", status="404 Not Found")

            body = _render_template(
                "review.html",
                title=f"Review {job.job_id}",
                job=job,
                points=points,
                review_status=build_review_status(timeline),
                cache_bust=job.updated_at,
                auto_refresh_sec=5 if job.status == "running" else None,
                **message_ctx,
            )
            return _respond_html(start_response, body=body)

        review_point_match = re.match(r"^/jobs/([^/]+)/review/([^/]+)$", path)
        if review_point_match and method == "POST":
            job_id = review_point_match.group(1)
            point_id = review_point_match.group(2)
            form = _read_form(environ)
            try:
                review_job_point(
                    job_json_path_from_id(job_id, jobs_root),
                    point_id=point_id,
                    action=str(form.get("action", "")),
                    winner=form.get("winner", None),
                    reviewer="local_operator",
                )
            except Exception as exc:
                return _redirect(start_response, f"/jobs/{job_id}/review?kind=error&message={quote_plus(str(exc))}")
            return _redirect(start_response, f"/jobs/{job_id}/review?message={quote_plus(f'Updated {point_id}')}")

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

        return _respond_text(start_response, f"Unhandled route: {escape(path)}", status="404 Not Found")

    return app
