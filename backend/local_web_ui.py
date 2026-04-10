from __future__ import annotations

from datetime import datetime, timezone
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

from backend.engine import ScoreEngine
from backend.models import MatchSnapshot, MatchState, RallyEvent
from backend.production_jobs import (
    MatchJob,
    build_review_status,
    create_match_job,
    format_seconds_mmss,
    job_json_path_from_id,
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
  <script>
    (function () {
      var elapsedSec = 0;
      function _setText(id, val) { var el = document.getElementById(id); if (el && val !== undefined) el.textContent = val; }
      function _syncElapsed(label) {
        var em = (label || "").match(/(\d+)\s*phút\s*(\d+)\s*giây/);
        if (em) elapsedSec = parseInt(em[1]) * 60 + parseInt(em[2]);
      }
      function pollStatus() {
        var card = document.querySelector(".progress-card[data-job-id]");
        if (!card) return;
        var jobId = card.getAttribute("data-job-id");
        var knownStatus = card.getAttribute("data-job-status");
        if (!jobId) return;
        fetch("/api/job-status?job_id=" + encodeURIComponent(jobId))
          .then(function (r) { return r.ok ? r.json() : null; })
          .then(function (d) {
            if (!d) return;
            if (d.status !== knownStatus) { location.reload(); return; }
            var p = d.progress;
            _setText("prog_label", p.label);
            _setText("prog_step", p.step_label);
            _setText("prog_rallies", p.rallies_label);
            _setText("prog_resolved", p.resolved_label);
            _setText("prog_pending", p.pending_label);
            _syncElapsed(p.elapsed_label);
            var fill = document.querySelector(".progress-fill");
            var pctLabel = document.getElementById("progress_pct_label");
            if (fill) {
              fill.style.width = p.percent + "%";
              if (pctLabel) pctLabel.textContent = p.percent + "%";
            }
          })
          .catch(function () {});
      }
      document.addEventListener("DOMContentLoaded", function () {
        var elapsedEl = document.getElementById("prog_elapsed");
        if (elapsedEl) _syncElapsed(elapsedEl.textContent);
        setInterval(function () {
          elapsedSec++;
          var mm = Math.floor(elapsedSec / 60), ss = elapsedSec % 60;
          _setText("prog_elapsed", mm + " phút " + ss + " giây");
        }, 1000);
        setInterval(pollStatus, {{ auto_refresh_sec * 1000 }});
      });
    })();
  </script>
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
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
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
    video.main-player {
      max-width: 100%;
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
    .input-with-action {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
    }
    .browse-table td:last-child,
    .browse-table th:last-child {
      text-align: right;
      white-space: nowrap;
    }
    .hint-box {
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #f8f2e8;
    }
    .tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .tabs a {
      text-decoration: none;
      color: var(--ink);
      border: 1px solid var(--line);
      background: #f7f2e9;
      border-radius: 999px;
      padding: 8px 12px;
      font-weight: 600;
    }
    .tabs a.active {
      background: #e7f8ef;
      border-color: #9fd1b9;
      color: #166534;
    }
    .section-title {
      margin: 18px 0 10px 0;
    }
    .setup-shell {
      display: grid;
      grid-template-columns: minmax(320px, 460px) 1fr;
      gap: 18px;
      align-items: start;
    }
    .reviewer-shell {
      display: grid;
      grid-template-columns: minmax(420px, 1.15fr) minmax(340px, 0.85fr);
      gap: 18px;
      align-items: start;
    }
    .trim-player,
    .review-player {
      width: 100%;
      border-radius: 14px;
      background: black;
    }
    .scoreboard-live {
      margin-top: 14px;
      padding: 16px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #fffdf8 0%, #f6ecdd 100%);
    }
    .scoreboard-header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }
    .score-big {
      font-size: 42px;
      line-height: 1;
      font-weight: 800;
      letter-spacing: 0.02em;
    }
    .score-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
      padding: 10px 0;
      border-top: 1px solid #e6d8c6;
    }
    .score-row:first-of-type {
      border-top: 0;
      padding-top: 0;
    }
    .score-name {
      font-size: 18px;
      font-weight: 700;
    }
    .action-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    .action-grid form {
      margin: 0;
    }
    .action-btn {
      width: 100%;
      min-height: 72px;
      font-size: 22px;
      font-weight: 800;
      border-radius: 16px;
    }
    .action-btn.near {
      background: #b42318;
    }
    .action-btn.far {
      background: #174b6a;
    }
    .action-btn.let {
      background: #6b7280;
    }
    .timeline-list {
      margin-top: 16px;
      display: grid;
      gap: 8px;
      max-height: 520px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .timeline-item {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #fbf7f0;
      color: var(--ink);
      text-decoration: none;
    }
    .timeline-item.current {
      border-color: #174b6a;
      background: #eef5f8;
      box-shadow: inset 0 0 0 1px #174b6a;
    }
    .timeline-item.pending {
      background: #fff6df;
    }
    .timeline-item.resolved-a {
      background: #fdf0ee;
    }
    .timeline-item.resolved-b {
      background: #edf5fb;
    }
    .timeline-item.let {
      background: #f2f4f7;
    }
    .timeline-status {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .timeline-status.pending {
      background: #facc15;
      color: #713f12;
    }
    .timeline-status.resolved-a {
      background: #fecaca;
      color: #7f1d1d;
    }
    .timeline-status.resolved-b {
      background: #bfdbfe;
      color: #1e3a8a;
    }
    .timeline-status.let {
      background: #d1d5db;
      color: #111827;
    }
    .progress-card {
      margin-top: 14px;
      padding: 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #f9f4ec;
    }
    .progress-bar {
      height: 12px;
      width: 100%;
      border-radius: 999px;
      background: #eadfce;
      overflow: hidden;
      margin: 10px 0 12px 0;
    }
    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, #0b6e4f 0%, #d97706 100%);
      border-radius: 999px;
      transition: width 0.3s ease;
    }
    .progress-meta {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
    }
    .progress-stat {
      background: #fffaf2;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
    }
    @media (max-width: 900px) {
      .point-card {
        grid-template-columns: 1fr;
      }
      .wrap {
        padding: 14px;
      }
      .input-with-action {
        grid-template-columns: 1fr;
      }
      .setup-shell,
      .reviewer-shell {
        grid-template-columns: 1fr;
      }
    }
  </style>
  <script>
    function applyRawVideoPath(path) {
      var field = document.getElementById("raw_video_path");
      if (field) {
        field.value = path;
        field.focus();
      }
    }
    function playMainVideo(src, label) {
      var player = document.getElementById("main_video_player");
      if (!player || !src) {
        return false;
      }
      player.src = src;
      player.load();
      var title = document.getElementById("main_now_playing");
      if (title && label) {
        title.textContent = label;
      }
      return true;
    }
    function showSelectedRawVideo(path) {
      applyRawVideoPath(path);
      var nextUrl = "/?raw_video_path=" + encodeURIComponent(path);
      window.history.replaceState({}, "", nextUrl);
      if (!playMainVideo("/local-video?path=" + encodeURIComponent(path), "Playing selected raw video")) {
        window.location.href = nextUrl;
      }
      return false;
    }
    function openRawVideoBrowser() {
      var popup = window.open("/browse/raw-video", "raw-video-browser", "width=980,height=760");
      if (!popup) {
        window.location.href = "/browse/raw-video";
      }
    }
    function chooseRawVideo(path) {
      if (window.opener && !window.opener.closed && typeof window.opener.showSelectedRawVideo === "function") {
        window.opener.showSelectedRawVideo(path);
        window.close();
        return false;
      }
      window.location.href = "/?raw_video_path=" + encodeURIComponent(path);
      return false;
    }
    document.addEventListener("keydown", function (event) {
      var active = document.activeElement;
      var tag = active && active.tagName ? active.tagName.toLowerCase() : "";
      if (tag === "input" || tag === "textarea" || tag === "select") {
        return;
      }
      if (event.key === "ArrowLeft") {
        var nearBtn = document.getElementById("review_action_near");
        if (nearBtn) {
          event.preventDefault();
          nearBtn.click();
        }
      } else if (event.key === "ArrowRight") {
        var farBtn = document.getElementById("review_action_far");
        if (farBtn) {
          event.preventDefault();
          farBtn.click();
        }
      }
    });
    document.addEventListener("DOMContentLoaded", function () {
      var elapsedEl = document.getElementById("prog_elapsed");
      if (elapsedEl) {
        var m = elapsedEl.textContent.match(/(\d+)\s*phút\s*(\d+)\s*giây/);
        if (m) {
          var elapsedSec = parseInt(m[1]) * 60 + parseInt(m[2]);
          setInterval(function () {
            elapsedSec++;
            var mm = Math.floor(elapsedSec / 60);
            var ss = elapsedSec % 60;
            elapsedEl.textContent = mm + " phút " + ss + " giây";
          }, 1000);
        }
      }
    });
    setInterval(function () {
      fetch("/heartbeat", { method: "POST" }).catch(function () {});
    }, 5000);
  </script>
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
{% if screen_mode == "review" and current_job and current_point %}
<div class="reviewer-shell">
  <div class="panel">
    <h2>The Reviewer</h2>
    <div class="meta">{{ current_job.player_a_name }} (Near) vs {{ current_job.player_b_name }} (Far)</div>
    <div class="meta" style="margin-top:4px;">Current rally: {{ current_point.id }} | {{ current_point.time_range }}</div>
    <div class="progress-card" data-job-id="{{ current_job.job_id }}" data-job-status="{{ current_job.status }}">
      <div class="row" style="justify-content:space-between;">
        <strong id="prog_label">{{ progress.label }}</strong>
        <span id="progress_pct_label">{{ progress.percent }}%</span>
      </div>
      <div class="progress-bar"><div class="progress-fill" style="width: {{ progress.percent }}%;" data-server-pct="{{ progress.percent }}" data-job-id="{{ current_job.job_id if current_job else '' }}"></div></div>
      <div class="meta" id="prog_step">{{ progress.step_label }}</div>
      <div class="progress-meta" style="margin-top:12px;">
        <div class="progress-stat"><strong>Elapsed</strong><div id="prog_elapsed">{{ progress.elapsed_label }}</div></div>
        <div class="progress-stat"><strong>Rallies</strong><div id="prog_rallies">{{ progress.rallies_label }}</div></div>
        <div class="progress-stat"><strong>Resolved</strong><div id="prog_resolved">{{ progress.resolved_label }}</div></div>
        <div class="progress-stat"><strong>Pending</strong><div id="prog_pending">{{ progress.pending_label }}</div></div>
      </div>
    </div>
    {% if stage_message %}
      <div class="hint-box" style="margin-top:14px;">{{ stage_message }}</div>
    {% endif %}
    {% if current_job.error_message %}
      <div class="message error" style="margin-top:14px;">{{ current_job.error_message }}</div>
    {% endif %}
    <video id="main_video_player" class="review-player" controls preload="metadata" autoplay loop playsinline src="{{ main_video_src }}"></video>
    <div class="point-actions">
      {% if full_match_src %}
        <button class="secondary" type="button" onclick='return playMainVideo({{ full_match_src|tojson }}, {{ full_match_label|tojson }});'>Play Full Match</button>
      {% endif %}
      {% if final_video_src %}
        <button class="secondary" type="button" onclick='return playMainVideo({{ final_video_src|tojson }}, {{ final_video_label|tojson }});'>Play Exported Video</button>
      {% endif %}
    </div>
    <div class="scoreboard-live">
      <div class="scoreboard-header">
        <div>
          <div class="meta">Live Scoreboard</div>
          <div class="meta">State before {{ current_point.id }}</div>
        </div>
        <div class="meta">Set {{ scoreboard.set_number }} | Sets {{ scoreboard.sets_a }} - {{ scoreboard.sets_b }}</div>
      </div>
      <div class="score-row">
        <div class="score-name">[NEAR] {{ current_job.player_a_name }}</div>
        <div class="score-big">{{ scoreboard.score_a }}</div>
      </div>
      <div class="score-row">
        <div class="score-name">[FAR] {{ current_job.player_b_name }}</div>
        <div class="score-big">{{ scoreboard.score_b }}</div>
      </div>
    </div>
  </div>
  <div class="panel" id="main-panel">
    <h2>Action Panel</h2>
    <div class="hint-box">Left Arrow = Near win | Right Arrow = Far win</div>
    <div class="meta" style="margin-top:12px;">AI suggestion: {{ current_point.ai_winner_label }} | {{ current_point.review_status_label }}</div>
    <div class="action-grid" style="margin-top:14px;">
      <form method="post" action="/jobs/{{ current_job.job_id }}/review/{{ current_point.id }}">
        <input type="hidden" name="filter" value="{{ active_filter }}">
        <input type="hidden" name="current_point" value="{{ current_point.id }}">
        <input type="hidden" name="action" value="set_winner">
        <input type="hidden" name="winner" value="player_a">
        <button id="review_action_near" class="action-btn near" type="submit">NEAR WIN</button>
      </form>
      <form method="post" action="/jobs/{{ current_job.job_id }}/review/{{ current_point.id }}">
        <input type="hidden" name="filter" value="{{ active_filter }}">
        <input type="hidden" name="current_point" value="{{ current_point.id }}">
        <input type="hidden" name="action" value="set_winner">
        <input type="hidden" name="winner" value="player_b">
        <button id="review_action_far" class="action-btn far" type="submit">FAR WIN</button>
      </form>
    </div>
    <div class="point-actions" style="margin-top:16px;">
      <a class="subtle-link" href="/?job_id={{ current_job.job_id }}&review_filter={% if active_filter == 'pending' %}all{% else %}pending{% endif %}&current_point={{ current_point.id }}">
        {% if active_filter == 'pending' %}Show All Points{% else %}Review Difficult Only{% endif %}
      </a>
      <form method="post" action="/jobs/{{ current_job.job_id }}/final-export">
        <button class="warn" type="submit" {% if not review_status.final_export_ready %}disabled{% endif %}>Export</button>
      </form>
    </div>
    {% if not review_status.final_export_ready %}
      <div class="meta" style="margin-top:8px;">Export unlocks after every scoring rally has a confirmed winner.</div>
    {% endif %}
    <div class="timeline-list">
      {% for point in points %}
        <a class="timeline-item {{ point.status_class }} {% if point.id == current_point_id %}current{% endif %}" href="/?job_id={{ current_job.job_id }}&review_filter={{ active_filter }}&current_point={{ point.id }}">
          <div>
            <strong>{{ point.id }}</strong>
            <div class="meta">{{ point.time_range }}</div>
          </div>
          <span class="timeline-status {{ point.status_class }}">{{ point.timeline_status_label }}</span>
        </a>
      {% endfor %}
    </div>
  </div>
</div>
{% else %}
<div class="setup-shell">
  <div class="panel">
    <h2>Setup</h2>
    {% if current_job %}
      <div class="progress-card" data-job-id="{{ current_job.job_id }}" data-job-status="{{ current_job.status }}">
        <div class="row" style="justify-content:space-between;">
          <strong id="prog_label">{{ progress.label }}</strong>
          <span id="progress_pct_label">{{ progress.percent }}%</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" style="width: {{ progress.percent }}%;" data-server-pct="{{ progress.percent }}" data-job-id="{{ current_job.job_id if current_job else '' }}"></div></div>
        <div class="meta" id="prog_step">{{ progress.step_label }}</div>
        <div class="progress-meta" style="margin-top:12px;">
          <div class="progress-stat"><strong>Elapsed</strong><div id="prog_elapsed">{{ progress.elapsed_label }}</div></div>
          <div class="progress-stat"><strong>Rallies</strong><div id="prog_rallies">{{ progress.rallies_label }}</div></div>
          <div class="progress-stat"><strong>Resolved</strong><div id="prog_resolved">{{ progress.resolved_label }}</div></div>
          <div class="progress-stat"><strong>Pending</strong><div id="prog_pending">{{ progress.pending_label }}</div></div>
        </div>
      </div>
    {% endif %}
    {% if stage_message %}
      <div class="hint-box" style="margin-bottom:14px;">{{ stage_message }}</div>
    {% endif %}
    {% if current_job and current_job.error_message %}
      <div class="message error" style="margin-bottom:14px;">{{ current_job.error_message }}</div>
    {% endif %}
    <form method="post" action="/jobs">
      <div style="margin-bottom:12px;">
        <label for="raw_video_path">Raw Video Path</label>
        <div class="input-with-action">
          <input id="raw_video_path" name="raw_video_path" placeholder="C:/videos/match.mp4" value="{{ raw_video_path_value }}" required>
          <button class="secondary" type="button" onclick="openRawVideoBrowser()">Browse</button>
        </div>
        <div class="meta" style="margin-top:8px;">Browse root: {{ raw_matches_root }}</div>
      </div>
      <div class="grid two">
        <div>
          <label for="player_a_name">Player NEAR</label>
          <input id="player_a_name" name="player_a_name" value="{{ player_a_value }}" required>
        </div>
        <div>
          <label for="player_b_name">Player FAR</label>
          <input id="player_b_name" name="player_b_name" value="{{ player_b_value }}" required>
        </div>
      </div>
      <div class="grid two" style="margin-top:12px;">
        <div>
          <label for="best_of">Format</label>
          <select id="best_of" name="best_of">
            <option value="3" {% if best_of_value == 3 %}selected{% endif %}>Best of 3</option>
            <option value="5" {% if best_of_value == 5 %}selected{% endif %}>Best of 5</option>
            <option value="7" {% if best_of_value == 7 %}selected{% endif %}>Best of 7</option>
          </select>
        </div>
        <div>
          <label for="trim_start">Trim Start</label>
          <input id="trim_start" name="trim_start" value="{{ trim_start_value }}" placeholder="mm:ss or seconds">
        </div>
      </div>
      <div class="point-actions" style="margin-top:16px;">
        <button type="submit">Run AI Pipeline</button>
      </div>
    </form>
  </div>
  <div class="panel">
    <h2>Trim Start</h2>
    {% if main_video_src %}
      <div class="meta">Preview raw video here, then type Trim Start manually in the Setup form.</div>
      <video class="trim-player" controls preload="metadata" src="{{ main_video_src }}"></video>
    {% else %}
      <div class="hint-box">Choose the raw match video first. A small preview player will appear here for trim setup.</div>
    {% endif %}
  </div>
</div>
{% endif %}
{% if false %}
<div class="grid two">
  <div class="panel">
    <h2>Create Match Job</h2>
    <form method="post" action="/jobs">
      <div style="margin-bottom:12px;">
        <label for="raw_video_path">Raw Video Path</label>
        <div class="input-with-action">
          <input id="raw_video_path" name="raw_video_path" placeholder="C:/videos/match.mp4" value="{{ raw_video_path_value }}" required>
          <button class="secondary" type="button" onclick="openRawVideoBrowser()">Browse</button>
        </div>
        <div class="meta" style="margin-top:8px;">Browse root: {{ raw_matches_root }}</div>
      </div>
      <div class="grid two">
        <div>
          <label for="player_a_name">Player A Name (Near)</label>
          <input id="player_a_name" name="player_a_name" value="{{ player_a_value }}" required>
        </div>
        <div>
          <label for="player_b_name">Player B Name (Far)</label>
          <input id="player_b_name" name="player_b_name" value="{{ player_b_value }}" required>
        </div>
      </div>
      <div class="grid two" style="margin-top:12px;">
        <div>
          <label for="trim_start">Trim Start</label>
          <input id="trim_start" name="trim_start" value="{{ trim_start_value }}" placeholder="mm:ss or seconds">
        </div>
        <div>
          <label for="best_of">Best Of</label>
          <select id="best_of" name="best_of">
            <option value="3" {% if best_of_value == 3 %}selected{% endif %}>3</option>
            <option value="5" {% if best_of_value == 5 %}selected{% endif %}>5</option>
            <option value="7" {% if best_of_value == 7 %}selected{% endif %}>7</option>
          </select>
        </div>
      </div>
      <div class="point-actions" style="margin-top:16px;">
        <button type="submit">Create Match Job</button>
      </div>
    </form>
  </div>
  <div class="panel" id="main-panel">
    <h2>Main</h2>
    {% if current_job %}
      <div class="meta">{{ current_job.player_a_name }} (Near) vs {{ current_job.player_b_name }} (Far)</div>
      <div class="meta" style="margin-top:4px;">{{ current_job.raw_video_path }}</div>
    {% elif raw_video_path_value %}
      <div class="meta">{{ raw_video_path_value }}</div>
    {% else %}
      <div class="meta">Select a raw video, then this panel becomes the review workspace.</div>
    {% endif %}

    {% if stage_message %}
      <div class="hint-box" style="margin-top:14px;">{{ stage_message }}</div>
    {% endif %}

    {% if current_job and current_job.error_message %}
      <div class="message error" style="margin-top:14px;">{{ current_job.error_message }}</div>
    {% endif %}

    {% if main_video_src %}
      <h3 class="section-title">Video Player</h3>
      <div class="meta" id="main_now_playing">{{ main_now_playing }}</div>
      <video id="main_video_player" class="main-player" controls preload="metadata" src="{{ main_video_src }}"></video>
      <div class="point-actions">
        {% if full_match_src %}
          <button class="secondary" type="button" onclick='return playMainVideo({{ full_match_src|tojson }}, {{ full_match_label|tojson }});'>Play Full Match</button>
        {% endif %}
        {% if final_video_src %}
          <button class="secondary" type="button" onclick='return playMainVideo({{ final_video_src|tojson }}, {{ final_video_label|tojson }});'>Play Exported Video</button>
        {% endif %}
      </div>
    {% else %}
      <div class="hint-box" style="margin-top:14px;">
        Browse a raw video first. As soon as a video is selected, it will appear here for quick preview.
      </div>
    {% endif %}

    {% if current_job %}
      <h3 class="section-title">Review Queue</h3>
      {% if has_timeline %}
        <div class="tabs">
          <a class="{% if active_filter == 'pending' %}active{% endif %}" href="/?job_id={{ current_job.job_id }}&review_filter=pending">Need Review {{ review_status.unresolved_scoring_points }}</a>
          <a class="{% if active_filter == 'all' %}active{% endif %}" href="/?job_id={{ current_job.job_id }}&review_filter=all">All {{ review_status.scoring_points }}</a>
        </div>
        {% if points %}
          <table style="margin-top:12px;">
            <thead>
              <tr>
                <th>Rally</th>
                <th>AI Winner</th>
                <th>Review</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
            {% for point in points %}
              <tr id="{{ point.id }}">
                <td>
                  <div><strong>{{ point.id }}</strong></div>
                  <div class="meta">{{ point.time_range }}</div>
                </td>
                <td>{{ point.ai_winner_label }}</td>
                <td>{{ point.review_status_label }}</td>
                <td>
                  <div class="point-actions">
                    <button class="secondary" type="button" onclick='return playMainVideo({{ point.clip_src|tojson }}, {{ point.play_label|tojson }});'>Play Rally</button>
                    {% if point.can_keep %}
                    <form method="post" action="/jobs/{{ current_job.job_id }}/review/{{ point.id }}">
                      <input type="hidden" name="filter" value="{{ active_filter }}">
                      <button type="submit" name="action" value="keep">AI Correct</button>
                    </form>
                    {% endif %}
                    <form method="post" action="/jobs/{{ current_job.job_id }}/review/{{ point.id }}">
                      <input type="hidden" name="filter" value="{{ active_filter }}">
                      <input type="hidden" name="action" value="set_winner">
                      <input type="hidden" name="winner" value="player_a">
                      <button class="secondary" type="submit">{{ current_job.player_a_name }} Wins</button>
                    </form>
                    <form method="post" action="/jobs/{{ current_job.job_id }}/review/{{ point.id }}">
                      <input type="hidden" name="filter" value="{{ active_filter }}">
                      <input type="hidden" name="action" value="set_winner">
                      <input type="hidden" name="winner" value="player_b">
                      <button class="secondary" type="submit">{{ current_job.player_b_name }} Wins</button>
                    </form>
                  </div>
                </td>
              </tr>
            {% endfor %}
            </tbody>
          </table>
        {% else %}
          <div class="hint-box" style="margin-top:12px;">No rallies match the current filter.</div>
        {% endif %}
      {% else %}
        <div class="hint-box">
          The system is still preparing rally clips for review. This page refreshes automatically while processing continues.
        </div>
      {% endif %}

      <div class="point-actions" style="margin-top:18px;">
        {% if not has_timeline %}
        <form method="post" action="/jobs/{{ current_job.job_id }}/run">
          <button type="submit">Prepare Review</button>
        </form>
        {% endif %}
        <form method="post" action="/jobs/{{ current_job.job_id }}/final-export">
          <button class="warn" type="submit" {% if not review_status.final_export_ready %}disabled{% endif %}>Export</button>
        </form>
      </div>
      {% if not review_status.final_export_ready %}
        <div class="meta" style="margin-top:8px;">Export unlocks after every scoring rally has a confirmed winner.</div>
      {% endif %}
    {% endif %}
  </div>
</div>
{% endif %}
{% if false %}
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
{% endif %}
{% endblock %}
""",
    "browse_raw_video.html": """
{% extends "base.html" %}
{% block body %}
<div class="panel">
  <div class="row" style="justify-content:space-between;align-items:flex-start;">
    <div>
      <h2>Select Raw Video</h2>
      <div class="meta">Browse root: {{ raw_matches_root }}</div>
      <div class="meta" style="margin-top:4px;">Current folder: {{ current_dir_label }}</div>
    </div>
    <a class="subtle-link" href="/">Back To Create Job</a>
  </div>
  <div class="point-actions" style="margin-top:14px;">
    {% if parent_href %}
      <a class="subtle-link" href="{{ parent_href }}">Open Parent Folder</a>
    {% endif %}
    <button class="secondary" type="button" onclick="window.location.reload()">Refresh</button>
  </div>
</div>

<div class="panel">
  {% if entries %}
    <table class="browse-table">
      <thead>
        <tr>
          <th>Name</th>
          <th>Type</th>
          <th>Path</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody>
      {% for entry in entries %}
        <tr>
          <td><strong>{{ entry.name }}</strong></td>
          <td>{{ entry.kind_label }}</td>
          <td class="meta">{{ entry.display_path }}</td>
          <td>
            {% if entry.is_dir %}
              <a class="subtle-link" href="{{ entry.open_href }}">Open</a>
            {% else %}
              <button type="button" onclick='return chooseRawVideo({{ entry.absolute_path|tojson }});'>Select</button>
            {% endif %}
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  {% else %}
    <div class="meta">No folders or supported raw videos found here yet.</div>
    <div class="meta" style="margin-top:8px;">Supported video extensions: {{ supported_extensions }}</div>
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

<div class="panel">
  <h2>Rally Review Queue</h2>
  <div class="hint-box">
    Check each rally clip and tell the system whether the AI winner is correct. If AI is wrong or missing, choose the real winner for that point.
  </div>
  <div class="point-actions" style="margin-top:14px;">
    <a class="subtle-link" href="/jobs/{{ job.job_id }}/review?filter=pending">Review Pending ({{ review_status.unresolved_scoring_points }})</a>
    <a class="subtle-link" href="/jobs/{{ job.job_id }}/review?filter=all">Review All Rallies</a>
    {% if next_unresolved_href %}
      <a class="subtle-link" href="{{ next_unresolved_href }}">Jump To Next Pending</a>
    {% endif %}
  </div>
</div>
{% else %}
<div class="panel">
  <h2>Rally Review Queue</h2>
  <div class="hint-box">
    Run Pipeline first to generate the rally timeline and review clips. After that, this page will show the per-rally review queue with feedback buttons for AI Correct or AI Wrong.
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
  <div class="hint-box" style="margin-top:14px;">
    Review rule: if the AI winner is correct, press <strong>AI Correct</strong>. If it is wrong or unknown, press the real winner for that rally.
  </div>
  <div class="point-actions">
    <a class="subtle-link" href="/jobs/{{ job.job_id }}">Back To Job</a>
    <form method="post" action="/jobs/{{ job.job_id }}/preview"><button class="secondary" type="submit">Refresh Preview</button></form>
    <form method="post" action="/jobs/{{ job.job_id }}/final-export"><button class="warn" type="submit">Final Export</button></form>
  </div>
  <div class="tabs" style="margin-top:14px;">
    <a class="{% if active_filter == 'pending' %}active{% endif %}" href="/jobs/{{ job.job_id }}/review?filter=pending">Pending {{ review_status.unresolved_scoring_points }}</a>
    <a class="{% if active_filter == 'blocked' %}active{% endif %}" href="/jobs/{{ job.job_id }}/review?filter=blocked">Blocked {{ review_status.blocked_points }}</a>
    <a class="{% if active_filter == 'resolved' %}active{% endif %}" href="/jobs/{{ job.job_id }}/review?filter=resolved">Resolved {{ review_status.resolved_scoring_points }}</a>
    <a class="{% if active_filter == 'all' %}active{% endif %}" href="/jobs/{{ job.job_id }}/review?filter=all">All {{ review_status.scoring_points }}</a>
  </div>
</div>

{% if points %}
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
        <div class="stat"><strong>Review Status</strong><div>{{ point.review_status_label }}</div></div>
        <div class="stat"><strong>Current Winner</strong><div>{{ point.current_winner_label }}</div></div>
        <div class="stat"><strong>AI Winner</strong><div>{{ point.ai_winner_label }}</div></div>
        <div class="stat"><strong>Category</strong><div>{{ point.category }}</div></div>
        <div class="stat"><strong>Source</strong><div>{{ point.source }}</div></div>
      </div>
      <div class="meta" style="margin-top:12px;">{{ point.review_prompt }}</div>
      <div class="point-actions">
        {% if point.can_keep %}
        <form method="post" action="/jobs/{{ job.job_id }}/review/{{ point.id }}">
          <input type="hidden" name="filter" value="{{ active_filter }}">
          <input type="hidden" name="action" value="keep">
          <button type="submit">AI Correct</button>
        </form>
        {% endif %}
        <form method="post" action="/jobs/{{ job.job_id }}/review/{{ point.id }}">
          <input type="hidden" name="filter" value="{{ active_filter }}">
          <input type="hidden" name="action" value="set_winner">
          <input type="hidden" name="winner" value="player_a">
          <button class="secondary" type="submit">AI Wrong: {{ job.player_a_name }} Wins</button>
        </form>
        <form method="post" action="/jobs/{{ job.job_id }}/review/{{ point.id }}">
          <input type="hidden" name="filter" value="{{ active_filter }}">
          <input type="hidden" name="action" value="set_winner">
          <input type="hidden" name="winner" value="player_b">
          <button class="secondary" type="submit">AI Wrong: {{ job.player_b_name }} Wins</button>
        </form>
      </div>
      {% if point.last_note %}
        <div class="meta" style="margin-top:12px;">Last review note: {{ point.last_note }}</div>
      {% endif %}
    </div>
  </div>
</div>
{% endfor %}
{% else %}
<div class="panel">
  <div class="meta">No rallies match the current filter.</div>
</div>
{% endif %}
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
        blocked = not is_non_scoring and (point.winner_decision == "blocked" or point.winner not in {"player_a", "player_b"})
        if active_filter == "pending" and resolved:
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
                "resolved": resolved,
                "blocked": blocked,
                "can_keep": point.winner in {"player_a", "player_b"} or point.winner_candidate in {"player_a", "player_b"},
                "last_note": point.corrections[-1].note if point.corrections else "",
                "review_status_label": _review_status_label(point),
                "review_prompt": _review_prompt(point, job),
                "clip_src": f"/jobs/{job.job_id}/clips/{point.id}.mp4?ts={job.updated_at}",
                "play_label": f"Playing rally {point.id}",
                "is_non_scoring": is_non_scoring,
                "status_class": (
                    "let"
                    if is_non_scoring
                    else ("resolved-a" if resolved and point.winner == "player_a" else ("resolved-b" if resolved and point.winner == "player_b" else "pending"))
                ),
                "timeline_status_label": (
                    "LET / Hong"
                    if is_non_scoring
                    else (job.player_a_name if resolved and point.winner == "player_a" else (job.player_b_name if resolved and point.winner == "player_b" else "Chua duyet"))
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
    return f"{m} phút {s} giây"


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
        "generate_rally_timeline": (34, "Detecting rallies"),
        "export_review_clips": (55, "Cutting rally clips"),
        "predict_winners_with_adapter": (76, "Running winner AI"),
        "render_preview": (90, "Preparing preview artifacts"),
        "preview_ready": (94, "AI pipeline finished"),
        "review_required_no_preview": (92, "AI pipeline finished"),
        "preview_skipped_no_known_winner": (92, "AI pipeline finished"),
        "review_updated": (95, "Review in progress"),
        "final_export": (98, "Rendering export video"),
        "final_export_complete": (100, "Export complete"),
        "failed": (100, "Failed"),
    }
    base_percent, step_label = step_map.get(job.current_step, (8, job.current_step.replace("_", " ").strip() or "Processing"))
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


def _timeline_score_before_map(job: MatchJob, timeline) -> dict[str, MatchSnapshot]:
    engine = ScoreEngine(MatchState(best_of=job.best_of))
    snapshot_before = _initial_match_snapshot()
    result: dict[str, MatchSnapshot] = {}
    for point in timeline.points:
        result[point.id] = snapshot_before
        if counts_toward_score(point) and point_is_review_resolved(point) and point.winner in {"player_a", "player_b"}:
            snapshot_before = engine.process_event(RallyEvent(winner=point.winner, timestamp=float(point.t_end)))
    return result


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


def _start_heartbeat_watcher(timeout_sec: float = 20.0) -> "list[float]":
    import os
    import time

    last_beat: list[float] = [time.monotonic()]

    def _watch() -> None:
        while True:
            time.sleep(5)
            if time.monotonic() - last_beat[0] > timeout_sec:
                os._exit(0)

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    return last_beat


def create_local_web_app(
    *,
    config: ProductionPipelineConfig | None = None,
    jobs_root: Path | None = None,
):
    config = config or ProductionPipelineConfig()
    raw_matches_root = _raw_matches_root()
    raw_matches_root.mkdir(parents=True, exist_ok=True)
    runner = JobTaskRunner(config=config, jobs_root=jobs_root)
    last_beat = _start_heartbeat_watcher(timeout_sec=20.0)

    def app(environ, start_response):
        import time

        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = str(environ.get("PATH_INFO", "/"))
        message_ctx = _message_context(environ)

        if path == "/heartbeat" and method == "POST":
            last_beat[0] = time.monotonic()
            return _respond_text(start_response, "ok")

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
                ),
            )
            kind = "info" if ok else "error"
            return _redirect(start_response, f"/?job_id={job.job_id}&kind={kind}&message={quote_plus(msg)}")

        job_match = re.match(r"^/jobs/([^/]+)$", path)
        if job_match and method == "GET":
            job_id = job_match.group(1)
            return _redirect(start_response, f"/?job_id={job_id}")

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

        return _respond_text(start_response, f"Unhandled route: {escape(path)}", status="404 Not Found")

    return app
