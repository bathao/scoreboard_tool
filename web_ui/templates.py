from __future__ import annotations

from jinja2 import DictLoader, Environment, select_autoescape


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
        var em = (label || "").match(/(\\d+)\\s*min\\s*(\\d+)\\s*sec/);
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
              var isActive = (d.status === "running" || d.status === "created");
              fill.classList.toggle("running", isActive);
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
          _setText("prog_elapsed", mm + " min " + ss + " sec");
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
    .step31-review-shell {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .step31-workspace-panel {
      padding: 0;
      overflow: hidden;
      background: #14110e;
      border-color: #2f2922;
      color: #f7efe4;
    }
    .step31-hero {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 18px;
      padding: 20px 22px;
      background:
        radial-gradient(circle at 12% 0%, rgba(11, 110, 79, 0.28), transparent 38%),
        linear-gradient(135deg, #18120d 0%, #0f1512 100%);
      border-bottom: 1px solid #302820;
    }
    .step31-hero h2 {
      margin: 4px 0 8px 0;
      font-size: 24px;
      letter-spacing: -0.02em;
    }
    .step31-badges {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
      min-width: 260px;
    }
    .step31-pill {
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.08);
      border-radius: 999px;
      padding: 7px 11px;
      font-size: 12px;
      color: #e8dccb;
      white-space: nowrap;
    }
    .step31-pill strong {
      color: #fff;
      font-size: 14px;
      margin-right: 4px;
    }
    .step31-layout {
      display: grid;
      grid-template-columns: minmax(300px, 360px) minmax(0, 1fr);
      gap: 18px;
      padding: 18px;
      align-items: start;
    }
    .step31-sidebar {
      display: grid;
      gap: 12px;
      position: sticky;
      top: 12px;
    }
    .step31-side-card,
    .step31-table-card,
    .step31-frame-card {
      background: #0f0f0f;
      border: 1px solid #2e2e2e;
      border-radius: 14px;
      overflow: hidden;
    }
    .step31-side-card {
      padding: 14px;
    }
    .step31-side-title {
      color: #8fc;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .step31-mini-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .step31-mini-stat {
      border: 1px solid #2b2b2b;
      background: #171717;
      border-radius: 12px;
      padding: 10px;
    }
    .step31-mini-stat strong {
      display: block;
      font-size: 22px;
      color: #fff;
      line-height: 1;
      margin-bottom: 4px;
    }
    .step31-mini-stat span {
      color: #aaa;
      font-size: 11px;
    }
    .step31-video {
      width: 100%;
      max-width: none;
      border-radius: 12px;
      background: #000;
    }
    .step31-path {
      font-family: "Cascadia Code", "Consolas", monospace;
      font-size: 11px;
      color: #b8b0a6;
      word-break: break-all;
      line-height: 1.45;
    }
    .step31-table-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 13px 15px;
      border-bottom: 1px solid #303030;
      background: #151515;
    }
    .step31-table-wrap {
      max-height: 66vh;
      overflow: auto;
    }
    .step31-audit-table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 13px;
      min-width: 860px;
    }
    .step31-audit-table th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #1b1b1b;
      color: #b9b9b9;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border-bottom: 1px solid #383838;
    }
    .step31-audit-table th,
    .step31-audit-table td {
      padding: 10px 11px;
      border-bottom: 1px solid #262626;
      vertical-align: middle;
    }
    .step31-audit-table tbody tr:hover {
      background: #1c261f !important;
    }
    .step31-id {
      font-family: "Cascadia Code", "Consolas", monospace;
      color: #f0e8dc;
      font-weight: 700;
      white-space: nowrap;
    }
    .step31-time {
      font-family: "Cascadia Code", "Consolas", monospace;
      font-size: 15px;
      font-weight: 800;
      color: #fff;
      white-space: nowrap;
    }
    .step31-kind {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 8px;
      font-family: "Cascadia Code", "Consolas", monospace;
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .step31-kind.scoring {
      background: rgba(76, 175, 80, 0.16);
      color: #8fc;
      border: 1px solid rgba(143, 255, 204, 0.22);
    }
    .step31-kind.let {
      background: rgba(0, 188, 212, 0.14);
      color: #8fd;
      border: 1px solid rgba(136, 255, 221, 0.20);
    }
    .step31-kind.needs_review {
      background: rgba(217, 119, 6, 0.18);
      color: #fb8;
      border: 1px solid rgba(255, 187, 136, 0.28);
    }
    .step31-note {
      color: #fb8;
      max-width: 300px;
    }
    .step31-muted {
      color: #8d8882;
      font-size: 12px;
    }
    .step31-frame-card {
      margin: 0 18px 18px 18px;
    }
    .step31-frame-card summary {
      cursor: pointer;
      padding: 13px 15px;
      color: #ddd;
      font-weight: 700;
      background: #151515;
      border-bottom: 1px solid #303030;
    }
    .step31-thumb-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 10px;
      padding: 14px;
    }
    .step31-thumb {
      background: #111;
      border: 1px solid #333;
      border-radius: 10px;
      overflow: hidden;
    }
    .step31-thumb img {
      width: 100%;
      height: 108px;
      object-fit: cover;
      display: block;
    }
    .step31-actions,
    .step31-log {
      grid-column: 1 / -1;
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
    .set-score-table { width:100%; border-collapse:collapse; margin-top:8px; font-size:0.88em; }
    .set-score-table td { padding:3px 5px; vertical-align:middle; }
    .set-score-table .ss-label { color:#888; font-size:0.82em; white-space:nowrap; }
    .set-score-table .ss-num { font-weight:700; font-size:1.05em; text-align:center; min-width:28px; }
    .set-score-table .ss-dash { color:#ccc; text-align:center; }
    .set-score-table .ss-status { color:#4caf50; font-size:0.85em; padding-left:4px; }
    .set-score-table tr.ss-active { background:#f0fbf0; border-radius:6px; }
    .set-score-table tr.ss-active .ss-label { color:#2e7d32; font-weight:600; }
    .set-score-table .ss-none { color:#ccc; font-size:0.85em; }
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
    .timeline-set-header {
      padding: 4px 8px;
      margin-top: 10px;
      margin-bottom: 2px;
      font-size: 0.75em;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #6b7280;
      border-bottom: 1px solid #e5e7eb;
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
      position: relative;
      overflow: hidden;
    }
    @keyframes shimmer {
      0%   { transform: translateX(-100%); }
      100% { transform: translateX(400%); }
    }
    .progress-fill.running::after {
      content: "";
      position: absolute;
      top: 0; left: 0;
      width: 25%;
      height: 100%;
      background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.45) 50%, transparent 100%);
      animation: shimmer 1.6s ease-in-out infinite;
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
      .step31-review-shell,
      .reviewer-shell {
        grid-template-columns: 1fr;
      }
      .step31-layout {
        grid-template-columns: 1fr;
      }
      .step31-sidebar {
        position: static;
      }
      .step31-hero {
        flex-direction: column;
      }
      .step31-badges {
        justify-content: flex-start;
        min-width: 0;
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
        var m = elapsedEl.textContent.match(/(\\d+)\\s*min\\s*(\\d+)\\s*sec/);
        if (m) {
          var elapsedSec = parseInt(m[1]) * 60 + parseInt(m[2]);
          setInterval(function () {
            elapsedSec++;
            var mm = Math.floor(elapsedSec / 60);
            var ss = elapsedSec % 60;
            elapsedEl.textContent = mm + " min " + ss + " sec";
          }, 1000);
        }
      }
      // Scroll the currently selected rally into view in the timeline list
      var cur = document.querySelector(".timeline-item.current");
      if (cur) cur.scrollIntoView({ block: "nearest", behavior: "instant" });
    });
    function _sendHeartbeat() {
      fetch("/heartbeat", { method: "POST" }).catch(function () {});
    }
    setInterval(_sendHeartbeat, 20000);
    document.addEventListener("visibilitychange", function () {
      if (!document.hidden) _sendHeartbeat();
    });

    {% if screen_mode == 'review' and all_points_data %}
    var POINT_DATA = {{ all_points_data | tojson }};
    var _REVIEW_JOB_ID = {{ current_job.job_id | tojson }};
    var _REVIEW_FILTER = {{ active_filter | tojson }};

    function selectPoint(id) {
      var d = POINT_DATA[id];
      if (!d) return true;

      // Update video player
      var player = document.getElementById("main_video_player");
      if (player && d.clip_src) {
        player.src = d.clip_src;
        player.load();
        player.play().catch(function(){});
      }

      // Update "Before" score box
      var lbl = document.getElementById("score_before_label");
      if (lbl) lbl.textContent = "Before " + id;
      var si = document.getElementById("score_set_info");
      if (si) si.textContent = "Set " + d.set_number + " | Sets " + d.sets_a + "-" + d.sets_b;
      var sa = document.getElementById("score_a_val");
      if (sa) sa.textContent = d.score_a;
      var sb = document.getElementById("score_b_val");
      if (sb) sb.textContent = d.score_b;

      // Update main panel
      var title = document.getElementById("main_panel_title");
      if (title) title.textContent = id;
      var aiLbl = document.getElementById("ai_winner_text");
      if (aiLbl) aiLbl.textContent = d.ai_winner_label;
      var needsWarn = document.getElementById("needs_input_warning");
      if (needsWarn) needsWarn.style.display = d.needs_input ? "inline" : "none";
      var corrBadge = document.getElementById("manually_corrected_badge");
      if (corrBadge) corrBadge.style.display = d.manually_corrected ? "inline" : "none";

      // Update player labels and button text for this point
      var nearLbl = document.getElementById("near_player_label");
      if (nearLbl) nearLbl.textContent = d.near_abbrev || "";
      var farLbl = document.getElementById("far_player_label");
      if (farLbl) farLbl.textContent = d.far_abbrev || "";
      var setLbl = document.getElementById("side_set_label");
      if (setLbl) setLbl.textContent = d.set_number || "";
      var nearBtn = document.getElementById("review_action_near");
      if (nearBtn) nearBtn.textContent = (d.near_abbrev || "Player 1") + " WIN";
      var farBtn = document.getElementById("review_action_far");
      if (farBtn) farBtn.textContent = (d.far_abbrev || "Player 2") + " WIN";

      // Update review form actions and hidden inputs
      var nearForm = document.getElementById("form_near_win");
      if (nearForm) nearForm.action = "/jobs/" + _REVIEW_JOB_ID + "/review/" + id;
      var farForm = document.getElementById("form_far_win");
      if (farForm) farForm.action = "/jobs/" + _REVIEW_JOB_ID + "/review/" + id;
      document.querySelectorAll(".input_current_point").forEach(function(inp) { inp.value = id; });

      // Update timeline highlight
      document.querySelectorAll(".timeline-item").forEach(function(el) { el.classList.remove("current"); });
      var cur = document.querySelector(".timeline-item[data-id='" + id + "']");
      if (cur) cur.classList.add("current");

      // Update URL without page reload
      try {
        var p = new URLSearchParams(window.location.search);
        p.set("current_point", id);
        history.pushState({}, "", "?" + p.toString());
      } catch(e) {}

      return false;
    }
    {% endif %}
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
{% if screen_mode == "exporting" and current_job %}
<div class="setup-shell">
  <div class="panel" data-job-id="{{ current_job.job_id }}" data-job-status="{{ current_job.status }}">
    <div class="row" style="justify-content:space-between;align-items:center;margin-bottom:14px;">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
        <span class="badge running" style="font-size:13px;padding:4px 10px;">● EXPORTING</span>
        <span style="font-weight:600;">{{ current_job.player_a_name }} vs {{ current_job.player_b_name }}</span>
        <span class="meta">Final scoreboard video</span>
      </div>
      <span class="meta" id="prog_elapsed">{{ progress.elapsed_label }}</span>
    </div>
    <div class="progress-card" data-job-id="{{ current_job.job_id }}" data-job-status="{{ current_job.status }}" style="margin-bottom:0;border:none;padding:0;background:none;box-shadow:none;">
      <div class="row" style="justify-content:space-between;margin-bottom:4px;">
        <span id="prog_step" class="meta">{{ progress.step_label }}</span>
        <span id="progress_pct_label" style="font-weight:600;">{{ progress.percent }}%</span>
      </div>
      <div class="progress-bar"><div class="progress-fill running" style="width: {{ progress.percent }}%;" data-server-pct="{{ progress.percent }}" data-job-id="{{ current_job.job_id }}"></div></div>
    </div>
  </div>
  <div class="panel" style="padding:0;overflow:hidden;">
    <div style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid var(--line);background:#111;">
      <span style="color:#8fc;font-size:12px;font-family:monospace;font-weight:600;">export.log — {{ current_job.job_id }}</span>
      <span id="prog_label" style="color:#aaa;font-size:12px;">{{ progress.label }}</span>
    </div>
    <pre id="pipeline_log" style="margin:0;background:#111;color:#d4d0c8;font-size:12.5px;line-height:1.6;padding:16px;height:420px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;font-family:'Cascadia Code','Consolas','Fira Mono',monospace;">(waiting for export to start...)</pre>
  </div>
  <script>
    (function () {
      var jobId = "{{ current_job.job_id }}";
      var _pinned = true;
      var logEl = document.getElementById("pipeline_log");
      if (logEl) {
        logEl.addEventListener("scroll", function () {
          _pinned = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 20;
        });
      }
      function fetchLog() {
        fetch("/jobs/" + encodeURIComponent(jobId) + "/log")
          .then(function (r) { return r.ok ? r.text() : null; })
          .then(function (t) {
            if (t === null || !logEl) return;
            logEl.textContent = t || "(no output yet)";
            if (_pinned) logEl.scrollTop = logEl.scrollHeight;
          }).catch(function () {});
      }
      fetchLog();
      setInterval(fetchLog, 3000);
    })();
  </script>
</div>
{% elif screen_mode == "review" and current_job and current_point %}
<div class="reviewer-shell">
  <div class="panel">
    <h2>The Reviewer</h2>
    <div class="meta">{{ current_job.player_a_name }} vs {{ current_job.player_b_name }}</div>
    <div class="meta" style="margin-top:4px;">Current rally: {{ current_point.id }} | {{ current_point.time_range }}</div>
    <div class="progress-card" data-job-id="{{ current_job.job_id }}" data-job-status="{{ current_job.status }}">
      <div class="row" style="justify-content:space-between;">
        <strong id="prog_label">{{ progress.label }}</strong>
        <span id="progress_pct_label">{{ progress.percent }}%</span>
      </div>
      <div class="progress-bar"><div class="progress-fill{% if current_job and current_job.status in ('running', 'created') %} running{% endif %}" style="width: {{ progress.percent }}%;" data-server-pct="{{ progress.percent }}" data-job-id="{{ current_job.job_id if current_job else '' }}"></div></div>
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
    <div class="scoreboard-live" style="border-left:3px solid #4caf50;">
      <div class="meta" style="font-weight:700;color:#2e7d32;margin-bottom:4px;">
        Match Score — {{ current_job.player_a_name }} vs {{ current_job.player_b_name }}
        {% if final_scoreboard.is_finished %}<span style="color:#888;font-weight:400;font-size:0.85em;"> (finished)</span>{% endif %}
      </div>
      <table class="set-score-table">
        {% for s in set_scores_display %}
        <tr class="{% if s.active %}ss-active{% endif %}">
          <td class="ss-label">Set {{ s.set_num }}</td>
          {% if s.score_a is not none %}
            <td class="ss-num" style="color:#1565c0;">{{ s.score_a }}</td>
            <td class="ss-dash">—</td>
            <td class="ss-num" style="color:#b71c1c;">{{ s.score_b }}</td>
            <td class="ss-status">{% if s.done %}✓{% else %}●{% endif %}</td>
          {% else %}
            <td class="ss-none" colspan="4">not started</td>
          {% endif %}
        </tr>
        {% endfor %}
      </table>
    </div>
    <div class="scoreboard-live" style="margin-top:10px;">
      <div class="meta" id="score_before_label">Before {{ current_point.id }}</div>
      <div class="meta" id="score_set_info">Set {{ scoreboard.set_number }} | Sets {{ scoreboard.sets_a }}-{{ scoreboard.sets_b }}</div>
      <div class="score-row" style="margin-top:8px;">
        <div class="score-name">{{ current_job.player_a_name }}</div>
        <div class="score-big" id="score_a_val">{{ scoreboard.score_a }}</div>
      </div>
      <div class="score-row">
        <div class="score-name">{{ current_job.player_b_name }}</div>
        <div class="score-big" id="score_b_val">{{ scoreboard.score_b }}</div>
      </div>
    </div>
  </div>
  <div class="panel" id="main-panel">
    <h2 id="main_panel_title">{{ current_point.id }}</h2>
    <div class="hint-box">← {{ current_near_abbrev }} Win &nbsp;|&nbsp; → {{ current_far_abbrev }} Win</div>
    <div class="meta" style="margin-top:12px;">
      AI: <strong id="ai_winner_text">{{ current_point.ai_winner_label }}</strong>
      <span id="manually_corrected_badge" style="color:#f39c12;display:{% if current_point.manually_corrected %}inline{% else %}none{% endif %}"> (manually corrected)</span>
      <span id="needs_input_warning" style="color:#e74c3c;display:{% if current_point.needs_input %}inline{% else %}none{% endif %}"> — no prediction, input required</span>
    </div>
    <div class="meta" id="side_context_line" style="margin-top:8px;color:#888;font-size:0.85em;">
      Set <span id="side_set_label">{{ current_point.set_number }}</span>
      &nbsp;·&nbsp;
      <span id="near_player_label">{{ current_near_abbrev }}</span> vs <span id="far_player_label">{{ current_far_abbrev }}</span>
    </div>
    <div class="action-grid" style="margin-top:14px;">
      <form id="form_near_win" method="post" action="/jobs/{{ current_job.job_id }}/review/{{ current_point.id }}">
        <input type="hidden" name="filter" value="{{ active_filter }}">
        <input type="hidden" name="current_point" value="{{ current_point.id }}" class="input_current_point">
        <input type="hidden" name="action" value="set_winner">
        <input type="hidden" name="winner" value="player_a">
        <button id="review_action_near" class="action-btn near" type="submit">{{ current_near_abbrev }} WIN</button>
      </form>
      <form id="form_far_win" method="post" action="/jobs/{{ current_job.job_id }}/review/{{ current_point.id }}">
        <input type="hidden" name="filter" value="{{ active_filter }}">
        <input type="hidden" name="current_point" value="{{ current_point.id }}" class="input_current_point">
        <input type="hidden" name="action" value="set_winner">
        <input type="hidden" name="winner" value="player_b">
        <button id="review_action_far" class="action-btn far" type="submit">{{ current_far_abbrev }} WIN</button>
      </form>
    </div>
    <div class="point-actions" style="margin-top:16px;">
      <a class="subtle-link" href="/?job_id={{ current_job.job_id }}&review_filter={% if active_filter == 'pending' %}all{% else %}pending{% endif %}&current_point={{ current_point.id }}">
        {% if active_filter == 'pending' %}Show All Points{% else %}Review Difficult Only{% endif %}
      </a>
      <form method="post" action="/jobs/{{ current_job.job_id }}/final-export">
        <button class="warn" type="submit">Export</button>
      </form>
    </div>
    <div class="timeline-list">
      {% set ns = namespace(last_set=0) %}
      {% for point in points %}
        {% if point.set_number != ns.last_set %}
          {% set ns.last_set = point.set_number %}
          {% set set_idx = point.set_number - 1 %}
          {% set set_info = set_scores_display[set_idx] if set_idx < set_scores_display | length else none %}
          <div class="timeline-set-header">
            SET {{ point.set_number }}
            {% if set_info and set_info.done %}
              <span class="meta" style="margin-left:6px;">{{ set_info.score_a }} – {{ set_info.score_b }}</span>
            {% elif set_info and set_info.active %}
              <span class="meta" style="margin-left:6px;">{{ set_info.score_a }} – {{ set_info.score_b }} (ongoing)</span>
            {% endif %}
          </div>
        {% endif %}
        <a class="timeline-item {{ point.status_class }} {% if point.id == current_point_id %}current{% endif %}" href="/?job_id={{ current_job.job_id }}&review_filter={{ active_filter }}&current_point={{ point.id }}" data-id="{{ point.id }}" onclick="return selectPoint('{{ point.id }}');">
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
<div class="{% if current_job and current_job.status == 'awaiting_confirmation' and current_job.current_step == 'confirm_total_rallies' %}step31-review-shell{% else %}setup-shell{% endif %}">
  {% if current_job and current_job.status in ('running', 'created') %}
  {# ── PIPELINE RUNNING VIEW ── #}
  <div class="panel{% if current_job.current_step == 'confirm_total_rallies' %} step31-workspace-panel{% endif %}" data-job-id="{{ current_job.job_id }}" data-job-status="{{ current_job.status }}">
    <div class="row" style="justify-content:space-between;align-items:center;margin-bottom:14px;">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
        <span class="badge running" style="font-size:13px;padding:4px 10px;">● RUNNING</span>
        <span style="font-weight:600;">{{ current_job.player_a_name }} vs {{ current_job.player_b_name }}</span>
        <span class="meta">{{ Path(current_job.raw_video_path).name if current_job.raw_video_path else '' }}</span>
        {% if current_job.trim_start_sec %}<span class="meta">trim {{ current_job.trim_start_sec }}s</span>{% endif %}
        <span class="meta">Best of {{ current_job.best_of }}</span>
      </div>
      <div style="display:flex;align-items:center;gap:12px;">
        <span class="meta" id="prog_elapsed">{{ progress.elapsed_label }}</span>
        <form method="post" action="/jobs/{{ current_job.job_id }}/stop" style="margin:0;">
          <button type="submit" class="secondary" style="padding:5px 14px;font-size:13px;" onclick="return confirm('Stop after current step finishes?')">Stop</button>
        </form>
      </div>
    </div>
    <div class="progress-card" data-job-id="{{ current_job.job_id }}" data-job-status="{{ current_job.status }}" style="margin-bottom:0;border:none;padding:0;background:none;box-shadow:none;">
      <div class="row" style="justify-content:space-between;margin-bottom:4px;">
        <span id="prog_step" class="meta">{{ progress.step_label }}</span>
        <span id="progress_pct_label" style="font-weight:600;">{{ progress.percent }}%</span>
      </div>
      <div class="progress-bar"><div class="progress-fill running" style="width: {{ progress.percent }}%;" data-server-pct="{{ progress.percent }}" data-job-id="{{ current_job.job_id }}"></div></div>
    </div>
  </div>
  <div class="panel" style="padding:0;overflow:hidden;">
    <div style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid var(--line);background:#111;">
      <span style="color:#8fc;font-size:12px;font-family:monospace;font-weight:600;">pipeline.log — {{ current_job.job_id }}</span>
      <span id="prog_label" style="color:#aaa;font-size:12px;">{{ progress.label }}</span>
    </div>
    <pre id="pipeline_log" style="margin:0;background:#111;color:#d4d0c8;font-size:12.5px;line-height:1.6;padding:16px;height:420px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;font-family:'Cascadia Code','Consolas','Fira Mono',monospace;">(waiting for first log entry...)</pre>
  </div>
  <script>
    (function () {
      var jobId = "{{ current_job.job_id }}";
      var _pinned = true;
      var logEl = document.getElementById("pipeline_log");
      if (logEl) {
        logEl.addEventListener("scroll", function () {
          _pinned = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 20;
        });
      }
      function fetchLog() {
        fetch("/jobs/" + encodeURIComponent(jobId) + "/log")
          .then(function (r) { return r.ok ? r.text() : null; })
          .then(function (t) {
            if (t === null || !logEl) return;
            logEl.textContent = t || "(no output yet)";
            if (_pinned) logEl.scrollTop = logEl.scrollHeight;
          }).catch(function () {});
      }
      fetchLog();
      setInterval(fetchLog, 3000);
    })();
  </script>
  {% elif current_job and current_job.status == 'awaiting_confirmation' %}
  {# ── CONFIRMATION PAUSE VIEW ── operator reviews output, clicks Next ── #}
  <div class="panel" data-job-id="{{ current_job.job_id }}" data-job-status="{{ current_job.status }}">
    <div class="row" style="justify-content:space-between;align-items:center;margin-bottom:14px;">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
        <span class="badge" style="font-size:13px;padding:4px 10px;background:#f0ad4e;color:#000;">⏸ AWAITING CONFIRMATION</span>
        <span style="font-weight:600;">{{ current_job.player_a_name }} vs {{ current_job.player_b_name }}</span>
        <span class="meta">Best of {{ current_job.best_of }}</span>
      </div>
    </div>

    {% if current_job.current_step == 'confirm_total_rallies' %}
      {% set rally_data = current_job.timeline_summary.get('detected_total_rallies', {}) %}
      {% set events = rally_data.get('events', []) %}
      <div class="step31-hero">
        <div>
          <div class="step31-muted" style="text-transform:uppercase;letter-spacing:0.08em;font-weight:800;color:#8fc;">Debug Pause</div>
          <h2>Step 3.1 - Rally Start Review</h2>
          <div class="step31-muted">
            Review every detected start-time before Step 3.2. If one rally or LET is missing here, all downstream side/winner logic will be wrong.
          </div>
        </div>
        <div class="step31-badges">
          <span class="step31-pill"><strong>{{ rally_data.get('total', 0) }}</strong> total starts</span>
          <span class="step31-pill"><strong>{{ rally_data.get('scoring', 0) }}</strong> scoring</span>
          <span class="step31-pill"><strong>{{ rally_data.get('lets', 0) }}</strong> LET/non-scoring</span>
          {% if rally_data.get('needs_review', 0) %}
          <span class="step31-pill"><strong>{{ rally_data.get('needs_review', 0) }}</strong> needs review</span>
          {% endif %}
        </div>
      </div>

      <div class="step31-layout">
        <aside class="step31-sidebar">
          <div class="step31-side-card">
            <div class="step31-side-title">Video Reference</div>
            {% if full_match_src %}
            <video class="step31-video" controls preload="metadata" src="{{ full_match_src }}"></video>
            <div class="step31-muted" style="margin-top:8px;">Use the player to jump to a table start-time, then compare against the audit table.</div>
            {% else %}
            <div class="step31-muted">No working video source is available for this job.</div>
            {% endif %}
          </div>

          <div class="step31-side-card">
            <div class="step31-side-title">Quick Counts</div>
            <div class="step31-mini-grid">
              <div class="step31-mini-stat"><strong>{{ rally_data.get('total', 0) }}</strong><span>Total starts</span></div>
              <div class="step31-mini-stat"><strong>{{ rally_data.get('scoring', 0) }}</strong><span>Scoring rallies</span></div>
              <div class="step31-mini-stat"><strong>{{ rally_data.get('lets', 0) }}</strong><span>LET/non-scoring</span></div>
              <div class="step31-mini-stat"><strong>{{ rally_data.get('needs_review', 0) }}</strong><span>Needs review</span></div>
            </div>
            {% if rally_data.get('rule_gap_review_count', 0) or rally_data.get('rule_conflict_review_count', 0) %}
            <div class="step31-muted" style="margin-top:10px;">
              Gap markers: {{ rally_data.get('rule_gap_review_count', 0) }} · Rule conflicts: {{ rally_data.get('rule_conflict_review_count', 0) }}
            </div>
            {% endif %}
          </div>

          <div class="step31-side-card">
            <div class="step31-side-title">Artifacts</div>
            {% if rally_data.get('algorithm') %}
            <div class="step31-muted">Algorithm</div>
            <div class="step31-path" style="margin-bottom:10px;">{{ rally_data.get('algorithm') }}</div>
            {% endif %}
            {% if rally_data.get('error') %}
            <div style="color:#ffb4a8;margin-bottom:10px;">Error: {{ rally_data.get('error') }}</div>
            {% endif %}
            <div class="step31-muted">Start frame folder</div>
            <div class="step31-path" style="margin-bottom:10px;">{{ rally_data.get('start_frames_dir', '-') }}</div>
            <div class="step31-muted">CSV</div>
            <div class="step31-path">{{ rally_data.get('csv_path', '-') }}</div>
          </div>

          <div class="step31-side-card">
            <div class="step31-side-title">Review Checklist</div>
            <div class="step31-muted" style="line-height:1.6;">
              1. Check total count first.<br>
              2. Scan the Start column in order.<br>
              3. Verify LET rows are labeled correctly.<br>
              4. Note exact missing/false timestamps for the next debug pass.
            </div>
          </div>
        </aside>

        <main class="step31-table-card">
          <div class="step31-table-head">
            <div>
              <strong style="color:#8fc;">Start-Time Audit Table</strong>
              <div class="step31-muted">Primary review surface. Times are in final input video time.</div>
            </div>
            <div class="step31-muted">{{ events|length }} row(s)</div>
          </div>
          {% if events %}
          <div class="step31-table-wrap">
            <table class="step31-audit-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Type</th>
                <th>Start</th>
                <th>End</th>
                <th>Server</th>
                <th>Source</th>
                <th>Note</th>
                <th>Image</th>
              </tr>
            </thead>
            <tbody>
              {% for e in events %}
              {% set st = e.get('source_t_start', e.get('t_start', 0.0)) %}
              {% set et = e.get('source_t_end', e.get('t_end', st)) %}
              <tr style="background:{% if e.get('kind') == 'let' %}#122228{% elif e.get('kind') == 'needs_review' or e.get('review_reason') %}#281f15{% else %}transparent{% endif %};">
                <td><span class="step31-id">{{ e.get('id', '') }}</span></td>
                <td>
                  <span class="step31-kind {{ e.get('kind', '') }}">{{ e.get('kind', '') }}</span>
                </td>
                <td><span class="step31-time">{{ fmt_time(st) }}</span></td>
                <td><span class="step31-muted">{{ fmt_time(et) }}</span></td>
                <td>{{ e.get('server_player_name', 'unknown') }}</td>
                <td><span class="step31-muted">{{ e.get('source', '') }} {{ e.get('point_id', '') }}</span></td>
                <td class="step31-note">{{ e.get('review_reason', '') }}</td>
                <td>
                  {% if e.get('image_file') %}
                  <a href="/jobs/{{ current_job.job_id }}/rally-start-frames/{{ e.get('image_file') }}" target="_blank" style="color:#8fc;">open</a>
                  {% endif %}
                </td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
          </div>
          {% else %}
          <div style="padding:18px;color:#fb8;">No Step 3.1 events were exported.</div>
          {% endif %}
        </main>
      </div>

      {% if events %}
      <details class="step31-frame-card">
        <summary>Annotated start frames ({{ events|length }})</summary>
        <div class="step31-thumb-grid">
        {% for e in events %}
        <div class="step31-thumb">
          {% if e.get('image_file') %}
          <a href="/jobs/{{ current_job.job_id }}/rally-start-frames/{{ e.get('image_file') }}" target="_blank">
            <img src="/jobs/{{ current_job.job_id }}/rally-start-frames/{{ e.get('image_file') }}">
          </a>
          {% endif %}
          <div style="padding:8px;">
            <div style="font-weight:700;font-size:13px;">{{ e.get('id') }} - {{ e.get('kind') }}</div>
            <div style="color:#ccc;font-size:12px;">start {{ "%.3f"|format(e.get('t_start', 0.0)) }}s</div>
            {% if e.get('server_player_name') %}
            <div style="color:#bbb;font-size:11px;">server {{ e.get('server_player_name') }} · side {{ e.get('current_side', 'unknown') }}</div>
            {% endif %}
            {% if e.get('side_evidence_status') %}
            <div style="color:#888;font-size:11px;">
              side evidence {{ e.get('side_evidence_status') }}
              {% if e.get('side_identified_player_name') %}: {{ e.get('side_identified_player_name') }}={{ e.get('side_identified_current_side') }}{% endif %}
            </div>
            {% endif %}
            {% if e.get('review_reason') %}
            <div style="color:#fb8;font-size:11px;">{{ e.get('review_reason') }}</div>
            {% endif %}
            <div style="color:#777;font-size:11px;">{{ e.get('source') }} {{ e.get('point_id') }}</div>
          </div>
        </div>
        {% endfor %}
        </div>
      </details>
      {% endif %}

    {% elif current_job.current_step == 'confirm_side_state' %}
    <div style="background:#1a2a1a;border:1px solid #3a5a3a;border-radius:8px;padding:16px;margin-bottom:16px;">
      <h3 style="margin:0 0 10px 0;color:#8fc;">Step 3.2 - Side State Detection Complete</h3>
      {% set side_data = current_job.timeline_summary.get('detected_side_state', {}) %}
      {% set side_id = side_data.get('side_identification', {}) %}
      {% set events = side_data.get('events', []) %}
      <p style="font-size:16px;font-weight:700;color:#fff;margin:0 0 12px;">
        {{ side_id.get('identified', 0) }} identified / {{ side_id.get('inferred', 0) }} inferred / {{ side_id.get('unknown', 0) }} unknown
        <span style="color:#aaa;font-size:13px;font-weight:500;">
          {% if side_id.get('retry_attempted', 0) %} / retry {{ side_id.get('retry_identified', 0) }} of {{ side_id.get('retry_attempted', 0) }} unknown row(s){% endif %}
          {% if side_id.get('continuity_filled', 0) %} / continuity filled {{ side_id.get('continuity_filled', 0) }}{% endif %}
        </span>
      </p>
      {% if side_data.get('algorithm') or side_data.get('error') %}
      <p style="color:#aaa;font-size:12px;margin:0 0 10px;">
        {% if side_data.get('algorithm') %}Algorithm: <code>{{ side_data.get('algorithm') }}</code>{% endif %}
        {% if side_data.get('error') %} &nbsp; Error: <code>{{ side_data.get('error') }}</code>{% endif %}
      </p>
      {% endif %}
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px;margin:12px 0;">
        <div style="background:#111;border:1px solid #333;border-radius:6px;padding:10px;">
          <div style="color:#aaa;font-size:12px;margin-bottom:4px;">Side-state frame folder</div>
          <code style="font-size:12px;word-break:break-all;">{{ side_data.get('start_frames_dir', '-') }}</code>
        </div>
        <div style="background:#111;border:1px solid #333;border-radius:6px;padding:10px;">
          <div style="color:#aaa;font-size:12px;margin-bottom:4px;">Events JSON</div>
          <code style="font-size:12px;word-break:break-all;">{{ side_data.get('events_json_path', '-') }}</code>
        </div>
      </div>
      {% if events %}
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px;margin:12px 0 14px;">
        {% for e in events %}
        <div style="background:#111;border:1px solid {% if e.get('current_side') == 'unknown' %}#7a4{% else %}#333{% endif %};border-radius:8px;overflow:hidden;">
          {% if e.get('image_file') %}
          <a href="/jobs/{{ current_job.job_id }}/rally-start-frames/{{ e.get('image_file') }}" target="_blank">
            <img src="/jobs/{{ current_job.job_id }}/rally-start-frames/{{ e.get('image_file') }}" style="width:100%;height:110px;object-fit:cover;display:block;">
          </a>
          {% endif %}
          <div style="padding:8px;">
            <div style="font-weight:700;font-size:13px;">{{ e.get('id') }} - {{ e.get('kind') }}</div>
            <div style="color:#ccc;font-size:12px;">source {{ "%.3f"|format(e.get('source_t_start', e.get('t_start', 0.0))) }}s</div>
            <div style="color:#bbb;font-size:11px;">server {{ e.get('server_player_name', 'unknown') }} Â· side {{ e.get('current_side', 'unknown') }}</div>
            <div style="color:#888;font-size:11px;">
              {{ e.get('side_evidence_window_mode', '') }}
              {% if e.get('side_identified_player_name') %}: {{ e.get('side_identified_player_name') }}={{ e.get('side_identified_current_side') }}{% endif %}
            </div>
            {% if e.get('side_evidence_reason') %}
            <div style="color:#fb8;font-size:11px;">{{ e.get('side_evidence_reason') }}</div>
            {% endif %}
          </div>
        </div>
        {% endfor %}
      </div>
      {% endif %}
      {% if step3_summary_md %}
      <div style="margin-top:16px;">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:8px;">
          <strong style="color:#8fc;font-size:13px;">summary.md</strong>
          <code style="color:#aaa;font-size:11px;word-break:break-all;">{{ step3_summary_path }}</code>
        </div>
        <pre style="background:#0f0f0f;color:#d8d8d8;border:1px solid #333;border-radius:8px;padding:14px;max-height:520px;overflow:auto;white-space:pre-wrap;word-break:normal;font-family:'Cascadia Code','Consolas','Fira Mono',monospace;font-size:12px;line-height:1.55;">{{ step3_summary_md }}</pre>
      </div>
      {% endif %}
      <p style="color:#aaa;font-size:13px;margin:10px 0 0;">Debug pause: Step 3.2 is complete. Review summary.md here and stop before Step 3.3/Step 4.</p>
    </div>

    {% elif current_job.current_step == 'confirm_sets' %}
    <div style="background:#1a2a1a;border:1px solid #3a5a3a;border-radius:8px;padding:16px;margin-bottom:16px;">
      <h3 style="margin:0 0 10px 0;color:#8fc;">Step 3.1 — Set Detection Complete</h3>
      {% set sets_data = current_job.timeline_summary.get('detected_sets', {}) %}
      {% set n_sets = sets_data.get('n_sets', 0) %}
      {% set swaps = sets_data.get('swaps', []) %}
      {% set break_candidates = sets_data.get('break_candidates', []) %}
      <p style="font-size:16px;font-weight:700;color:#fff;margin:0 0 12px;">{{ n_sets }} set(s) detected</p>
      {% if sets_data.get('algorithm') or sets_data.get('note') %}
      <p style="color:#aaa;font-size:12px;margin:0 0 10px;">
        {% if sets_data.get('algorithm') %}Algorithm: <code>{{ sets_data.get('algorithm') }}</code>{% endif %}
        {% if sets_data.get('note') %} &nbsp; Note: <code>{{ sets_data.get('note') }}</code>{% endif %}
      </p>
      {% endif %}
      {% if swaps %}
      <table style="width:100%;border-collapse:collapse;margin-bottom:10px;">
        <tr style="border-bottom:1px solid #333;">
          <th style="text-align:left;padding:6px 12px;color:#aaa;font-size:12px;">Swap #</th>
          <th style="text-align:left;padding:6px 12px;color:#aaa;font-size:12px;">Break Window</th>
          <th style="text-align:left;padding:6px 12px;color:#aaa;font-size:12px;">Set Boundary</th>
        </tr>
        {% for s in swaps %}
        <tr style="border-bottom:1px solid #222;">
          <td style="padding:6px 12px;">{{ loop.index }}</td>
          <td style="padding:6px 12px;">
            {% if s.get('t_break_start') is not none and s.get('t_break_end') is not none %}
              {{ "%.1f"|format(s.t_break_start) }}s — {{ "%.1f"|format(s.t_break_end) }}s
            {% else %}
              ~{{ "%.1f"|format(s.t_cutoff) }}s
            {% endif %}
          </td>
          <td style="padding:6px 12px;font-weight:600;">Set {{ loop.index }} → Set {{ loop.index + 1 }}</td>
        </tr>
        {% endfor %}
      </table>
      {% endif %}
      {% if break_candidates %}
      <details style="margin:10px 0;">
        <summary style="cursor:pointer;color:#ccc;font-size:13px;">Break candidates ({{ break_candidates|length }})</summary>
        <table style="width:100%;border-collapse:collapse;margin-top:8px;">
          <tr style="border-bottom:1px solid #333;">
            <th style="text-align:left;padding:5px 10px;color:#aaa;font-size:12px;">#</th>
            <th style="text-align:left;padding:5px 10px;color:#aaa;font-size:12px;">Window</th>
            <th style="text-align:right;padding:5px 10px;color:#aaa;font-size:12px;">Duration</th>
            <th style="text-align:right;padding:5px 10px;color:#aaa;font-size:12px;">Avg Energy</th>
          </tr>
          {% for c in break_candidates %}
          <tr style="border-bottom:1px solid #222;">
            <td style="padding:5px 10px;">{{ loop.index }}</td>
            <td style="padding:5px 10px;">{{ "%.1f"|format(c.t_break_start) }}s - {{ "%.1f"|format(c.t_break_end) }}s</td>
            <td style="padding:5px 10px;text-align:right;">{{ "%.1f"|format(c.duration) }}s</td>
            <td style="padding:5px 10px;text-align:right;">{{ "%.2f"|format(c.avg_energy) }}</td>
          </tr>
          {% endfor %}
        </table>
      </details>
      {% endif %}
      <p style="color:#aaa;font-size:13px;margin:10px 0 0;">If set count and swap times are correct, click <strong>Next</strong> to detect rallies per set.</p>
    </div>

    {% elif current_job.current_step == 'confirm_rallies' %}
    <div style="background:#1a2a1a;border:1px solid #3a5a3a;border-radius:8px;padding:16px;margin-bottom:16px;">
      <h3 style="margin:0 0 10px 0;color:#8fc;">Step 3.2 — Rally Detection Complete</h3>
      {% set per_set = current_job.timeline_summary.get('per_set_rallies', []) %}
      {% set total = current_job.timeline_summary.get('total_rallies', 0) %}
      <table style="width:100%;border-collapse:collapse;margin-bottom:10px;">
        <tr style="border-bottom:1px solid #333;">
          <th style="text-align:left;padding:6px 12px;color:#aaa;font-size:12px;">Set</th>
          <th style="text-align:right;padding:6px 12px;color:#aaa;font-size:12px;">Scoring Rallies</th>
          <th style="text-align:right;padding:6px 12px;color:#aaa;font-size:12px;">LETs</th>
          <th style="text-align:right;padding:6px 12px;color:#aaa;font-size:12px;">Total</th>
        </tr>
        {% for s in per_set %}
        <tr style="border-bottom:1px solid #222;">
          <td style="padding:6px 12px;font-weight:600;">Set {{ s.set }}</td>
          <td style="padding:6px 12px;text-align:right;">{{ s.scoring }}</td>
          <td style="padding:6px 12px;text-align:right;">{{ s.lets }}</td>
          <td style="padding:6px 12px;text-align:right;font-weight:600;">{{ s.total }}</td>
        </tr>
        {% endfor %}
        <tr style="border-top:2px solid #555;">
          <td style="padding:6px 12px;font-weight:700;">TOTAL</td>
          <td colspan="2"></td>
          <td style="padding:6px 12px;text-align:right;font-weight:700;">{{ total }}</td>
        </tr>
      </table>
      <p style="color:#aaa;font-size:13px;margin:10px 0 0;">If rally counts are correct, click <strong>Next</strong> to export clips and predict winners.</p>
    </div>
    {% endif %}

    <div class="{% if current_job.current_step == 'confirm_total_rallies' %}step31-actions{% endif %}" style="display:flex;gap:10px;margin-bottom:16px;">
      {% if current_job.current_step in ['confirm_total_rallies', 'confirm_sets', 'confirm_rallies'] %}
      <form method="post" action="/jobs/{{ current_job.job_id }}/next-step" style="margin:0;">
        <button type="submit" style="padding:10px 32px;font-size:15px;font-weight:700;background:#4a9;color:#000;border:none;border-radius:6px;cursor:pointer;">▶ Next</button>
      </form>
      {% else %}
      <span class="meta" style="align-self:center;">Waiting for operator feedback before the next Step 3 stage.</span>
      {% endif %}
      <form method="post" action="/jobs/{{ current_job.job_id }}/stop" style="margin:0;">
        <button type="submit" class="secondary" style="padding:10px 20px;font-size:13px;" onclick="return confirm('Stop pipeline?')">Stop</button>
      </form>
    </div>
  </div>
  {# Log panel (same as running view) #}
  <div class="panel{% if current_job.current_step == 'confirm_total_rallies' %} step31-log{% endif %}" style="padding:0;overflow:hidden;">
    <div style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid var(--line);background:#111;">
      <span style="color:#8fc;font-size:12px;font-family:monospace;font-weight:600;">pipeline.log — {{ current_job.job_id }}</span>
    </div>
    <pre id="pipeline_log" style="margin:0;background:#111;color:#d4d0c8;font-size:12.5px;line-height:1.6;padding:16px;height:300px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;font-family:'Cascadia Code','Consolas','Fira Mono',monospace;">(loading log...)</pre>
  </div>
  <script>
    (function () {
      var jobId = "{{ current_job.job_id }}";
      var logEl = document.getElementById("pipeline_log");
      function fetchLog() {
        fetch("/jobs/" + encodeURIComponent(jobId) + "/log")
          .then(function (r) { return r.ok ? r.text() : null; })
          .then(function (t) {
            if (t === null || !logEl) return;
            logEl.textContent = t || "(no output yet)";
            logEl.scrollTop = logEl.scrollHeight;
          }).catch(function () {});
      }
      fetchLog();
    })();
  </script>

  {% else %}
  {# ── SETUP / IDLE VIEW ── #}
  <div class="panel">
    <h2>Setup</h2>
    {% if current_job %}
      <div class="progress-card" data-job-id="{{ current_job.job_id }}" data-job-status="{{ current_job.status }}">
        <div class="row" style="justify-content:space-between;">
          <strong id="prog_label">{{ progress.label }}</strong>
          <span id="progress_pct_label">{{ progress.percent }}%</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" style="width: {{ progress.percent }}%;" data-server-pct="{{ progress.percent }}" data-job-id="{{ current_job.job_id }}"></div></div>
        <div class="meta" id="prog_step">{{ progress.step_label }}</div>
        <div class="progress-meta" style="margin-top:12px;">
          <div class="progress-stat"><strong>Elapsed</strong><div id="prog_elapsed">{{ progress.elapsed_label }}</div></div>
          <div class="progress-stat"><strong>Rallies</strong><div id="prog_rallies">{{ progress.rallies_label }}</div></div>
          <div class="progress-stat"><strong>Resolved</strong><div id="prog_resolved">{{ progress.resolved_label }}</div></div>
          <div class="progress-stat"><strong>Pending</strong><div id="prog_pending">{{ progress.pending_label }}</div></div>
        </div>
      </div>
      {% if current_job.status == 'failed' %}
      <div style="margin-top:12px;">
        <div class="label" style="margin-bottom:6px;">Pipeline Log</div>
        <pre id="pipeline_log" style="background:#1a1a1a;color:#d4d0c8;font-size:12px;padding:12px;border-radius:10px;max-height:260px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;font-family:monospace;">(loading...)</pre>
      </div>
      <script>
        (function () {
          var jobId = "{{ current_job.job_id }}";
          fetch("/jobs/" + encodeURIComponent(jobId) + "/log")
            .then(function (r) { return r.ok ? r.text() : null; })
            .then(function (t) {
              var el = document.getElementById("pipeline_log");
              if (el && t) { el.textContent = t; el.scrollTop = el.scrollHeight; }
            }).catch(function () {});
        })();
      </script>
      {% endif %}
    {% endif %}
    {% if stage_message %}
      <div class="hint-box" style="margin-bottom:14px;">{{ stage_message }}</div>
    {% endif %}
    {% if current_job and current_job.error_message %}
      <div class="message error" style="margin-bottom:14px;">{{ current_job.error_message }}</div>
    {% endif %}
    <form method="post" action="/jobs">
      <input type="hidden" name="scan_id" id="input_scan_id" value="">
      <input type="hidden" name="debug_step3_only" id="debug_step3_only" value="0">
      <input type="hidden" name="debug_step3_phase" id="debug_step3_phase" value="">

      <!-- ── Phase 1: video browse + identify ────────────────────────── -->
      <div id="ph_browse">
        <div class="input-with-action" style="margin-bottom:6px;">
          <input id="raw_video_path" name="raw_video_path" placeholder="C:/videos/match.mp4" value="{{ raw_video_path_value }}" required>
          <button class="secondary" type="button" onclick="openRawVideoBrowser()">Browse</button>
        </div>
        <div class="meta" style="margin-bottom:14px;">Browse root: {{ raw_matches_root }}</div>
        <div style="margin-bottom:14px;">
          <label for="trim_start">Trim Start</label>
          <input id="trim_start" name="trim_start" value="{{ trim_start_value }}" placeholder="mm:ss or seconds">
          <div class="meta" style="margin-top:6px;">Identify Players will first trim the video from this timestamp, then scan the trimmed working video.</div>
        </div>
        <button class="secondary" type="button" id="btn_identify" onclick="startIdentifyPlayers()"
                style="width:100%;padding:10px 0;font-size:1em;">
          Identify Players
        </button>
        <div style="margin-top:12px;background:#151515;border:1px dashed #555;border-radius:8px;padding:12px;">
          <div style="font-weight:700;color:#ddd;margin-bottom:6px;">Debug Step 3 only</div>
          <div class="meta" style="margin-bottom:10px;">
            Skip Step 1 trim and Step 2 identify. You will manually enter Player A/B names already present in FaceDB, then GUI runs Step 3.1 + Step 3.2 and pauses at summary review.
          </div>
          <button class="secondary" type="button" id="btn_debug_step3" onclick="startStep3DebugSetup()"
                  style="width:100%;padding:9px 0;font-size:0.95em;">
            Skip Step 1/2 - Debug Step 3
          </button>
        </div>
      </div>

      <!-- ── Phase scanning: live log terminal ───────────────────────── -->
      <div id="ph_log" style="display:none;">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
          <strong id="log_status_label" style="font-size:0.95em;">Identifying players...</strong>
          <span style="display:flex;align-items:center;gap:10px;">
            <span id="log_elapsed" style="font-family:monospace;color:#888;font-size:0.9em;min-width:3em;text-align:right;"></span>
            <button class="secondary" type="button" id="btn_skip" style="display:none;padding:3px 10px;font-size:0.82em;"
                    onclick="transitionToSetup()">Skip →</button>
          </span>
        </div>
        <div class="progress-bar" style="margin-bottom:10px;">
          <div id="id_progress_fill" class="progress-fill running" style="width:0%;transition:width 0.4s ease;"></div>
        </div>
        <pre id="log_terminal"
             style="background:#111;color:#c8c8c8;font-size:11px;padding:10px;border-radius:6px;
                    min-height:150px;max-height:280px;overflow-y:auto;white-space:pre-wrap;
                    word-break:break-all;font-family:'Consolas',monospace;margin:0;"></pre>
        <div id="enroll_area" style="margin-top:10px;"></div>
      </div>

      <!-- ── Phase 2: full setup form ────────────────────────────────── -->
      <div id="ph_setup" style="display:none;">
        <div style="margin-bottom:12px;display:flex;align-items:center;gap:10px;">
          <span class="meta" id="video_path_summary" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"></span>
          <button class="secondary" type="button" style="white-space:nowrap;padding:3px 10px;font-size:0.82em;"
                  onclick="backToBrowse()">← Change</button>
        </div>
        <div class="grid two">
          <div>
            <label for="player_a_name">Player 1 <span style="color:#888;font-size:0.85em;">(near side, set 1)</span></label>
            <input id="player_a_name" name="player_a_name" value="{{ player_a_value }}" placeholder="Player name...">
          </div>
          <div>
            <label for="player_b_name">Player 2 <span style="color:#888;font-size:0.85em;">(far side, set 1)</span></label>
            <input id="player_b_name" name="player_b_name" value="{{ player_b_value }}" placeholder="Player name...">
          </div>
        </div>
        <div style="margin-top:12px;">
          <label for="best_of">Format</label>
          <select id="best_of" name="best_of">
            <option value="3" {% if best_of_value == 3 %}selected{% endif %}>Best of 3</option>
            <option value="5" {% if best_of_value == 5 %}selected{% endif %}>Best of 5</option>
            <option value="7" {% if best_of_value == 7 %}selected{% endif %}>Best of 7</option>
          </select>
        </div>
        <div style="margin-top:12px;">
          <label for="tournament_name">Tournament</label>
          <input id="tournament_name" name="tournament_name" value="{{ current_job.tournament_name if current_job else '' }}" placeholder="e.g. WTT Champions Frankfurt 2025">
        </div>
        <div style="margin-top:8px;">
          <label for="round_name">Round</label>
          <input id="round_name" name="round_name" value="{{ current_job.round_name if current_job else '' }}" placeholder="e.g. Semifinal, Quarterfinal, Final...">
        </div>
        <div style="margin-top:12px;">
          <label for="job_purpose">Job Purpose</label>
          <select id="job_purpose" name="job_purpose">
            <option value="output_only" {% if job_purpose_value == 'output_only' %}selected{% endif %}>Output Only — finish scoreboard video, no dataset write</option>
            <option value="output_and_dataset" {% if job_purpose_value == 'output_and_dataset' %}selected{% endif %}>Output + Dataset — finish video and grow reviewed dataset</option>
          </select>
        </div>
        <div id="normal_run_actions" class="point-actions" style="margin-top:16px;">
          <button id="run_pipeline_button" type="submit" onclick="setNormalPipelineMode()">Run AI Pipeline (Start Step 3)</button>
        </div>
        <div id="debug_run_actions" class="point-actions" style="margin-top:16px;display:none;gap:10px;flex-wrap:wrap;">
          <button type="submit" onclick="setStep3DebugPhase('3_1')">Run Step 3.1 Only</button>
          <button class="secondary" type="submit" onclick="setStep3DebugPhase('3_1_2')">Run Step 3.1 + 3.2</button>
        </div>
        <div id="run_pipeline_hint" class="meta" style="margin-top:8px;">
          Normal mode: Step 3.1 runs first, then waits for review before Step 3.2.
        </div>
      </div>

    </form>
    <script>
    var _scanId = null;
    var _logLineCount = 0;
    var _elapsedSec = 0;
    var _elapsedTimer = null;

    var _SCAN_ESTIMATE_SEC = 240; // ~4 min estimate for progress bar

    function _startElapsed() {
      _elapsedSec = 0;
      document.getElementById("log_elapsed").textContent = "0:00";
      var fill = document.getElementById("id_progress_fill");
      if (fill) { fill.style.width = "0%"; fill.className = "progress-fill running"; }
      _elapsedTimer = setInterval(function() {
        _elapsedSec++;
        var m = Math.floor(_elapsedSec / 60), s = _elapsedSec % 60;
        document.getElementById("log_elapsed").textContent = m + ":" + (s < 10 ? "0" : "") + s;
        // Animate bar: approach 95% asymptotically, never reach 100% until done
        var pct = Math.min(95, Math.round(_elapsedSec / _SCAN_ESTIMATE_SEC * 100));
        if (fill) fill.style.width = pct + "%";
      }, 1000);
    }

    function _stopElapsed() {
      if (_elapsedTimer) { clearInterval(_elapsedTimer); _elapsedTimer = null; }
      var fill = document.getElementById("id_progress_fill");
      if (fill) { fill.style.width = "100%"; fill.className = "progress-fill"; }
    }

    function _setPhase(phase) {
      document.getElementById("ph_browse").style.display = phase === "browse" ? "" : "none";
      document.getElementById("ph_log").style.display   = phase === "log"    ? "" : "none";
      document.getElementById("ph_setup").style.display = phase === "setup"  ? "" : "none";
    }

    function startIdentifyPlayers() {
      var videoPath = document.getElementById("raw_video_path").value.trim();
      var trimStart = document.getElementById("trim_start").value.trim();
      if (!videoPath) { alert("Select a video first."); return; }
      document.getElementById("debug_step3_only").value = "0";
      document.getElementById("debug_step3_phase").value = "";
      _scanId = null;
      _logLineCount = 0;
      document.getElementById("input_scan_id").value = "";
      document.getElementById("log_terminal").textContent = "";
      document.getElementById("enroll_area").innerHTML = "";
      document.getElementById("log_status_label").textContent = "Identifying players...";
      document.getElementById("btn_skip").style.display = "none";
      _setPhase("log");
      _startElapsed();
      fetch("/api/identify-players", {
        method: "POST",
        headers: {"Content-Type": "application/x-www-form-urlencoded"},
        body: "video_path=" + encodeURIComponent(videoPath) + "&trim_start=" + encodeURIComponent(trimStart)
      })
      .then(function(r) { return r.json(); })
      .then(function(d) {
        if (d.error) { _appendLog("ERROR: " + d.error); document.getElementById("btn_skip").style.display = ""; return; }
        _scanId = d.scan_id;
        _pollScan();
      })
      .catch(function(e) { _appendLog("Connection error: " + e); document.getElementById("btn_skip").style.display = ""; });
    }

    function setNormalPipelineMode() {
      document.getElementById("debug_step3_only").value = "0";
      document.getElementById("debug_step3_phase").value = "";
    }

    function setStep3DebugPhase(phase) {
      document.getElementById("debug_step3_only").value = "1";
      document.getElementById("debug_step3_phase").value = phase;
    }

    function startStep3DebugSetup() {
      var videoPath = document.getElementById("raw_video_path").value.trim();
      if (!videoPath) { alert("Select a video first."); return; }
      _stopElapsed();
      _scanId = null;
      _logLineCount = 0;
      document.getElementById("input_scan_id").value = "";
      document.getElementById("debug_step3_only").value = "1";
      document.getElementById("debug_step3_phase").value = "3_1_2";
      document.getElementById("video_path_summary").textContent = videoPath + "  |  DEBUG: skip Step 1/2";
      var normalActions = document.getElementById("normal_run_actions");
      var debugActions = document.getElementById("debug_run_actions");
      if (normalActions) normalActions.style.display = "none";
      if (debugActions) debugActions.style.display = "flex";
      var hint = document.getElementById("run_pipeline_hint");
      if (hint) hint.textContent = "Debug mode: choose Run Step 3.1 Only to pause at start-time review, or Run Step 3.1 + 3.2 to continue into side-state detection. Player names must already exist in FaceDB for Step 3.2 side-id.";
      _setPhase("setup");
    }

    function _appendLog(line) {
      var el = document.getElementById("log_terminal");
      el.textContent += line + "\\n";
      el.scrollTop = el.scrollHeight;
    }

    function _pollScan() {
      if (!_scanId) return;
      fetch("/api/identify-players/" + _scanId)
      .then(function(r) { return r.json(); })
      .then(function(d) {
        var logs = d.logs || [];
        for (var i = _logLineCount; i < logs.length; i++) { _appendLog(logs[i]); }
        _logLineCount = logs.length;

        if (d.status === "scanning") { setTimeout(_pollScan, 800); return; }

        if (d.status === "failed") {
          _stopElapsed();
          document.getElementById("log_status_label").textContent = "Error — check log above.";
          document.getElementById("btn_skip").style.display = "";
          return;
        }

        // Done
        _stopElapsed();
        // Persist scan_id so POST /jobs can retrieve table_roi from scan store
        document.getElementById("input_scan_id").value = _scanId || "";
        // Fill any identified names immediately
        if (d.near_name) document.getElementById("player_a_name").value = d.near_name;
        if (d.far_name)  document.getElementById("player_b_name").value  = d.far_name;

        var idStatus = d.id_status || "failed";
        var statusMsg = idStatus === "identified" ? "Both players identified."
          : idStatus === "partial" ? "One player identified — enrollment needed."
          : "Could not identify players — manual entry needed.";
        document.getElementById("log_status_label").textContent = statusMsg;

        // Build enrollment list: DB-unknown (face detected) + no-face unidentified players
        var unknowns = d.unknowns || [];
        var toEnroll = unknowns.slice();
        var unknownRoles = unknowns.map(function(u) { return u.role; });
        if (!d.near_name && unknownRoles.indexOf("near") === -1) {
          toEnroll.push({ role: "near", crop_b64: null, no_face: true });
        }
        if (!d.far_name && unknownRoles.indexOf("far") === -1) {
          toEnroll.push({ role: "far", crop_b64: null, no_face: true });
        }

        if (toEnroll.length > 0) {
          _showEnrollForms(toEnroll);
          document.getElementById("btn_skip").style.display = "";
        } else {
          transitionToSetup();
        }
      })
      .catch(function(e) { _appendLog("Poll error: " + e); setTimeout(_pollScan, 2000); });
    }

    function _showEnrollForms(items) {
      var header = "<div style='margin-bottom:8px;color:#f39c12;font-size:0.9em;'>Player info needed — fill to continue or click Skip:</div>";
      var cards = "";
      items.forEach(function(u) {
        var lbl = u.role === "near" ? "near side (Player A)" : "far side (Player B)";
        cards += "<div id='enroll_card_" + u.role + "' style='padding:8px;border:1px solid #555;border-radius:4px;margin-bottom:8px;display:flex;gap:10px;align-items:flex-start;'>";
        if (u.crop_b64) {
          cards += "<img src='data:image/png;base64," + u.crop_b64 + "' style='width:60px;height:60px;object-fit:cover;border-radius:4px;flex-shrink:0;'>";
        } else {
          cards += "<div style='width:60px;height:60px;background:#1e1e1e;border:1px dashed #555;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:26px;color:#666;flex-shrink:0;'>?</div>";
        }
        cards += "<div style='flex:1;min-width:0;'>";
        cards += "<div style='color:#f39c12;margin-bottom:4px;font-size:0.85em;'>Unknown — " + lbl + "</div>";
        if (u.no_face) {
          cards += "<div style='color:#888;font-size:0.78em;margin-bottom:6px;'>No face detected — name will be set for this match only (not added to face database)</div>";
        } else {
          cards += "<div style='color:#888;font-size:0.78em;margin-bottom:6px;'>Face detected but not in database — enter name to enroll</div>";
        }
        cards += "<div class='input-with-action' style='gap:6px;'>";
        cards += "<input id='enroll_inp_" + u.role + "' placeholder='Player name...' style='flex:1;font-size:0.88em;'>";
        if (u.no_face) {
          cards += "<button class='secondary' type='button' style='padding:4px 10px;font-size:0.82em;white-space:nowrap;' data-role='" + u.role + "' onclick='setNameManually(this.dataset.role)'>Set name</button>";
        } else {
          cards += "<button class='secondary' type='button' style='padding:4px 10px;font-size:0.82em;white-space:nowrap;' data-role='" + u.role + "' onclick='enrollPlayer(this.dataset.role)'>Enroll</button>";
        }
        cards += "</div></div></div>";
      });
      document.getElementById("enroll_area").innerHTML = header + cards;
    }

    function _checkAllResolved() {
      var area = document.getElementById("enroll_area");
      if (!area) return;
      var remaining = area.querySelectorAll("[id^='enroll_card_']");
      if (remaining.length === 0) { transitionToSetup(); }
    }

    function setNameManually(role) {
      var inp = document.getElementById("enroll_inp_" + role);
      var name = inp ? inp.value.trim() : "";
      if (!name) { alert("Enter player name first."); return; }
      if (role === "near") document.getElementById("player_a_name").value = name;
      else                 document.getElementById("player_b_name").value = name;
      _appendLog("Set name: " + name + " (" + role + " side) — not enrolled in face DB");
      var card = document.getElementById("enroll_card_" + role);
      if (card) card.remove();
      _checkAllResolved();
    }

    function enrollPlayer(role) {
      var inp = document.getElementById("enroll_inp_" + role);
      var name = inp ? inp.value.trim() : "";
      if (!name) { alert("Enter player name first."); return; }
      fetch("/api/enroll-player", {
        method: "POST",
        headers: {"Content-Type": "application/x-www-form-urlencoded"},
        body: "scan_id=" + encodeURIComponent(_scanId) + "&role=" + encodeURIComponent(role) + "&name=" + encodeURIComponent(name)
      })
      .then(function(r) { return r.json(); })
      .then(function(d) {
        if (d.error) { alert("Error: " + d.error); return; }
        _appendLog("Enrolled: " + name + " (" + role + " side)");
        if (role === "near") document.getElementById("player_a_name").value = name;
        else                 document.getElementById("player_b_name").value = name;
        var card = document.getElementById("enroll_card_" + role);
        if (card) card.remove();
        _checkAllResolved();
      })
      .catch(function(e) { alert("Error: " + e); });
    }

    function transitionToSetup() {
      var vp = document.getElementById("raw_video_path").value.trim();
      document.getElementById("video_path_summary").textContent = vp;
      document.getElementById("debug_step3_only").value = "0";
      document.getElementById("debug_step3_phase").value = "";
      var normalActions = document.getElementById("normal_run_actions");
      var debugActions = document.getElementById("debug_run_actions");
      if (normalActions) normalActions.style.display = "";
      if (debugActions) debugActions.style.display = "none";
      var hint = document.getElementById("run_pipeline_hint");
      if (hint) hint.textContent = "Normal mode: Step 3.1 runs first, then waits for review before Step 3.2.";
      _setPhase("setup");
    }

    function backToBrowse() {
      _stopElapsed();
      _scanId = null; _logLineCount = 0;
      document.getElementById("input_scan_id").value = "";
      document.getElementById("debug_step3_only").value = "0";
      document.getElementById("debug_step3_phase").value = "";
      document.getElementById("log_terminal").textContent = "";
      document.getElementById("enroll_area").innerHTML = "";
      _setPhase("browse");
    }
    </script>
  </div>
  <div class="panel">
    <h2>Trim Start</h2>
    {% if main_video_src %}
      <div class="meta">Preview raw video here, then enter Trim Start before clicking Identify Players.</div>
      <video class="trim-player" controls preload="metadata" src="{{ main_video_src }}"></video>
    {% else %}
      <div class="hint-box">Choose the raw match video first. A small preview player will appear here for trim setup.</div>
    {% endif %}
  </div>
  {% endif %}
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
          <label for="player_a_name">Player NEAR (Set 1)</label>
          <input id="player_a_name" name="player_a_name" value="{{ player_a_value }}" required>
        </div>
        <div>
          <label for="player_b_name">Player FAR (Set 1)</label>
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
      <div class="meta">{{ current_job.player_a_name }} vs {{ current_job.player_b_name }}</div>
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
          <a class="{% if active_filter == 'pending' %}active{% endif %}" href="/?job_id={{ current_job.job_id }}&review_filter=pending">Needs Input {% if review_status.unresolved_scoring_points > 0 %}<span class="badge" style="background:#c0392b">{{ review_status.unresolved_scoring_points }}</span>{% endif %}</a>
          <a class="{% if active_filter == 'all' %}active{% endif %}" href="/?job_id={{ current_job.job_id }}&review_filter=all">All Rallies ({{ review_status.scoring_points }})</a>
        </div>
        {% if active_filter == 'pending' and review_status.unresolved_scoring_points == 0 %}
          <div class="hint-box" style="margin-top:12px;background:#1a2a1a;border-color:#4caf50;color:#4caf50;">
            AI has predicted winners for all rallies. Review any rally in the "All Rallies" tab if you think the AI is wrong.
          </div>
        {% endif %}
        {% if points %}
          <table style="margin-top:12px;">
            <thead>
              <tr>
                <th>Rally</th>
                <th>Winner</th>
                <th>Source</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
            {% for point in points %}
              <tr id="{{ point.id }}" class="timeline-item {{ point.status_class }}">
                <td>
                  <div><strong>{{ point.id }}</strong></div>
                  <div class="meta">{{ point.time_range }}</div>
                </td>
                <td>
                  {% if point.needs_input %}
                    <span style="color:#e74c3c">? Unknown</span>
                  {% elif point.manually_corrected %}
                    <span>{{ point.effective_winner_label }}</span>
                    <div class="meta" style="color:#f39c12">manually corrected</div>
                  {% else %}
                    <span>{{ point.effective_winner_label }}</span>
                    <div class="meta" style="color:#888">AI</div>
                  {% endif %}
                </td>
                <td><span class="meta">{{ point.decision }}</span></td>
                <td>
                  <div class="point-actions">
                    <button class="secondary" type="button" onclick='return playMainVideo("{{ point.clip_src }}", "{{ point.play_label }}");'>Play</button>
                    <form method="post" action="/jobs/{{ current_job.job_id }}/review/{{ point.id }}">
                      <input type="hidden" name="filter" value="{{ active_filter }}">
                      <input type="hidden" name="action" value="set_winner">
                      <input type="hidden" name="winner" value="player_a">
                      <button class="secondary" type="submit">{{ current_job.player_a_name }}</button>
                    </form>
                    <form method="post" action="/jobs/{{ current_job.job_id }}/review/{{ point.id }}">
                      <input type="hidden" name="filter" value="{{ active_filter }}">
                      <input type="hidden" name="action" value="set_winner">
                      <input type="hidden" name="winner" value="player_b">
                      <button class="secondary" type="submit">{{ current_job.player_b_name }}</button>
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
          <button class="warn" type="submit">Export</button>
        </form>
      </div>
    {% endif %}
  </div>
</div>
{% endif %}
{% if false %}
  <div class="panel">
    <h2>Current Flow</h2>
    <div class="meta">
      raw video → trim → rally timeline → winner adapter → review corrections → Export (render + final video)
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
      <div class="stat"><strong>Player A</strong><div>{{ job.player_a_name }}</div></div>
      <div class="stat"><strong>Player B</strong><div>{{ job.player_b_name }}</div></div>
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
    Confirm or correct each rally winner below. Export unlocks when all scoring rallies have a confirmed winner.
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
    <div class="stat"><strong>AI Predicted</strong><div>{{ review_status.preview_known_points }}/{{ review_status.scoring_points }}</div></div>
    <div class="stat"><strong>Needs Input</strong><div>{{ review_status.blocked_points }}</div></div>
    <div class="stat"><strong>Manually Corrected</strong><div>{{ review_status.resolved_scoring_points }}</div></div>
  </div>
  <div class="hint-box" style="margin-top:14px;">
    AI winners are used directly. Only correct rallies where the AI got it wrong, or input winners for rallies AI couldn't predict.
  </div>
  <div class="point-actions">
    <a class="subtle-link" href="/jobs/{{ job.job_id }}">Back To Job</a>
    <form method="post" action="/jobs/{{ job.job_id }}/preview"><button class="secondary" type="submit">Refresh Preview</button></form>
    <form method="post" action="/jobs/{{ job.job_id }}/final-export"><button class="warn" type="submit">Final Export</button></form>
  </div>
  <div class="tabs" style="margin-top:14px;">
    <a class="{% if active_filter == 'pending' %}active{% endif %}" href="/jobs/{{ job.job_id }}/review?filter=pending">Needs Input ({{ review_status.blocked_points }})</a>
    <a class="{% if active_filter == 'all' %}active{% endif %}" href="/jobs/{{ job.job_id }}/review?filter=all">All Rallies ({{ review_status.scoring_points }})</a>
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
          {% if point.needs_input %}
            <span class="badge failed">needs input</span>
          {% elif point.manually_corrected %}
            <span class="badge completed">corrected</span>
          {% else %}
            <span class="badge running">AI</span>
          {% endif %}
        </div>
      </div>
      <div class="stats" style="margin-top:10px;">
        <div class="stat"><strong>Winner</strong><div>{{ point.effective_winner_label }}</div></div>
        <div class="stat"><strong>AI Prediction</strong><div>{{ point.ai_winner_label }}</div></div>
        <div class="stat"><strong>Category</strong><div>{{ point.category }}</div></div>
        <div class="stat"><strong>Source</strong><div>{{ point.source }}</div></div>
      </div>
      {% if point.manually_corrected and point.last_note %}
        <div class="meta" style="margin-top:8px;color:#f39c12;">Note: {{ point.last_note }}</div>
      {% endif %}
      <div class="point-actions">
        <form method="post" action="/jobs/{{ job.job_id }}/review/{{ point.id }}">
          <input type="hidden" name="filter" value="{{ active_filter }}">
          <input type="hidden" name="action" value="set_winner">
          <input type="hidden" name="winner" value="player_a">
          <button class="secondary" type="submit">{{ job.player_a_name }} Wins</button>
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
