from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.ai_table_roi import TableROI
from backend.config import PROJECT_ROOT
from backend.production_defaults import PRODUCTION_RALLY_DEFAULTS
from backend.production_jobs import load_match_job
from backend.step3_rally_start_review import (
    Step3PlayerContext,
    Step3SideIdentificationConfig,
    annotate_serve_order_rule_reviews,
    annotate_events_with_single_player_side_identification,
    build_step3_1_rally_start_review,
    write_rally_start_events_json,
)


def _fmt_seconds_tag(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3f}".replace(".", "p")


def _fmt_mmss(value: float) -> str:
    value = max(0.0, float(value))
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes:02d}:{seconds:06.3f}"


def _run_ffmpeg_nvenc_clip(
    *,
    video_path: Path,
    clip_path: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    duration = max(0.0, float(end_sec) - float(start_sec))
    if duration <= 0:
        raise ValueError("--end must be greater than --start")
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-ss",
        f"{float(start_sec):.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p1",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(clip_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(message or f"ffmpeg failed with code {exc.returncode}") from exc


def _load_table_roi_from_job(job) -> TableROI | None:
    roi_data = job.timeline_summary.get("table_roi")
    if not isinstance(roi_data, dict):
        return None
    return TableROI(
        x=int(roi_data["x"]),
        y=int(roi_data["y"]),
        w=int(roi_data["w"]),
        h=int(roi_data["h"]),
        confidence=float(roi_data.get("confidence", 1.0)),
    )


def _write_summary_md(path: Path, summary: dict, events: list[dict]) -> None:
    first_server = summary.get("first_server") or {}
    gap_rescans = list(summary.get("gap_rescans", []) or [])
    side_id = summary.get("side_identification") or {}
    lines = [
        "# Step 3.1 Rally Start Debug",
        "",
        f"- Total starts: {summary.get('total', 0)}",
        f"- Detected starts: {summary.get('detected_total', summary.get('total', 0))}",
        f"- Scoring rallies: {summary.get('scoring', 0)}",
        f"- LET/non-scoring: {summary.get('lets', 0)}",
        f"- Needs-review rows: {summary.get('needs_review', 0)}",
        f"- Serve-order gap markers: {summary.get('rule_gap_review_count', 0)}",
        f"- Rule-conflict detected rows: {summary.get('rule_conflict_review_count', 0)}",
        f"- Side evidence: identified {side_id.get('identified', 0)} / inferred {side_id.get('inferred', 0)} / unknown {side_id.get('unknown', 0)} "
        f"({side_id.get('algorithm', side_id.get('reason', ''))})",
        f"- First server: {first_server.get('server_player_name', 'unknown')} "
        f"(side={first_server.get('current_side', '-')}, role={first_server.get('starter_role', '-')}, "
        f"start_time={_fmt_mmss(float(first_server.get('source_t_start', first_server.get('t_start', 0.0))))})",
        f"- Events JSON: {summary.get('events_json_path', '')}",
        f"- CSV: {summary.get('csv_path', '')}",
        f"- Start frames: {summary.get('start_frames_dir', '')}",
        f"- Source video: {summary.get('source_video_path', '')}",
        "",
    ]
    if gap_rescans:
        lines.extend(
            [
                "## Gap Rescan",
                "",
                "| gap | source_window | expected_server | candidates_in_gap | best_candidate | folder |",
                "|---|---:|---|---:|---|---|",
            ]
        )
        for item in gap_rescans:
            best = item.get("best_candidate") or {}
            if best:
                best_text = (
                    f"{float(best.get('source_timestamp', 0.0)):.3f}s "
                    f"role={best.get('role', '')} score={float(best.get('score', 0.0)):.3f}"
                )
            else:
                best_text = "none"
            lines.append(
                "| {gap_id} | {src_start} -> {src_end} | {server} ({role}) | {count} | {best} | {folder} |".format(
                    gap_id=item.get("gap_id", ""),
                    src_start=_fmt_mmss(float(item.get("source_gap_start", 0.0))),
                    src_end=_fmt_mmss(float(item.get("source_gap_end", 0.0))),
                    server=item.get("expected_server_name", "unknown"),
                    role=item.get("expected_role", ""),
                    count=int(item.get("candidate_count_in_gap", 0)),
                    best=best_text,
                    folder=item.get("rescan_dir", ""),
                )
            )
        lines.append("")
    lines.extend(
        [
        "| id | kind | start_time | server | current_side | side_evidence | note | image |",
        "|---|---|---:|---|---|---|---|---|",
        ]
    )
    for event in events:
        note = event.get("review_reason", "")
        expected_role = event.get("serve_order_expected_role", "")
        expected_server = event.get("serve_order_expected_server_name", "")
        if note and expected_role:
            note = f"{note}; expected {expected_server} ({expected_role})"
        side_evidence = event.get("side_evidence_status", "")
        identified = event.get("side_identified_player_name", "")
        identified_side = event.get("side_identified_current_side", "")
        if side_evidence == "identified" and identified:
            side_evidence = f"{identified}={identified_side}"
        elif side_evidence == "inferred":
            side_evidence = f"inferred({event.get('side_evidence_reason', '')})"
        lines.append(
            "| {id} | {kind} | {start_time} | {server} | {current_side} | {side_evidence} | {note} | {image} |".format(
                id=event.get("id", ""),
                kind=event.get("kind", ""),
                start_time=_fmt_mmss(float(event.get("source_t_start", event.get("t_start", 0.0)))),
                server=event.get("server_player_name", "unknown"),
                current_side=event.get("current_side", "unknown"),
                side_evidence=side_evidence,
                note=note,
                image=event.get("image_file", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_candidate_csv(csv_path: Path, *, source_time_offset_sec: float) -> list[dict]:
    if not csv_path.exists():
        return []
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            timestamp = float(row.get("timestamp") or 0.0)
            score = float(row.get("score") or 0.0)
            rows.append(
                {
                    "candidate_id": row.get("candidate_id", ""),
                    "image_file": row.get("image_file", ""),
                    "timestamp": timestamp,
                    "source_timestamp": timestamp + float(source_time_offset_sec),
                    "role": row.get("role", ""),
                    "score": score,
                    "prep_score": float(row.get("prep_score") or 0.0),
                    "launch_score": float(row.get("launch_score") or 0.0),
                    "receiver_peak_score": float(row.get("receiver_peak_score") or 0.0),
                    "live_peak_score": float(row.get("live_peak_score") or 0.0),
                }
            )
    return rows


def _rescan_review_gaps(
    *,
    clip_path: Path,
    events: list[dict],
    out_dir: Path,
    source_time_offset_sec: float,
    selection_mode: str,
    pad_sec: float,
) -> list[dict]:
    gap_events = [event for event in events if event.get("kind") == "needs_review"]
    if not gap_events:
        return []

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.append(str(scripts_dir))
    from export_player_rally_start_candidates import export_rally_start_candidates

    defaults = PRODUCTION_RALLY_DEFAULTS
    summaries: list[dict] = []
    for event in gap_events:
        gap_start = float(event.get("gap_start", event.get("t_start", 0.0)))
        gap_end = float(event.get("gap_end", event.get("t_end", gap_start)))
        scan_start = max(0.0, gap_start - float(pad_sec))
        scan_end = max(scan_start + 0.05, gap_end + float(pad_sec))
        gap_dir = out_dir / "gap_rescan" / f"{event.get('id', 'gap')}_{_fmt_seconds_tag(gap_start)}_{_fmt_seconds_tag(gap_end)}"
        print(
            "[step3.1] rescanning gap "
            f"{event.get('id', '')}: clip {scan_start:.3f}s -> {scan_end:.3f}s "
            f"(source {scan_start + source_time_offset_sec:.3f}s -> {scan_end + source_time_offset_sec:.3f}s)"
        )
        export_rally_start_candidates(
            str(clip_path),
            defaults.table_weights_path,
            pose_weights=defaults.pose_weights_path,
            out_dir_str=str(gap_dir),
            selection_mode=selection_mode,
            stride=defaults.stride,
            player_margin_px=defaults.player_margin_px,
            start_seconds=scan_start,
            max_seconds=scan_end - scan_start,
        )

        csv_path = gap_dir / "rally_start_candidates.csv"
        candidates = _read_candidate_csv(csv_path, source_time_offset_sec=source_time_offset_sec)
        in_gap = [
            candidate
            for candidate in candidates
            if gap_start <= float(candidate.get("timestamp", 0.0)) <= gap_end
        ]
        expected_role = str(event.get("serve_order_expected_role") or event.get("starter_role") or "")
        role_matches = [
            candidate
            for candidate in in_gap
            if str(candidate.get("role", "") or "") == expected_role
        ]
        ranked = role_matches or in_gap
        best = max(ranked, key=lambda row: float(row.get("score", 0.0))) if ranked else None
        summaries.append(
            {
                "gap_id": event.get("id", ""),
                "source_gap_start": float(event.get("source_gap_start", gap_start + source_time_offset_sec)),
                "source_gap_end": float(event.get("source_gap_end", gap_end + source_time_offset_sec)),
                "gap_start": gap_start,
                "gap_end": gap_end,
                "scan_start": scan_start,
                "scan_end": scan_end,
                "expected_role": expected_role,
                "expected_server_name": event.get("serve_order_expected_server_name")
                or event.get("server_player_name", "unknown"),
                "rescan_dir": str(gap_dir.resolve()).replace("\\", "/"),
                "csv_path": str(csv_path.resolve()).replace("\\", "/"),
                "selection_mode": selection_mode,
                "candidate_count_total": len(candidates),
                "candidate_count_in_gap": len(in_gap),
                "expected_role_candidate_count_in_gap": len(role_matches),
                "best_candidate": best,
            }
        )
    return summaries


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Export Step 3.1 rally/LET starts for a bounded debug window.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--start", type=float, default=0.0, help="Start timestamp in seconds.")
    parser.add_argument("--end", type=float, required=True, help="End timestamp in seconds.")
    parser.add_argument("--out-dir", default="", help="Output folder. Defaults to runtime_jobs/debug_<stem>_<start>_<end>/step3_1_review.")
    parser.add_argument("--job-json", default="", help="Existing MatchJob JSON to reuse Step 2 names and table ROI.")
    parser.add_argument("--player-a-name", default="", help="Trusted player_a name, initial near side if no job JSON is provided.")
    parser.add_argument("--player-b-name", default="", help="Trusted player_b name, initial far side if no job JSON is provided.")
    parser.add_argument("--player-a-starts-near", action="store_true", default=True)
    parser.add_argument("--force", action="store_true", help="Rebuild the timeline even if cached output exists.")
    parser.add_argument("--force-clip", action="store_true", help="Recreate the bounded clip even if it already exists.")
    parser.add_argument("--rescan-only", action="store_true", help="Load existing Step 3.1 output and only rescan rule-suspicious gap windows.")
    parser.add_argument("--rescan-review-gaps", action="store_true", help="Run raw candidate rescan around needs-review serve-order gaps.")
    parser.add_argument("--rescan-mode", choices=["raw", "sandwich"], default="raw", help="Candidate selection used for review-gap rescan.")
    parser.add_argument("--rescan-pad-sec", type=float, default=1.5, help="Seconds of padding on each side of a review gap rescan.")
    parser.add_argument("--with-side-id", action="store_true", help="Also run Step 3.2 side ID in this Step 3.1 debug command. Off by default.")
    parser.add_argument("--no-side-id", action="store_true", help="Compatibility flag; Step 3.1 is detector-only by default.")
    parser.add_argument("--face-db", default=str(PROJECT_ROOT / "data" / "players" / "faces.json"), help="FaceDB path.")
    parser.add_argument("--face-model", default=str(PROJECT_ROOT / "data" / "models" / "face" / "w600k_r50.onnx"), help="ArcFace ONNX model path.")
    parser.add_argument("--side-id-before-sec", type=float, default=1.00, help="Seconds before rally start for local side ID scan.")
    parser.add_argument("--side-id-after-sec", type=float, default=4.00, help="Seconds after rally start for the primary local side ID scan.")
    parser.add_argument("--side-id-break-gap-sec", type=float, default=12.00, help="Gap after a rally that blocks post-rally side ID extension.")
    parser.add_argument("--side-id-next-guard-sec", type=float, default=0.25, help="Seconds to stop before the next rally when extending side ID scan.")
    parser.add_argument("--side-id-sample-fps", type=float, default=4.0, help="Sampling FPS for local side ID scan.")
    parser.add_argument("--side-id-threshold", type=float, default=0.35, help="Face similarity threshold for trusted Step 2 players.")
    parser.add_argument("--side-id-margin", type=float, default=0.04, help="Minimum best-vs-second similarity margin.")
    parser.add_argument("--side-id-min-best-sim", type=float, default=0.45, help="Minimum best face similarity before accepting current side.")
    parser.add_argument("--side-id-min-avg-sim", type=float, default=0.38, help="Minimum average accepted face similarity before accepting current side.")
    parser.add_argument("--side-id-min-samples", type=int, default=4, help="Minimum accepted face samples before accepting current side.")
    parser.add_argument("--side-id-no-retry-unknown", action="store_true", help="Disable Step 3.2 retry scan for unknown rows.")
    parser.add_argument("--side-id-retry-after-sec", type=float, default=12.00, help="Longer start-anchored retry window for unknown rows.")
    parser.add_argument("--side-id-retry-fps", type=float, default=8.0, help="Sampling FPS for unknown retry scan.")
    parser.add_argument("--side-id-retry-min-samples", type=int, default=3, help="Minimum accepted face samples for unknown retry scan.")
    parser.add_argument("--side-id-no-promote-strong-candidate", action="store_true", help="Disable strong-candidate promotion for rows that only miss sample count.")
    parser.add_argument("--side-id-no-continuity-fill", action="store_true", help="Disable side continuity fill for rows still unknown after scan.")
    parser.add_argument("--side-id-continuity-terminal-gap-sec", type=float, default=12.00, help="Max start gap for terminal continuity fill.")
    parser.add_argument("--enable-jersey-side-id", action="store_true", help="Allow jersey-only side ID fallback. Off by default because similar shirts can be confidently wrong.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
    video_path = video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    start_sec = float(args.start)
    end_sec = float(args.end)
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        out_dir = (
            PROJECT_ROOT
            / "runtime_jobs"
            / f"debug_{video_path.stem}_{_fmt_seconds_tag(start_sec)}_{_fmt_seconds_tag(end_sec)}"
            / "step3_1_review"
        )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_path = out_dir / f"clip_{_fmt_seconds_tag(start_sec)}_{_fmt_seconds_tag(end_sec)}.mp4"
    side_id_config = Step3SideIdentificationConfig(
        window_before_sec=float(args.side_id_before_sec),
        window_after_sec=float(args.side_id_after_sec),
        break_gap_sec=float(args.side_id_break_gap_sec),
        next_event_guard_sec=float(args.side_id_next_guard_sec),
        sample_fps=float(args.side_id_sample_fps),
        match_threshold=float(args.side_id_threshold),
        match_margin=float(args.side_id_margin),
        min_best_similarity=float(args.side_id_min_best_sim),
        min_avg_similarity=float(args.side_id_min_avg_sim),
        min_accepted_samples=int(args.side_id_min_samples),
        enable_jersey_fallback=bool(args.enable_jersey_side_id),
        retry_unknown_enabled=not bool(args.side_id_no_retry_unknown),
        retry_window_after_sec=float(args.side_id_retry_after_sec),
        retry_sample_fps=float(args.side_id_retry_fps),
        retry_min_accepted_samples=int(args.side_id_retry_min_samples),
        promote_strong_candidate_enabled=not bool(args.side_id_no_promote_strong_candidate),
        continuity_fill_unknown_enabled=not bool(args.side_id_no_continuity_fill),
        continuity_terminal_max_gap_sec=float(args.side_id_continuity_terminal_gap_sec),
    )
    enable_side_id = bool(args.with_side_id) and not bool(args.no_side_id)
    if args.rescan_only:
        events_json_path = out_dir / "step3_1_rally_start_events.json"
        if not events_json_path.exists():
            raise FileNotFoundError(f"Cannot --rescan-only without existing events JSON: {events_json_path}")
        if not clip_path.exists():
            raise FileNotFoundError(f"Cannot --rescan-only without existing bounded clip: {clip_path}")
        payload = json.loads(events_json_path.read_text(encoding="utf-8"))
        summary = dict(payload.get("summary") or {})
        events = list(payload.get("events") or [])
        job = load_match_job(args.job_json) if args.job_json else None
        if job is not None:
            player_context = Step3PlayerContext(
                player_a_name=job.player_a_name,
                player_b_name=job.player_b_name,
                player_a_starts_near=job.player_a_starts_near,
            )
        elif args.player_a_name or args.player_b_name:
            player_context = Step3PlayerContext(
                player_a_name=args.player_a_name or "unknown",
                player_b_name=args.player_b_name or "unknown",
                player_a_starts_near=bool(args.player_a_starts_near),
            )
        else:
            player_context = None
        table_roi = _load_table_roi_from_job(job) if job is not None else None
        if table_roi is None and enable_side_id:
            from backend.player_identification import detect_table_roi_and_player_zone

            table_roi, _zone = detect_table_roi_and_player_zone(video_path, PRODUCTION_RALLY_DEFAULTS.table_weights_path)
        if enable_side_id:
            side_id_summary = annotate_events_with_single_player_side_identification(
                video_path,
                events,
                player_context=player_context,
                face_db_path=Path(args.face_db),
                face_model_path=Path(args.face_model),
                pose_weights_path=PRODUCTION_RALLY_DEFAULTS.pose_weights_path,
                table_roi=table_roi,
                config=side_id_config,
                time_field="source_t_start",
                end_time_field="source_t_end",
                log_fn=lambda msg: print(f"[step3.1] {msg}", flush=True),
            )
            summary.setdefault("side_identification", {}).update(side_id_summary)
            annotate_serve_order_rule_reviews(events, player_context=player_context)
        summary["source_video_path"] = str(video_path).replace("\\", "/")
        summary["clip_path"] = str(clip_path).replace("\\", "/")
        summary["clip_start_sec"] = start_sec
        summary["clip_end_sec"] = end_sec
        if args.rescan_review_gaps:
            summary["gap_rescans"] = _rescan_review_gaps(
                clip_path=clip_path,
                events=events,
                out_dir=out_dir,
                source_time_offset_sec=start_sec,
                selection_mode=args.rescan_mode,
                pad_sec=float(args.rescan_pad_sec),
            )
        write_rally_start_events_json(events_json_path, summary, events)
        summary_path = out_dir / "summary.md"
        _write_summary_md(summary_path, summary, events)
        print(f"[step3.1] rescan-only complete: {len(summary.get('gap_rescans', []))} targeted gap window(s)")
        print(f"[step3.1] summary: {summary_path}")
        return 0

    job = load_match_job(args.job_json) if args.job_json else None
    table_roi = _load_table_roi_from_job(job) if job is not None else None
    player_zone_xyxy = None
    if job is not None:
        zone_data = job.timeline_summary.get("player_zone")
        if isinstance(zone_data, dict):
            player_zone_xyxy = (
                float(zone_data["x1"]),
                float(zone_data["y1"]),
                float(zone_data["x2"]),
                float(zone_data["y2"]),
            )
    if table_roi is None:
        from backend.player_identification import detect_table_roi_and_player_zone

        table_roi, player_zone_xyxy = detect_table_roi_and_player_zone(video_path, PRODUCTION_RALLY_DEFAULTS.table_weights_path)
    if table_roi is None or table_roi.w <= 0:
        raise RuntimeError("Table ROI detection failed")

    player_context = None
    if job is not None:
        player_context = Step3PlayerContext(
            player_a_name=job.player_a_name,
            player_b_name=job.player_b_name,
            player_a_starts_near=job.player_a_starts_near,
        )
    elif args.player_a_name or args.player_b_name:
        player_context = Step3PlayerContext(
            player_a_name=args.player_a_name or "unknown",
            player_b_name=args.player_b_name or "unknown",
            player_a_starts_near=bool(args.player_a_starts_near),
        )

    if args.force_clip or not clip_path.exists():
        print(f"[step3.1] creating bounded clip: {clip_path}")
        _run_ffmpeg_nvenc_clip(
            video_path=video_path,
            clip_path=clip_path,
            start_sec=start_sec,
            end_sec=end_sec,
        )
    else:
        print(f"[step3.1] reusing bounded clip: {clip_path}")

    defaults = PRODUCTION_RALLY_DEFAULTS
    result = build_step3_1_rally_start_review(
        video_path=clip_path,
        timeline_path=out_dir / "step3_1_total_rally_timeline.json",
        events_json_path=out_dir / "step3_1_rally_start_events.json",
        frame_dir=out_dir / "start_frames",
        table_roi=table_roi,
        table_weights_path=defaults.table_weights_path,
        pose_weights_path=defaults.pose_weights_path,
        best_of=int(job.best_of if job is not None else 3),
        stride=defaults.stride,
        mode=defaults.mode,
        player_margin_px=defaults.player_margin_px,
        player_fuse_gain=defaults.player_fuse_gain,
        player_signal_source=defaults.player_signal_source,
        ball_fuse_gain=defaults.ball_fuse_gain,
        ball_signal_source=defaults.ball_signal_source,
        player_context=player_context,
        legacy_cache_path=None,
        force_rebuild=bool(args.force),
        source_time_offset_sec=start_sec,
        enable_side_identification=enable_side_id,
        side_identification_video_path=video_path,
        side_identification_time_field="source_t_start",
        side_identification_end_time_field="source_t_end",
        face_db_path=Path(args.face_db),
        face_model_path=Path(args.face_model),
        player_zone_xyxy=player_zone_xyxy,
        side_identification_config=side_id_config,
        log_fn=lambda msg: print(f"[step3.1] {msg}", flush=True),
    )
    summary_path = out_dir / "summary.md"
    result.summary["source_video_path"] = str(video_path).replace("\\", "/")
    result.summary["clip_path"] = str(clip_path).replace("\\", "/")
    result.summary["clip_start_sec"] = start_sec
    result.summary["clip_end_sec"] = end_sec
    if args.rescan_review_gaps:
        result.summary["gap_rescans"] = _rescan_review_gaps(
            clip_path=clip_path,
            events=result.events,
            out_dir=out_dir,
            source_time_offset_sec=start_sec,
            selection_mode=args.rescan_mode,
            pad_sec=float(args.rescan_pad_sec),
        )
    write_rally_start_events_json(out_dir / "step3_1_rally_start_events.json", result.summary, result.events)
    _write_summary_md(summary_path, result.summary, result.events)

    first_server = result.summary.get("first_server") or {}
    print(
        "[step3.1] detected "
        f"{result.summary.get('total', 0)} total "
        f"({result.summary.get('scoring', 0)} scoring + "
        f"{result.summary.get('lets', 0)} LET/non-scoring + "
        f"{result.summary.get('needs_review', 0)} needs-review)"
    )
    print(
        "[step3.1] first server: "
        f"{first_server.get('server_player_name', 'unknown')} "
        f"(role={first_server.get('starter_role', '-')}, "
        f"clip_t={float(first_server.get('t_start', 0.0)):.3f}s, "
        f"source_t={float(first_server.get('source_t_start', first_server.get('t_start', 0.0))):.3f}s)"
    )
    print(f"[step3.1] summary: {summary_path}")
    print(f"[step3.1] events:  {result.summary.get('events_json_path', '')}")
    print(f"[step3.1] csv:     {result.summary.get('csv_path', '')}")
    print(f"[step3.1] frames:  {result.summary.get('start_frames_dir', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
