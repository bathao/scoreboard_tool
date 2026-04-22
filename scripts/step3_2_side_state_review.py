from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.ai_table_roi import TableROI
from backend.config import PROJECT_ROOT
from backend.production_defaults import PRODUCTION_RALLY_DEFAULTS
from backend.production_jobs import load_match_job
from backend.step3_rally_start_review import (
    Step3PlayerContext,
    Step3SideIdentificationConfig,
    build_step3_2_side_state_review,
)


def _fmt_mmss(value: float) -> str:
    value = max(0.0, float(value))
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{minutes:02d}:{seconds:06.3f}"


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
    side_id = summary.get("side_identification") or {}
    first_server = summary.get("first_server") or {}
    lines = [
        "# Step 3.2 Side State Debug",
        "",
        f"- Total starts: {summary.get('total', 0)}",
        f"- Detected starts: {summary.get('detected_total', summary.get('total', 0))}",
        f"- Scoring rallies: {summary.get('scoring', 0)}",
        f"- LET/non-scoring: {summary.get('lets', 0)}",
        f"- Needs-review rows: {summary.get('needs_review', 0)}",
        f"- Side evidence: identified {side_id.get('identified', 0)} / inferred {side_id.get('inferred', 0)} / unknown {side_id.get('unknown', 0)} "
        f"({side_id.get('algorithm', side_id.get('reason', ''))})",
        f"- Retry unknown: attempted {side_id.get('retry_attempted', 0)}, "
        f"identified {side_id.get('retry_identified', 0)}",
        f"- Strong-candidate promotions: {side_id.get('promoted_strong_candidate', 0)}",
        f"- Continuity fills: {side_id.get('continuity_filled', 0)}",
        f"- First server: {first_server.get('server_player_name', 'unknown')} "
        f"(side={first_server.get('current_side', '-')}, role={first_server.get('starter_role', '-')}, "
        f"start_time={_fmt_mmss(float(first_server.get('source_t_start', first_server.get('t_start', 0.0))))})",
        f"- Source Step 3.1 events: {summary.get('source_step3_1_events_json_path', '')}",
        f"- Events JSON: {summary.get('events_json_path', '')}",
        f"- CSV: {summary.get('csv_path', '')}",
        f"- Start frames: {summary.get('start_frames_dir', '')}",
        "",
        "| id | kind | start_time | server | current_side | side_evidence | mode | note | image |",
        "|---|---|---:|---|---|---|---|---|---|",
    ]
    for event in events:
        note = event.get("review_reason", "") or ""
        expected_role = event.get("serve_order_expected_role", "") or ""
        expected_server = event.get("serve_order_expected_server_name", "") or ""
        if note and expected_role:
            note = f"{note}; expected {expected_server} ({expected_role})"
        side_evidence = event.get("side_evidence_status", "") or ""
        identified = event.get("side_identified_player_name", "") or ""
        identified_side = event.get("side_identified_current_side", "") or ""
        if side_evidence == "identified" and identified:
            side_evidence = f"{identified}={identified_side}"
        elif side_evidence == "inferred":
            side_evidence = f"inferred({event.get('side_evidence_reason', '')})"
        elif side_evidence == "unknown" and event.get("side_evidence_reason"):
            side_evidence = f"unknown({event.get('side_evidence_reason')})"
        lines.append(
            "| {id} | {kind} | {start_time} | {server} | {side} | {evidence} | {mode} | {note} | {image} |".format(
                id=event.get("id", ""),
                kind=event.get("kind", ""),
                start_time=_fmt_mmss(float(event.get("source_t_start", event.get("t_start", 0.0)))),
                server=event.get("server_player_name", "unknown"),
                side=event.get("current_side", "unknown"),
                evidence=side_evidence,
                mode=event.get("side_evidence_window_mode", ""),
                note=note,
                image=event.get("image_file", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Step 3.2 side-state review from existing Step 3.1 events.")
    parser.add_argument("--video", required=True, help="Full/source video path used by Step 3.1 source timestamps.")
    parser.add_argument("--events-json", required=True, help="Step 3.1 events JSON.")
    parser.add_argument("--out-dir", default="", help="Output folder. Defaults to sibling step3_2_side_state_review.")
    parser.add_argument("--job-json", default="", help="Existing MatchJob JSON to reuse Step 2 names and table ROI.")
    parser.add_argument("--player-a-name", default="", help="Trusted player_a name, initial near side if no job JSON is provided.")
    parser.add_argument("--player-b-name", default="", help="Trusted player_b name, initial far side if no job JSON is provided.")
    parser.add_argument("--player-a-starts-near", action="store_true", default=True)
    parser.add_argument("--face-db", default=str(PROJECT_ROOT / "data" / "players" / "faces.json"), help="FaceDB path.")
    parser.add_argument("--face-model", default=str(PROJECT_ROOT / "data" / "models" / "face" / "w600k_r50.onnx"), help="ArcFace ONNX model path.")
    parser.add_argument("--side-id-before-sec", type=float, default=1.00)
    parser.add_argument("--side-id-after-sec", type=float, default=4.00)
    parser.add_argument("--side-id-break-gap-sec", type=float, default=12.00)
    parser.add_argument("--side-id-next-guard-sec", type=float, default=0.25)
    parser.add_argument("--side-id-sample-fps", type=float, default=4.0)
    parser.add_argument("--side-id-threshold", type=float, default=0.35)
    parser.add_argument("--side-id-margin", type=float, default=0.04)
    parser.add_argument("--side-id-min-best-sim", type=float, default=0.45)
    parser.add_argument("--side-id-min-avg-sim", type=float, default=0.38)
    parser.add_argument("--side-id-min-samples", type=int, default=4)
    parser.add_argument("--side-id-no-retry-unknown", action="store_true")
    parser.add_argument("--side-id-retry-after-sec", type=float, default=12.00)
    parser.add_argument("--side-id-retry-fps", type=float, default=8.0)
    parser.add_argument("--side-id-retry-min-samples", type=int, default=3)
    parser.add_argument("--side-id-no-promote-strong-candidate", action="store_true")
    parser.add_argument("--side-id-no-continuity-fill", action="store_true")
    parser.add_argument("--side-id-continuity-terminal-gap-sec", type=float, default=12.00)
    parser.add_argument("--enable-jersey-side-id", action="store_true")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
    video_path = video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    source_events_json_path = Path(args.events_json)
    if not source_events_json_path.is_absolute():
        source_events_json_path = PROJECT_ROOT / source_events_json_path
    source_events_json_path = source_events_json_path.resolve()
    if not source_events_json_path.exists():
        raise FileNotFoundError(f"Step 3.1 events JSON not found: {source_events_json_path}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        out_dir = source_events_json_path.parent.parent / "step3_2_side_state_review"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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

        table_roi, player_zone_xyxy = detect_table_roi_and_player_zone(
            video_path,
            PRODUCTION_RALLY_DEFAULTS.table_weights_path,
        )
    if table_roi is None or table_roi.w <= 0:
        raise RuntimeError("Table ROI detection failed")

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
        raise ValueError("Step 3.2 requires trusted Step 2 player names")

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
    result = build_step3_2_side_state_review(
        video_path=video_path,
        source_events_json_path=source_events_json_path,
        events_json_path=out_dir / "step3_2_side_state_events.json",
        frame_dir=out_dir / "side_state_frames",
        table_roi=table_roi,
        pose_weights_path=PRODUCTION_RALLY_DEFAULTS.pose_weights_path,
        player_context=player_context,
        face_db_path=Path(args.face_db),
        face_model_path=Path(args.face_model),
        player_zone_xyxy=player_zone_xyxy,
        side_identification_config=side_id_config,
        time_field="source_t_start",
        end_time_field="source_t_end",
        log_fn=lambda msg: print(f"[step3.2] {msg}", flush=True),
    )
    summary_path = out_dir / "summary.md"
    _write_summary_md(summary_path, result.summary, result.events)

    side_id = result.summary.get("side_identification", {})
    print(
        "[step3.2] side state complete: "
        f"identified={side_id.get('identified', 0)} unknown={side_id.get('unknown', 0)} "
        f"retry_identified={side_id.get('retry_identified', 0)}"
    )
    print(f"[step3.2] summary: {summary_path}")
    print(f"[step3.2] events:  {result.summary.get('events_json_path', '')}")
    print(f"[step3.2] frames:  {result.summary.get('start_frames_dir', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
