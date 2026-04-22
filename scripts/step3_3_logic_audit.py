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
    Step3LogicAuditConfig,
    Step3PlayerContext,
    Step3SideIdentificationConfig,
    build_step3_3_logic_audit_review,
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


def _side_player_name(event: dict, side: str, player_context: Step3PlayerContext | None) -> str:
    side = str(side or "").upper()
    if side not in {"NEAR", "FAR"} or player_context is None:
        return "unknown"
    if str(event.get("player_a_current_side", "") or "").upper() == side:
        return player_context.player_a_name
    if str(event.get("player_b_current_side", "") or "").upper() == side:
        return player_context.player_b_name
    return "unknown"


def _player_name_for_key(player_key: str, player_context: Step3PlayerContext | None) -> str:
    if player_context is None:
        return "unknown"
    if player_key == "player_a":
        return player_context.player_a_name
    if player_key == "player_b":
        return player_context.player_b_name
    return "unknown"


def _side_for_player_key(event: dict, player_key: str) -> str:
    if player_key == "player_a":
        return str(event.get("player_a_current_side", "") or "unknown").upper()
    if player_key == "player_b":
        return str(event.get("player_b_current_side", "") or "unknown").upper()
    return "unknown"


def _other_player_key(player_key: str) -> str:
    if player_key == "player_a":
        return "player_b"
    if player_key == "player_b":
        return "player_a"
    return ""


def _side_evidence_label(event: dict) -> str:
    side_evidence = str(event.get("side_evidence_status", "") or "")
    identified = str(event.get("side_identified_player_name", "") or "")
    identified_side = str(event.get("side_identified_current_side", "") or "")
    if side_evidence == "identified" and identified:
        return f"{identified}={identified_side}"
    if side_evidence == "inferred":
        return f"inferred({event.get('side_evidence_reason', '')})"
    if side_evidence == "unknown" and event.get("side_evidence_reason"):
        return f"unknown({event.get('side_evidence_reason')})"
    return side_evidence or "unknown"


def _issue_map_by_event(summary: dict) -> dict[str, list[str]]:
    audit = summary.get("logic_audit") or {}
    by_event: dict[str, list[str]] = {}
    for issue in list(audit.get("issues") or []):
        issue_type = str(issue.get("type", "") or "")
        for event_id in issue.get("event_ids", []) or []:
            if issue_type:
                by_event.setdefault(str(event_id), []).append(issue_type)
    return by_event


def _write_summary_md(
    path: Path,
    summary: dict,
    events: list[dict],
    *,
    player_context: Step3PlayerContext | None = None,
) -> None:
    audit = summary.get("logic_audit") or {}
    iterations = list(summary.get("logic_audit_iterations") or [])
    side_id = summary.get("side_identification") or {}
    scoring = sum(1 for event in events if event.get("kind") == "scoring")
    lets = sum(1 for event in events if event.get("kind") == "let")
    needs_review = sum(1 for event in events if event.get("kind") == "needs_review")
    gate = "PASS -> Step 3.4 allowed" if summary.get("logic_ok") else "BLOCKED at Step 3.3"
    lines = [
        "# Step 3 Rally Summary",
        "",
        f"- Gate: {gate}",
        f"- Logic OK: {'yes' if summary.get('logic_ok') else 'no'}",
        f"- Blocking issues: {summary.get('logic_blocking_issue_count', 0)}",
        f"- Total rows: {len(events)} ({scoring} scoring + {lets} LET + {needs_review} needs-review)",
        f"- Side evidence: identified {side_id.get('identified', 0)} / inferred {side_id.get('inferred', 0)} / unknown {side_id.get('unknown', 0)}",
        f"- Logic segments: {audit.get('segments', 0)}",
        f"- Step 3.2 targeted rescans attempted/requested: {len(audit.get('rescan_event_ids', []) or [])}",
        f"- Possible Step 3.1 gap-rescan rows: {len(audit.get('requires_step3_1_gap_rescan_event_ids', []) or [])}",
        f"- Start time column uses the full input video timeline (`source_t_start`).",
        f"- Audited events JSON: {summary.get('events_json_path', '')}",
        "",
        "## Repair Iterations",
        "",
        "| iteration | ok | blocking_issues | rescan_rows | step3_1_gap_rows |",
        "|---:|---|---:|---:|---:|",
    ]
    for item in iterations:
        lines.append(
            "| {iteration} | {ok} | {blocking} | {rescans} | {gaps} |".format(
                iteration=item.get("iteration", ""),
                ok="yes" if item.get("ok") else "no",
                blocking=item.get("blocking_issue_count", 0),
                rescans=len(item.get("rescan_event_ids", []) or []),
                gaps=len(item.get("requires_step3_1_gap_rescan_event_ids", []) or []),
            )
        )

    issues = list(audit.get("issues") or [])
    lines.extend(
        [
            "",
            "## Issues",
            "",
            "| id | type | severity | route | events | message |",
            "|---|---|---|---|---|---|",
        ]
    )
    if issues:
        for issue in issues:
            lines.append(
                "| {id} | {typ} | {severity} | {route} | {events} | {message} |".format(
                    id=issue.get("id", ""),
                    typ=issue.get("type", ""),
                    severity=issue.get("severity", ""),
                    route=issue.get("repair_route", ""),
                    events=", ".join(issue.get("event_ids", []) or []),
                    message=str(issue.get("message", "")).replace("|", "/"),
                )
            )
    else:
        lines.append("| - | - | - | - | - | No logic issues found |")

    issue_by_event = _issue_map_by_event(summary)
    lines.extend(
        [
            "",
            "## Timeline",
            "",
            "| id | kind | start_time | server | server_side | opponent | opponent_side | logic | issue | image |",
            "|---|---|---:|---|---|---|---|---|---|---|",
        ]
    )
    for event in events:
        logic_ok = event.get("step3_3_logic_ok", "")
        if logic_ok == "":
            logic_label = ""
        else:
            logic_label = "OK" if bool(logic_ok) else "BLOCK"
        event_id = str(event.get("id", "") or "")
        issue = ", ".join(issue_by_event.get(event_id, []))
        server_key = str(event.get("server_player_key", "") or "")
        opponent_key = _other_player_key(server_key)
        lines.append(
            "| {id} | {kind} | {start_time} | {server} | {server_side} | {opponent} | {opponent_side} | {logic_ok} | {issue} | {image} |".format(
                id=event_id,
                kind=event.get("kind", ""),
                start_time=_fmt_mmss(float(event.get("source_t_start", event.get("t_start", 0.0)))),
                server=event.get("server_player_name", "unknown"),
                server_side=_side_for_player_key(event, server_key),
                opponent=_player_name_for_key(opponent_key, player_context),
                opponent_side=_side_for_player_key(event, opponent_key),
                logic_ok=logic_label,
                issue=issue,
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

    parser = argparse.ArgumentParser(description="Step 3.3 logic audit and targeted Step 3.2 repair.")
    parser.add_argument("--video", required=True, help="Full/source video path used by Step 3.2 source timestamps.")
    parser.add_argument("--events-json", required=True, help="Step 3.2 events JSON.")
    parser.add_argument("--out-dir", default="", help="Output folder. Defaults to sibling step3_3_logic_audit.")
    parser.add_argument("--job-json", default="", help="Existing MatchJob JSON to reuse Step 2 names and table ROI.")
    parser.add_argument("--player-a-name", default="", help="Trusted player_a name, initial near side if no job JSON is provided.")
    parser.add_argument("--player-b-name", default="", help="Trusted player_b name, initial far side if no job JSON is provided.")
    parser.add_argument("--player-a-starts-near", action="store_true", default=True)
    parser.add_argument("--face-db", default=str(PROJECT_ROOT / "data" / "players" / "faces.json"))
    parser.add_argument("--face-model", default=str(PROJECT_ROOT / "data" / "models" / "face" / "w600k_r50.onnx"))
    parser.add_argument("--max-repair-iterations", type=int, default=2)
    parser.add_argument("--set-boundary-gap-sec", type=float, default=12.0)
    parser.add_argument("--min-scoring-before-set-boundary", type=int, default=11)
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
        raise FileNotFoundError(f"Step 3.2 events JSON not found: {source_events_json_path}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        out_dir = source_events_json_path.parent.parent / "step3_3_logic_audit"
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
        raise ValueError("Step 3.3 requires trusted Step 2 player names")

    result = build_step3_3_logic_audit_review(
        video_path=video_path,
        source_events_json_path=source_events_json_path,
        events_json_path=out_dir / "step3_3_logic_audited_events.json",
        table_roi=table_roi,
        pose_weights_path=PRODUCTION_RALLY_DEFAULTS.pose_weights_path,
        player_context=player_context,
        face_db_path=Path(args.face_db),
        face_model_path=Path(args.face_model),
        player_zone_xyxy=player_zone_xyxy,
        side_identification_config=Step3SideIdentificationConfig(),
        logic_audit_config=Step3LogicAuditConfig(
            max_repair_iterations=int(args.max_repair_iterations),
            set_boundary_gap_sec=float(args.set_boundary_gap_sec),
            min_scoring_before_set_boundary=int(args.min_scoring_before_set_boundary),
        ),
        time_field="source_t_start",
        end_time_field="source_t_end",
        log_fn=lambda msg: print(f"[step3.3] {msg}", flush=True),
    )
    summary_path = source_events_json_path.parent / "summary.md"
    _write_summary_md(summary_path, result.summary, result.events, player_context=player_context)
    print(
        "[step3.3] logic audit complete: "
        f"ok={bool(result.summary.get('logic_ok'))} "
        f"blocking={result.summary.get('logic_blocking_issue_count', 0)}"
    )
    print(f"[step3.3] updated canonical summary: {summary_path}")
    print(f"[step3.3] events:  {result.summary.get('events_json_path', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
