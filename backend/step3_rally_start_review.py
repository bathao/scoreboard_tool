from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

from backend.config import PROJECT_ROOT
from backend.rally_timeline_contract import (
    RallyTimeline,
    counts_toward_score,
    load_rally_timeline,
    save_rally_timeline,
)
from backend.ai_multistream_rally import _infer_player_serve_mode_from_starter_roles


SCRIPTS_DIR = PROJECT_ROOT / "scripts"
STEP3_1_ALGORITHM = "total_rally_start_time_review_v2"


@dataclass(frozen=True)
class Step3PlayerContext:
    """Trusted Step 2 identity mapping for the initial set-1 sides."""

    player_a_name: str
    player_b_name: str
    player_a_starts_near: bool = True


@dataclass
class Step3RallyStartReviewResult:
    timeline: RallyTimeline
    events: list[dict[str, Any]]
    summary: dict[str, Any]


def _ensure_scripts_importable() -> None:
    scripts_path = str(SCRIPTS_DIR)
    if scripts_path not in sys.path:
        sys.path.append(scripts_path)


def _load_build_rally_timeline():
    _ensure_scripts_importable()
    from generate_rally_timeline import build_rally_timeline  # type: ignore

    return build_rally_timeline


def server_identity_for_starter_role(
    starter_role: str | None,
    player_context: Step3PlayerContext | None,
) -> dict[str, str]:
    """Map role-tracker starter role A/B to the trusted Step 2 player name.

    The current role tracker seeds role A on the initial near-side player and
    role B on the initial far-side player. Step 2 owns the actual names.
    """

    role = str(starter_role or "").strip()
    if role not in {"A", "B"}:
        return {
            "server_initial_side": "",
            "server_player_key": "",
            "server_player_name": "unknown",
            "server_identity_source": "unknown_starter_role",
        }
    if player_context is None:
        return {
            "server_initial_side": "near" if role == "A" else "far",
            "server_player_key": "",
            "server_player_name": "unknown",
            "server_identity_source": "starter_role_only",
        }

    near_key = "player_a" if bool(player_context.player_a_starts_near) else "player_b"
    far_key = "player_b" if bool(player_context.player_a_starts_near) else "player_a"
    key = near_key if role == "A" else far_key
    name = player_context.player_a_name if key == "player_a" else player_context.player_b_name
    return {
        "server_initial_side": "near" if role == "A" else "far",
        "server_player_key": key,
        "server_player_name": str(name).strip() or "unknown",
        "server_identity_source": "step2_initial_role_map",
    }


def _base_event(
    *,
    kind: str,
    source: str,
    point_id: str,
    t_start: float,
    t_end: float,
    starter_role: str,
    flags: list[str],
    player_context: Step3PlayerContext | None,
    source_time_offset_sec: float = 0.0,
) -> dict[str, Any]:
    source_t_start = float(t_start) + float(source_time_offset_sec)
    source_t_end = float(t_end) + float(source_time_offset_sec)
    event = {
        "id": "",
        "kind": str(kind),
        "source": str(source),
        "point_id": str(point_id),
        "t_start": float(t_start),
        "t_end": float(t_end),
        "source_t_start": source_t_start,
        "source_t_end": source_t_end,
        "starter_role": str(starter_role or ""),
        "flags": list(flags or []),
    }
    event.update(server_identity_for_starter_role(event["starter_role"], player_context))
    return event


def timeline_total_rally_start_events(
    timeline: RallyTimeline,
    *,
    player_context: Step3PlayerContext | None = None,
    include_serve_order_review_markers: bool = True,
    source_time_offset_sec: float = 0.0,
) -> list[dict[str, Any]]:
    """Merge scoring points and existing LET starts into one review list."""

    events: list[dict[str, Any]] = []
    for point in timeline.points:
        is_scoring = counts_toward_score(point)
        events.append(
            _base_event(
                kind="scoring" if is_scoring else "let",
                source="timeline_point",
                point_id=str(point.id),
                t_start=float(point.t_start),
                t_end=float(point.t_end),
                starter_role=point.starter_role or "",
                flags=list(point.flags),
                player_context=player_context,
                source_time_offset_sec=source_time_offset_sec,
            )
        )

    metadata = timeline.analysis_metadata if isinstance(timeline.analysis_metadata, dict) else {}
    for bucket in ("excluded_let_starts", "unattached_trailing_let_starts"):
        for item in metadata.get(bucket, []) or []:
            if not isinstance(item, dict):
                continue
            t_start = float(item.get("t_start", 0.0))
            events.append(
                _base_event(
                    kind="let",
                    source=bucket,
                    point_id="",
                    t_start=t_start,
                    t_end=float(item.get("t_end", t_start)),
                    starter_role=str(item.get("starter_role", "") or ""),
                    flags=list(item.get("flags", []) or []),
                    player_context=player_context,
                    source_time_offset_sec=source_time_offset_sec,
                )
            )

    events.sort(key=lambda row: (float(row["t_start"]), float(row["t_end"])))
    if include_serve_order_review_markers:
        events.extend(
            serve_order_review_markers(
                events,
                player_context=player_context,
                source_time_offset_sec=source_time_offset_sec,
            )
        )
        events.sort(key=lambda row: (float(row["t_start"]), float(row["t_end"])))
    for idx, event in enumerate(events, start=1):
        event["id"] = f"rally_{idx:04d}"
    annotate_serve_order_rule_reviews(events, player_context=player_context)
    return events


def serve_order_review_markers(
    events: list[dict[str, Any]],
    *,
    player_context: Step3PlayerContext | None = None,
    source_time_offset_sec: float = 0.0,
) -> list[dict[str, Any]]:
    """Add review-only markers when the existing serve-order engine sees a gap.

    This does not create a confirmed rally. It only marks a timestamp for the
    operator when double-serve order has a single scoring serve between two
    complete runs from the other player.
    """

    scoring_events = [event for event in events if event.get("kind") == "scoring"]
    starter_roles = [str(event.get("starter_role", "") or "") for event in scoring_events]
    if len(starter_roles) < 5:
        return []
    if _infer_player_serve_mode_from_starter_roles(starter_roles) != "double":
        return []

    runs: list[tuple[str, int, int]] = []
    run_start = 0
    while run_start < len(starter_roles):
        role = starter_roles[run_start]
        run_end = run_start + 1
        while run_end < len(starter_roles) and starter_roles[run_end] == role:
            run_end += 1
        runs.append((role, run_start, run_end))
        run_start = run_end

    markers: list[dict[str, Any]] = []
    for idx in range(1, len(runs) - 1):
        left_role, left_start, left_end = runs[idx - 1]
        mid_role, mid_start, mid_end = runs[idx]
        right_role, right_start, right_end = runs[idx + 1]
        left_len = left_end - left_start
        mid_len = mid_end - mid_start
        right_len = right_end - right_start
        if mid_len != 1:
            continue
        if left_role != right_role or left_role == mid_role:
            continue
        if left_len < 2 or right_len < 2:
            continue

        singleton = scoring_events[mid_start]
        next_event = scoring_events[right_start]
        gap_start = float(singleton.get("t_end", singleton.get("t_start", 0.0)))
        gap_end = float(next_event.get("t_start", gap_start))
        if gap_end - gap_start < 4.0:
            continue
        marker_t = float((gap_start + gap_end) / 2.0)
        marker = _base_event(
            kind="needs_review",
            source="serve_order_singleton_gap",
            point_id="",
            t_start=marker_t,
            t_end=marker_t,
            starter_role=mid_role,
            flags=[
                "serve_order_gap_review",
                "not_confirmed_rally",
                "serve_mode_double",
            ],
            player_context=player_context,
            source_time_offset_sec=source_time_offset_sec,
        )
        marker["review_reason"] = "double_serve_singleton_gap"
        marker["review_note"] = (
            "Double-serve order has a singleton scoring serve between two "
            "complete runs from the other player; operator should verify a "
            "missing rally start in this gap."
        )
        marker["gap_start"] = gap_start
        marker["gap_end"] = gap_end
        marker["source_gap_start"] = gap_start + float(source_time_offset_sec)
        marker["source_gap_end"] = gap_end + float(source_time_offset_sec)
        marker["prev_scoring_event_id"] = singleton.get("id", "")
        marker["next_scoring_event_id"] = next_event.get("id", "")
        markers.append(marker)

    return markers


def _append_unique_flag(event: dict[str, Any], flag: str) -> None:
    flags = list(event.get("flags", []) or [])
    if flag not in flags:
        flags.append(flag)
    event["flags"] = flags


def _expected_serve_role(first_role: str, score_index: int, legal_limit: int) -> str:
    other_role = "A" if first_role == "B" else "B"
    return first_role if (score_index // legal_limit) % 2 == 0 else other_role


def annotate_serve_order_rule_reviews(
    events: list[dict[str, Any]],
    *,
    player_context: Step3PlayerContext | None = None,
) -> None:
    """Flag detected rows that conflict with the inferred serve-order rule.

    Scoring rows and review-only gap markers advance the expected serve order.
    LET rows do not advance it, so a LET should match the next expected server.
    """

    scoring_like_events = [
        event
        for event in events
        if event.get("kind") in {"scoring", "needs_review"}
        and str(event.get("starter_role", "") or "") in {"A", "B"}
    ]
    starter_roles = [str(event.get("starter_role", "") or "") for event in scoring_like_events]
    if not starter_roles:
        return

    serve_mode = _infer_player_serve_mode_from_starter_roles(starter_roles)
    legal_limit = 2 if serve_mode == "double" else 1
    first_role = starter_roles[0]

    for score_index, event in enumerate(scoring_like_events):
        expected_role = _expected_serve_role(first_role, score_index, legal_limit)
        expected_identity = server_identity_for_starter_role(expected_role, player_context)
        event["serve_order_mode"] = serve_mode
        event["serve_order_index"] = score_index + 1
        event["serve_order_expected_role"] = expected_role
        event["serve_order_expected_server_name"] = expected_identity.get("server_player_name", "unknown")
        if str(event.get("starter_role", "") or "") == expected_role:
            event["serve_order_ok"] = True
            continue
        event["serve_order_ok"] = False
        event["review_reason"] = event.get("review_reason") or "serve_order_role_conflict"
        event["review_note"] = event.get("review_note") or (
            "Detected scoring-like start conflicts with the inferred table-tennis serve order."
        )
        _append_unique_flag(event, "serve_order_role_conflict")

    scoring_like_by_time = sorted(scoring_like_events, key=lambda row: float(row.get("t_start", 0.0)))
    for event in sorted(events, key=lambda row: float(row.get("t_start", 0.0))):
        if event.get("kind") != "let":
            continue
        role = str(event.get("starter_role", "") or "")
        if role not in {"A", "B"}:
            continue

        event_t = float(event.get("t_start", 0.0))
        score_index = 0
        for score_event in scoring_like_by_time:
            if float(score_event.get("t_start", 0.0)) < event_t:
                score_index += 1
                continue
            break

        expected_role = _expected_serve_role(first_role, score_index, legal_limit)
        expected_identity = server_identity_for_starter_role(expected_role, player_context)
        event["serve_order_mode"] = serve_mode
        event["serve_order_next_score_index"] = score_index + 1
        event["serve_order_expected_role"] = expected_role
        event["serve_order_expected_server_name"] = expected_identity.get("server_player_name", "unknown")
        if role == expected_role:
            event["serve_order_ok"] = True
            continue

        event["serve_order_ok"] = False
        event["review_reason"] = event.get("review_reason") or "let_server_conflicts_with_expected_turn"
        event["review_note"] = (
            "Detected LET does not match the server expected by table-tennis serve order. "
            "LET should replay the same server instead of switching service."
        )
        _append_unique_flag(event, "serve_order_role_conflict")
        _append_unique_flag(event, "let_server_conflict")


def export_rally_start_event_frames(
    video_path: str | Path,
    events: list[dict[str, Any]],
    *,
    table_roi,
    out_dir: Path,
) -> dict[str, Any]:
    """Export one annotated frame and a CSV row per rally/LET start."""

    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for rally-start frame export: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    roi_tuple = None
    if table_roi is not None:
        roi_tuple = (
            int(getattr(table_roi, "x")),
            int(getattr(table_roi, "y")),
            int(getattr(table_roi, "w")),
            int(getattr(table_roi, "h")),
        )

    rows: list[dict[str, Any]] = []
    try:
        for idx, event in enumerate(events, start=1):
            t_start = float(event["t_start"])
            t_end = float(event.get("t_end", t_start))
            frame_idx = max(0, int(round(t_start * fps)))
            if frame_count > 0:
                frame_idx = min(frame_idx, max(0, frame_count - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            image_file = ""
            if ret and frame is not None:
                if roi_tuple is not None:
                    x, y, w, h = roi_tuple
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
                    cv2.putText(
                        frame,
                        "TABLE ROI",
                        (x, max(30, y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 0, 0),
                        3,
                    )

                kind = str(event.get("kind", "")).upper()
                if kind == "SCORING":
                    color = (80, 255, 120)
                elif kind == "NEEDS_REVIEW":
                    color = (80, 170, 255)
                else:
                    color = (80, 210, 255)
                server_name = str(event.get("server_player_name", "unknown") or "unknown")
                cv2.rectangle(frame, (40, 40), (1540, 250), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"STEP 3.1 START #{idx:04d}  {kind}",
                    (60, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.15,
                    color,
                    3,
                )
                cv2.putText(
                    frame,
                    f"start={t_start:.3f}s  end={t_end:.3f}s  frame={frame_idx}",
                    (60, 124),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )
                source_t_start = event.get("source_t_start")
                if source_t_start is not None and abs(float(source_t_start) - t_start) > 0.001:
                    cv2.putText(
                        frame,
                        f"source video t={float(source_t_start):.3f}s",
                        (860, 124),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.82,
                        (255, 255, 255),
                        2,
                    )
                cv2.putText(
                    frame,
                    f"server={server_name}  role={event.get('starter_role', '') or '-'}  side={event.get('server_initial_side', '') or '-'}",
                    (60, 164),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.82,
                    (220, 220, 220),
                    2,
                )
                cv2.putText(
                    frame,
                    f"source={event.get('source', '')}  point={event.get('point_id', '') or '-'}",
                    (60, 204),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.82,
                    (220, 220, 220),
                    2,
                )
                image_file = f"{event['id']}_{event['kind']}_{t_start:08.3f}s.jpg"
                cv2.imwrite(str(out_dir / image_file), frame)

            event["frame_idx"] = int(frame_idx)
            event["image_file"] = image_file
            rows.append(
                {
                    "id": event["id"],
                    "kind": event["kind"],
                    "source": event["source"],
                    "point_id": event.get("point_id", ""),
                    "t_start": f"{t_start:.6f}",
                    "t_end": f"{t_end:.6f}",
                    "source_t_start": f"{float(event.get('source_t_start', t_start)):.6f}",
                    "source_t_end": f"{float(event.get('source_t_end', t_end)):.6f}",
                    "frame_idx": int(frame_idx),
                    "image_file": image_file,
                    "starter_role": event.get("starter_role", ""),
                    "server_initial_side": event.get("server_initial_side", ""),
                    "server_player_key": event.get("server_player_key", ""),
                    "server_player_name": event.get("server_player_name", ""),
                    "server_identity_source": event.get("server_identity_source", ""),
                    "serve_order_mode": event.get("serve_order_mode", ""),
                    "serve_order_index": event.get("serve_order_index", ""),
                    "serve_order_next_score_index": event.get("serve_order_next_score_index", ""),
                    "serve_order_expected_role": event.get("serve_order_expected_role", ""),
                    "serve_order_expected_server_name": event.get("serve_order_expected_server_name", ""),
                    "serve_order_ok": event.get("serve_order_ok", ""),
                    "review_reason": event.get("review_reason", ""),
                    "review_note": event.get("review_note", ""),
                    "gap_start": event.get("gap_start", ""),
                    "gap_end": event.get("gap_end", ""),
                    "source_gap_start": event.get("source_gap_start", ""),
                    "source_gap_end": event.get("source_gap_end", ""),
                    "flags": "|".join(str(flag) for flag in event.get("flags", []) or []),
                }
            )
    finally:
        cap.release()

    csv_path = out_dir / "rally_start_times.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "id",
            "kind",
            "source",
            "point_id",
            "t_start",
            "t_end",
            "source_t_start",
            "source_t_end",
            "frame_idx",
            "image_file",
            "starter_role",
            "server_initial_side",
            "server_player_key",
            "server_player_name",
            "server_identity_source",
            "serve_order_mode",
            "serve_order_index",
            "serve_order_next_score_index",
            "serve_order_expected_role",
            "serve_order_expected_server_name",
            "serve_order_ok",
            "review_reason",
            "review_note",
            "gap_start",
            "gap_end",
            "source_gap_start",
            "source_gap_end",
            "flags",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "start_frames_dir": str(out_dir.resolve()).replace("\\", "/"),
        "csv_path": str(csv_path.resolve()).replace("\\", "/"),
        "image_count": sum(1 for event in events if event.get("image_file")),
    }


def summarize_rally_start_events(
    events: list[dict[str, Any]],
    *,
    timeline_path: Path,
    events_json_path: Path,
    export_info: dict[str, Any],
) -> dict[str, Any]:
    scoring_count = sum(1 for event in events if event["kind"] == "scoring")
    let_count = sum(1 for event in events if event["kind"] == "let")
    gap_review_count = sum(1 for event in events if event["kind"] == "needs_review")
    rule_conflict_review_count = sum(
        1
        for event in events
        if bool(event.get("review_reason")) and event.get("kind") != "needs_review"
    )
    needs_review_count = gap_review_count + rule_conflict_review_count
    first_server = None
    if events:
        first = events[0]
        first_server = {
            "rally_id": first.get("id", ""),
            "starter_role": first.get("starter_role", ""),
            "server_initial_side": first.get("server_initial_side", ""),
            "server_player_key": first.get("server_player_key", ""),
            "server_player_name": first.get("server_player_name", "unknown"),
            "source": first.get("server_identity_source", ""),
            "t_start": float(first.get("t_start", 0.0)),
            "source_t_start": float(first.get("source_t_start", first.get("t_start", 0.0))),
        }

    return {
        "algorithm": STEP3_1_ALGORITHM,
        "total": len(events),
        "detected_total": scoring_count + let_count,
        "scoring": scoring_count,
        "lets": let_count,
        "needs_review": needs_review_count,
        "rule_gap_review_count": gap_review_count,
        "rule_conflict_review_count": rule_conflict_review_count,
        "timeline_path": str(timeline_path.resolve()).replace("\\", "/"),
        "events_json_path": str(events_json_path.resolve()).replace("\\", "/"),
        "first_server": first_server,
        **export_info,
        "events": events,
    }


def write_rally_start_events_json(
    events_json_path: Path,
    summary: dict[str, Any],
    events: list[dict[str, Any]],
) -> None:
    payload = {
        "summary": {k: v for k, v in summary.items() if k != "events"},
        "events": events,
    }
    events_json_path.parent.mkdir(parents=True, exist_ok=True)
    events_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_or_build_rally_timeline(
    *,
    video_path: str | Path,
    timeline_path: Path,
    legacy_cache_path: Path | None,
    table_weights_path: str,
    pose_weights_path: str,
    best_of: int,
    stride: int,
    mode: str,
    player_margin_px: int,
    player_fuse_gain: float,
    player_signal_source: str,
    ball_fuse_gain: float,
    ball_signal_source: str,
    table_roi,
    force_rebuild: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> RallyTimeline:
    if timeline_path.exists() and not force_rebuild:
        if log_fn:
            log_fn("Step 3.1: total rally start detection - reusing cached total-rally timeline")
        return load_rally_timeline(timeline_path)
    if legacy_cache_path is not None and legacy_cache_path.exists() and not force_rebuild:
        if log_fn:
            log_fn("Step 3.1: total rally start detection - migrating cached full-video detector output")
        timeline = load_rally_timeline(legacy_cache_path)
        save_rally_timeline(timeline_path, timeline)
        return timeline

    build_rally_timeline = _load_build_rally_timeline()
    if log_fn:
        log_fn("Step 3.1: total rally start detection - running existing start-time detector")
    timeline = build_rally_timeline(
        str(video_path),
        table_weights_path,
        pose_weights_path=pose_weights_path,
        best_of=best_of,
        stride=stride,
        mode=mode,
        player_margin_px=player_margin_px,
        player_fuse_gain=player_fuse_gain,
        player_signal_source=player_signal_source,
        ball_fuse_gain=ball_fuse_gain,
        ball_signal_source=ball_signal_source,
        table_roi=table_roi,
        log_fn=log_fn,
    )
    save_rally_timeline(timeline_path, timeline)
    return timeline


def build_step3_1_rally_start_review(
    *,
    video_path: str | Path,
    timeline_path: Path,
    events_json_path: Path,
    frame_dir: Path,
    table_roi,
    table_weights_path: str,
    pose_weights_path: str,
    best_of: int,
    stride: int,
    mode: str,
    player_margin_px: int,
    player_fuse_gain: float,
    player_signal_source: str,
    ball_fuse_gain: float,
    ball_signal_source: str,
    player_context: Step3PlayerContext | None,
    legacy_cache_path: Path | None = None,
    force_rebuild: bool = False,
    source_time_offset_sec: float = 0.0,
    log_fn: Callable[[str], None] | None = None,
) -> Step3RallyStartReviewResult:
    timeline = load_or_build_rally_timeline(
        video_path=video_path,
        timeline_path=timeline_path,
        legacy_cache_path=legacy_cache_path,
        table_weights_path=table_weights_path,
        pose_weights_path=pose_weights_path,
        best_of=best_of,
        stride=stride,
        mode=mode,
        player_margin_px=player_margin_px,
        player_fuse_gain=player_fuse_gain,
        player_signal_source=player_signal_source,
        ball_fuse_gain=ball_fuse_gain,
        ball_signal_source=ball_signal_source,
        table_roi=table_roi,
        force_rebuild=force_rebuild,
        log_fn=log_fn,
    )
    events = timeline_total_rally_start_events(
        timeline,
        player_context=player_context,
        source_time_offset_sec=source_time_offset_sec,
    )
    export_info = export_rally_start_event_frames(
        video_path,
        events,
        table_roi=table_roi,
        out_dir=frame_dir,
    )
    summary = summarize_rally_start_events(
        events,
        timeline_path=timeline_path,
        events_json_path=events_json_path,
        export_info=export_info,
    )
    write_rally_start_events_json(events_json_path, summary, events)
    return Step3RallyStartReviewResult(timeline=timeline, events=events, summary=summary)
