from pathlib import Path
import shutil
import uuid

from backend.production_jobs import (
    accept_point_prediction,
    apply_point_no_score,
    apply_point_review,
    build_review_status,
    create_match_job,
    load_match_job,
    parse_timecode_to_seconds,
)
from backend.rally_timeline_contract import RallyTimeline, RallyTimelinePoint


def _make_timeline() -> RallyTimeline:
    return RallyTimeline(
        schema_version="rally_timeline_v1",
        video_path="C:/video.mp4",
        video_fps=30.0,
        best_of=5,
        roi={"x": 0, "y": 0, "w": 100, "h": 100},
        points=[
            RallyTimelinePoint(
                id="pt_0001",
                t_start=1.0,
                t_end=2.0,
                winner="player_a",
                winner_candidate="player_a",
                winner_decision="review",
            ),
            RallyTimelinePoint(
                id="pt_0002",
                t_start=3.0,
                t_end=4.0,
                winner="player_b",
                winner_candidate="player_b",
                winner_decision="auto",
            ),
            RallyTimelinePoint(
                id="pt_0003",
                t_start=5.0,
                t_end=6.0,
                winner="unknown",
                winner_candidate="unknown",
                winner_decision="blocked",
                flags=["let_no_score"],
            ),
        ],
    )


def test_parse_timecode_to_seconds_supports_common_formats():
    assert parse_timecode_to_seconds("90") == 90.0
    assert parse_timecode_to_seconds("01:30") == 90.0
    assert parse_timecode_to_seconds("00:01:30") == 90.0


def test_build_review_status_ignores_non_scoring_points_for_final_gate():
    timeline = _make_timeline()

    status = build_review_status(timeline)

    assert status["scoring_points"] == 2
    assert status["non_scoring_points"] == 1
    assert status["unresolved_scoring_points"] == 1
    assert status["final_export_ready"] is False


def test_apply_point_review_marks_point_resolved_and_logs_correction():
    timeline = _make_timeline()

    point = apply_point_review(
        timeline,
        point_id="pt_0001",
        winner="player_b",
        reviewer="tester",
        note="manual correction",
    )

    assert point.winner == "player_b"
    assert point.winner_decision == "auto"
    assert point.source == "human"
    assert len(point.corrections) == 1
    assert point.corrections[0].by == "tester"


def test_accept_point_prediction_promotes_review_to_auto():
    timeline = _make_timeline()

    point = accept_point_prediction(timeline, point_id="pt_0001", reviewer="tester")

    assert point.winner == "player_a"
    assert point.winner_decision == "auto"
    assert point.source == "human"


def test_apply_point_no_score_marks_point_as_let_and_removes_it_from_score_gate():
    timeline = _make_timeline()

    point = apply_point_no_score(timeline, point_id="pt_0001", reviewer="tester")
    status = build_review_status(timeline)

    assert point.winner == "unknown"
    assert point.winner_decision == "auto"
    assert "let_no_score" in point.flags
    assert status["scoring_points"] == 1
    assert status["final_export_ready"] is True


def test_create_and_load_match_job_roundtrip():
    root = Path("test_runtime_jobs") / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=True)
    try:
        raw_video = root / "raw_input.mp4"
        raw_video.write_bytes(b"fake")

        created = create_match_job(
            raw_video_path=str(raw_video),
            player_a_name="Near",
            player_b_name="Far",
            trim_start_sec=12.5,
            best_of=5,
            jobs_root=root / "jobs",
        )
        loaded = load_match_job(created.artifacts.job_json)

        assert loaded.job_id == created.job_id
        assert loaded.player_a_name == "Near"
        assert loaded.player_b_name == "Far"
        assert loaded.trim_start_sec == 12.5
    finally:
        shutil.rmtree(root, ignore_errors=True)
