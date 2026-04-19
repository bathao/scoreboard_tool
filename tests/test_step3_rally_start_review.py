from __future__ import annotations

from pathlib import Path

from backend.rally_timeline_contract import RallyTimeline, RallyTimelinePoint
from backend.step3_rally_start_review import (
    Step3PlayerContext,
    annotate_serve_order_rule_reviews,
    server_identity_for_starter_role,
    serve_order_review_markers,
    summarize_rally_start_events,
    timeline_total_rally_start_events,
)


def test_server_identity_maps_role_a_b_to_step2_names() -> None:
    context = Step3PlayerContext(
        player_a_name="Near Player",
        player_b_name="Far Player",
        player_a_starts_near=True,
    )

    assert server_identity_for_starter_role("A", context)["server_player_name"] == "Near Player"
    assert server_identity_for_starter_role("B", context)["server_player_name"] == "Far Player"
    assert server_identity_for_starter_role("", context)["server_player_name"] == "unknown"


def test_timeline_total_rally_start_events_merges_lets_and_server_names() -> None:
    timeline = RallyTimeline(
        points=[
            RallyTimelinePoint(
                id="pt_0001",
                t_start=1.2,
                t_end=2.8,
                starter_role="B",
                flags=["player_sandwich", "rally_label_point"],
            ),
            RallyTimelinePoint(
                id="pt_0002",
                t_start=8.0,
                t_end=9.5,
                starter_role="A",
                flags=["player_sandwich", "rally_label_point"],
            ),
        ],
        analysis_metadata={
            "excluded_let_starts": [
                {
                    "t_start": 4.0,
                    "t_end": 4.8,
                    "starter_role": "B",
                    "flags": ["player_sandwich", "rally_label_let", "let_no_score"],
                }
            ],
            "unattached_trailing_let_starts": [],
        },
    )
    context = Step3PlayerContext(
        player_a_name="Near Player",
        player_b_name="Far Player",
        player_a_starts_near=True,
    )

    events = timeline_total_rally_start_events(timeline, player_context=context)

    assert [event["id"] for event in events] == ["rally_0001", "rally_0002", "rally_0003"]
    assert [event["kind"] for event in events] == ["scoring", "let", "scoring"]
    assert [event["server_player_name"] for event in events] == ["Far Player", "Far Player", "Near Player"]
    assert events[1]["source"] == "excluded_let_starts"


def test_summarize_rally_start_events_exposes_first_server() -> None:
    events = [
        {
            "id": "rally_0001",
            "kind": "scoring",
            "t_start": 1.2,
            "t_end": 2.8,
            "starter_role": "B",
            "server_initial_side": "far",
            "server_player_key": "player_b",
            "server_player_name": "Far Player",
            "server_identity_source": "step2_initial_role_map",
        },
        {
            "id": "rally_0002",
            "kind": "let",
            "t_start": 4.0,
            "t_end": 4.8,
            "starter_role": "B",
            "server_initial_side": "far",
            "server_player_key": "player_b",
            "server_player_name": "Far Player",
            "server_identity_source": "step2_initial_role_map",
        },
    ]

    summary = summarize_rally_start_events(
        events,
        timeline_path=Path("timeline.json"),
        events_json_path=Path("events.json"),
        export_info={"start_frames_dir": "frames", "csv_path": "frames/rally_start_times.csv", "image_count": 2},
    )

    assert summary["total"] == 2
    assert summary["scoring"] == 1
    assert summary["lets"] == 1
    assert summary["needs_review"] == 0
    assert summary["first_server"]["server_player_name"] == "Far Player"


def test_serve_order_review_markers_flag_double_serve_singleton_gap() -> None:
    context = Step3PlayerContext(
        player_a_name="Near Player",
        player_b_name="Far Player",
        player_a_starts_near=True,
    )
    events = [
        {"id": "rally_0001", "kind": "scoring", "t_start": 1.0, "t_end": 2.0, "starter_role": "B"},
        {"id": "rally_0002", "kind": "scoring", "t_start": 8.0, "t_end": 9.0, "starter_role": "B"},
        {"id": "rally_0003", "kind": "scoring", "t_start": 20.0, "t_end": 24.0, "starter_role": "A"},
        {"id": "rally_0004", "kind": "scoring", "t_start": 40.0, "t_end": 42.0, "starter_role": "B"},
        {"id": "rally_0005", "kind": "scoring", "t_start": 50.0, "t_end": 52.0, "starter_role": "B"},
    ]

    markers = serve_order_review_markers(events, player_context=context)

    assert len(markers) == 1
    assert markers[0]["kind"] == "needs_review"
    assert markers[0]["starter_role"] == "A"
    assert markers[0]["server_player_name"] == "Near Player"
    assert markers[0]["t_start"] == 32.0
    assert markers[0]["review_reason"] == "double_serve_singleton_gap"


def test_serve_order_audit_flags_let_from_wrong_server() -> None:
    context = Step3PlayerContext(
        player_a_name="Near Player",
        player_b_name="Far Player",
        player_a_starts_near=True,
    )
    events = [
        {"id": "rally_0001", "kind": "scoring", "t_start": 1.0, "t_end": 2.0, "starter_role": "B"},
        {"id": "rally_0002", "kind": "scoring", "t_start": 8.0, "t_end": 9.0, "starter_role": "B"},
        {"id": "rally_0003", "kind": "scoring", "t_start": 20.0, "t_end": 21.0, "starter_role": "A"},
        {"id": "rally_0004", "kind": "let", "t_start": 24.0, "t_end": 24.8, "starter_role": "B"},
        {"id": "rally_0005", "kind": "needs_review", "t_start": 28.0, "t_end": 28.0, "starter_role": "A"},
        {"id": "rally_0006", "kind": "scoring", "t_start": 35.0, "t_end": 36.0, "starter_role": "B"},
        {"id": "rally_0007", "kind": "scoring", "t_start": 44.0, "t_end": 45.0, "starter_role": "B"},
    ]

    annotate_serve_order_rule_reviews(events, player_context=context)

    assert events[3]["serve_order_expected_role"] == "A"
    assert events[3]["serve_order_expected_server_name"] == "Near Player"
    assert events[3]["serve_order_ok"] is False
    assert events[3]["review_reason"] == "let_server_conflicts_with_expected_turn"
