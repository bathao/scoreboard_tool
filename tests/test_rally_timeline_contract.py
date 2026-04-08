from backend.rally_timeline_contract import RallyTimeline, RallyTimelinePoint, counts_toward_score, to_core_rally_events


def test_counts_toward_score_false_for_let_flags():
    point = RallyTimelinePoint(
        id="pt_0001",
        t_start=1.0,
        t_end=1.8,
        winner="player_a",
        flags=["player_state_machine", "rally_label_let", "let_no_score"],
    )

    assert counts_toward_score(point) is False


def test_to_core_rally_events_skips_non_scoring_let_points():
    timeline = RallyTimeline(
        video_path="demo.mp4",
        video_fps=30.0,
        roi={"x": 0, "y": 0, "w": 100, "h": 50},
        points=[
            RallyTimelinePoint(
                id="pt_0001",
                t_start=1.0,
                t_end=2.0,
                winner="player_a",
                flags=["rally_label_point"],
            ),
            RallyTimelinePoint(
                id="pt_0002",
                t_start=3.0,
                t_end=3.8,
                winner="player_b",
                flags=["player_state_machine", "rally_label_let", "let_no_score"],
            ),
            RallyTimelinePoint(
                id="pt_0003",
                t_start=5.0,
                t_end=6.0,
                winner="player_b",
                flags=["rally_label_point"],
            ),
        ],
    )

    core = to_core_rally_events(timeline)

    assert [event.winner for event in core] == ["player_a", "player_b"]
    assert [event.timestamp for event in core] == [2.0, 6.0]
    assert timeline.build_summary()["non_scoring_rallies"] == 1


def test_rally_timeline_point_roundtrip_preserves_starter_role():
    point = RallyTimelinePoint(
        id="pt_0001",
        t_start=1.0,
        t_end=2.0,
        active_start=1.0,
        active_end=5.0,
        search_upper_bound=4.5,
        starter_role="B",
        preceding_let_count=2,
        preceding_let_starts=[0.4, 0.8],
        service_attempt_index=3,
        boundary_mode="next_start_exclusive",
        endpoint_mode="last_live_island",
        endpoint_confidence=0.77,
        point_end_event="clean_winner_like",
        winner_candidate="player_a",
        winner_confidence=0.86,
        winner_decision="auto",
        winner_reason="clear last shot",
        winner_model="Qwen3-VL-4B-Instruct",
        winner_end_category="touched_but_out",
        winner_loser_candidate="player_b",
        winner_last_hitter_candidate="player_a",
        winner="unknown",
        confidence=0.8,
        flags=["player_only"],
    )

    restored = RallyTimelinePoint.from_dict(point.to_dict())

    assert restored.active_start == 1.0
    assert restored.active_end == 5.0
    assert restored.search_upper_bound == 4.5
    assert restored.starter_role == "B"
    assert restored.preceding_let_count == 2
    assert restored.preceding_let_starts == [0.4, 0.8]
    assert restored.service_attempt_index == 3
    assert restored.boundary_mode == "next_start_exclusive"
    assert restored.endpoint_mode == "last_live_island"
    assert restored.endpoint_confidence == 0.77
    assert restored.point_end_event == "clean_winner_like"
    assert restored.winner_candidate == "player_a"
    assert restored.winner_confidence == 0.86
    assert restored.winner_decision == "auto"
    assert restored.winner_reason == "clear last shot"
    assert restored.winner_model == "Qwen3-VL-4B-Instruct"
    assert restored.winner_end_category == "touched_but_out"
    assert restored.winner_loser_candidate == "player_b"
    assert restored.winner_last_hitter_candidate == "player_a"

