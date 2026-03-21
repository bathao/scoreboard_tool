from backend.ai_contract import DraftMatch, DraftPointEvent, counts_toward_score, to_core_rally_events


def test_counts_toward_score_false_for_let_flags():
    point = DraftPointEvent(
        id="pt_0001",
        t_start=1.0,
        t_end=1.8,
        winner="player_a",
        flags=["player_state_machine", "rally_label_let", "let_no_score"],
    )

    assert counts_toward_score(point) is False


def test_to_core_rally_events_skips_non_scoring_let_points():
    draft = DraftMatch(
        video_path="demo.mp4",
        video_fps=30.0,
        roi={"x": 0, "y": 0, "w": 100, "h": 50},
        points=[
            DraftPointEvent(
                id="pt_0001",
                t_start=1.0,
                t_end=2.0,
                winner="player_a",
                flags=["rally_label_point"],
            ),
            DraftPointEvent(
                id="pt_0002",
                t_start=3.0,
                t_end=3.8,
                winner="player_b",
                flags=["player_state_machine", "rally_label_let", "let_no_score"],
            ),
            DraftPointEvent(
                id="pt_0003",
                t_start=5.0,
                t_end=6.0,
                winner="player_b",
                flags=["rally_label_point"],
            ),
        ],
    )

    core = to_core_rally_events(draft)

    assert [event.winner for event in core] == ["player_a", "player_b"]
    assert [event.timestamp for event in core] == [2.0, 6.0]
    assert draft.build_summary()["non_scoring_rallies"] == 1
