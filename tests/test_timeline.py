import pytest
from backend.timeline import build_match_timeline


def test_timeline_single_set_win():

    winner_sequence = ["player_a"] * 11

    timeline = build_match_timeline(best_of=5, winner_sequence=winner_sequence)

    assert len(timeline) == 11

    last_state = timeline[-1]

    assert last_state["score_a"] == 11
    assert last_state["score_b"] == 0
    assert last_state["sets_a"] == 1
    assert last_state["sets_b"] == 0
    assert last_state["is_finished"] is False


def test_timeline_match_finish():

    # player_a wins 3 sets (best_of=5 → 3 sets to win)
    winner_sequence = (
        ["player_a"] * 11 +
        ["player_a"] * 11 +
        ["player_a"] * 11
    )

    timeline = build_match_timeline(best_of=5, winner_sequence=winner_sequence)

    last_state = timeline[-1]

    assert last_state["sets_a"] == 3
    assert last_state["is_finished"] is True
    assert last_state["winner"] == "player_a"


def test_timeline_stops_after_match_finish():

    winner_sequence = (
        ["player_a"] * 11 +
        ["player_a"] * 11 +
        ["player_a"] * 11 +
        ["player_b"] * 50   # extra points that should never apply
    )

    timeline = build_match_timeline(best_of=5, winner_sequence=winner_sequence)

    # Should stop at 33 rallies
    assert len(timeline) == 33

    last_state = timeline[-1]

    assert last_state["is_finished"] is True
    assert last_state["winner"] == "player_a"


def test_timeline_set_progression():

    winner_sequence = ["player_a"] * 11 + ["player_b"] * 11

    timeline = build_match_timeline(best_of=5, winner_sequence=winner_sequence)

    # First set
    first_set_end = timeline[10]
    assert first_set_end["sets_a"] == 1
    assert first_set_end["set_number"] == 1

    # Second set
    second_set_end = timeline[-1]
    assert second_set_end["sets_b"] == 1
    assert second_set_end["set_number"] == 2