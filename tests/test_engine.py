import pytest

from backend.engine import ScoreEngine
from backend.models import MatchState, RallyEvent


def create_engine(best_of=5):
    return ScoreEngine(MatchState(best_of=best_of))


# -------------------------------------------------
# Basic Rally Scoring
# -------------------------------------------------

def test_single_rally_player_a():
    engine = create_engine()

    snapshot = engine.process_event(
        RallyEvent(timestamp=0.0, winner="player_a")
    )

    assert snapshot.score_a == 1
    assert snapshot.score_b == 0
    assert not snapshot.is_finished


def test_single_rally_player_b():
    engine = create_engine()

    snapshot = engine.process_event(
        RallyEvent(timestamp=0.0, winner="player_b")
    )

    assert snapshot.score_a == 0
    assert snapshot.score_b == 1
    assert not snapshot.is_finished


# -------------------------------------------------
# Set Win Logic
# -------------------------------------------------

def test_player_a_wins_one_set():
    engine = create_engine()

    snapshot = None
    for i in range(11):
        snapshot = engine.process_event(
            RallyEvent(timestamp=i, winner="player_a")
        )

    assert snapshot.sets_a == 1
    assert snapshot.score_a == 0
    assert snapshot.score_b == 0


def test_set_requires_two_point_difference():
    engine = create_engine()

    for i in range(10):
        engine.process_event(RallyEvent(timestamp=i, winner="player_a"))
        engine.process_event(RallyEvent(timestamp=i+0.1, winner="player_b"))

    snapshot = engine.process_event(
        RallyEvent(timestamp=100, winner="player_a")
    )
    assert snapshot.sets_a == 0

    snapshot = engine.process_event(
        RallyEvent(timestamp=101, winner="player_a")
    )

    assert snapshot.sets_a == 1


# -------------------------------------------------
# Match Finish Logic
# -------------------------------------------------

def test_best_of_5_match_finish():
    engine = create_engine(best_of=5)

    snapshot = None

    for s in range(3):
        for i in range(11):
            snapshot = engine.process_event(
                RallyEvent(timestamp=s*100 + i, winner="player_a")
            )

    assert snapshot.is_finished
    assert snapshot.winner == "player_a"


def test_no_score_after_match_finished():
    engine = create_engine(best_of=3)

    for s in range(2):
        for i in range(11):
            snapshot = engine.process_event(
                RallyEvent(timestamp=s*100 + i, winner="player_a")
            )

    assert snapshot.is_finished

    snapshot_after = engine.process_event(
        RallyEvent(timestamp=999, winner="player_b")
    )

    assert snapshot_after.sets_a == 2
    assert snapshot_after.winner == "player_a"


# -------------------------------------------------
# Invalid Winner
# -------------------------------------------------

def test_invalid_winner_value():
    engine = create_engine()

    with pytest.raises(ValueError):
        engine.process_event(
            RallyEvent(timestamp=0, winner="invalid_player")
        )