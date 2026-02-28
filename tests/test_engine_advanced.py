import random
import pytest

from backend.engine import ScoreEngine
from backend.models import MatchState, RallyEvent


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def create_engine(best_of=3):
    return ScoreEngine(MatchState(best_of=best_of))


# ---------------------------------------------------------
# Timestamp Invariants
# ---------------------------------------------------------

def test_timestamp_must_be_monotonic():
    engine = create_engine()

    engine.process_event(RallyEvent(timestamp=10, winner="player_a"))

    with pytest.raises(ValueError):
        engine.process_event(RallyEvent(timestamp=5, winner="player_b"))


# ---------------------------------------------------------
# best_of Validation
# ---------------------------------------------------------

def test_best_of_must_be_odd():
    with pytest.raises(ValueError):
        create_engine(best_of=4)


def test_best_of_must_be_positive():
    with pytest.raises(ValueError):
        create_engine(best_of=0)


# ---------------------------------------------------------
# Snapshot Immutability
# ---------------------------------------------------------

def test_snapshot_is_immutable():
    engine = create_engine()

    snapshot1 = engine.process_event(
        RallyEvent(timestamp=1, winner="player_a")
    )

    engine.process_event(
        RallyEvent(timestamp=2, winner="player_a")
    )

    # Snapshot1 must remain unchanged
    assert snapshot1.score_a == 1
    assert snapshot1.score_b == 0


# ---------------------------------------------------------
# No Extra Set After Match Finish
# ---------------------------------------------------------

def test_no_extra_set_after_match_finish():
    engine = create_engine(best_of=3)

    # Player A wins 2 sets
    for s in range(2):
        for i in range(11):
            engine.process_event(
                RallyEvent(timestamp=s * 100 + i, winner="player_a")
            )

    assert engine.match.is_finished
    assert len(engine.match.sets) == 2


# ---------------------------------------------------------
# Long Deuce Scenario
# ---------------------------------------------------------

def test_long_deuce_scenario():
    engine = create_engine()

    # Reach 10-10
    for i in range(10):
        engine.process_event(RallyEvent(timestamp=i * 2, winner="player_a"))
        engine.process_event(RallyEvent(timestamp=i * 2 + 1, winner="player_b"))

    # Now 10-10, must win by 2
    engine.process_event(RallyEvent(timestamp=100, winner="player_a"))
    engine.process_event(RallyEvent(timestamp=101, winner="player_b"))

    # 11-11
    engine.process_event(RallyEvent(timestamp=102, winner="player_a"))
    snapshot = engine.process_event(RallyEvent(timestamp=103, winner="player_a"))

    assert snapshot.sets_a == 1
    assert snapshot.score_a == 0
    assert snapshot.score_b == 0


# ---------------------------------------------------------
# Deterministic Replay
# ---------------------------------------------------------

def test_replay_is_deterministic():
    events = [
        RallyEvent(timestamp=i, winner=random.choice(["player_a", "player_b"]))
        for i in range(50)
    ]

    engine1 = create_engine()
    engine2 = create_engine()

    for event in events:
        snapshot1 = engine1.process_event(event)
        snapshot2 = engine2.process_event(event)

    assert snapshot1.sets_a == snapshot2.sets_a
    assert snapshot1.sets_b == snapshot2.sets_b
    assert snapshot1.is_finished == snapshot2.is_finished
    assert snapshot1.winner == snapshot2.winner


# ---------------------------------------------------------
# Large Random Simulation Safety
# ---------------------------------------------------------

def test_large_random_simulation_always_finishes():
    engine = create_engine(best_of=5)

    timestamp = 0

    while not engine.match.is_finished:
        winner = random.choice(["player_a", "player_b"])
        engine.process_event(
            RallyEvent(timestamp=timestamp, winner=winner)
        )
        timestamp += 1

    required_sets = (engine.match.best_of // 2) + 1
    a_sets = sum(1 for s in engine.match.sets if s.winner == "player_a")
    b_sets = sum(1 for s in engine.match.sets if s.winner == "player_b")

    assert a_sets == required_sets or b_sets == required_sets


# ---------------------------------------------------------
# Idempotent After Finish
# ---------------------------------------------------------

def test_no_state_mutation_after_finish():
    engine = create_engine(best_of=3)

    # Finish match
    for s in range(2):
        for i in range(11):
            engine.process_event(
                RallyEvent(timestamp=s * 100 + i, winner="player_a")
            )

    snapshot_before = engine.process_event(
        RallyEvent(timestamp=999, winner="player_b")
    )

    snapshot_after = engine.process_event(
        RallyEvent(timestamp=1000, winner="player_b")
    )

    assert snapshot_before.sets_a == snapshot_after.sets_a
    assert snapshot_before.sets_b == snapshot_after.sets_b
    assert snapshot_before.score_a == snapshot_after.score_a
    assert snapshot_before.score_b == snapshot_after.score_b