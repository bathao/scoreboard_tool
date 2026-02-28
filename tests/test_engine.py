from backend.models import MatchState
from backend.engine import ScoreEngine
from backend.exceptions import MatchFinishedError, InvalidOperationError
import pytest


def create_engine(best_of=5):
    match = MatchState(schema_version=1, best_of=best_of)
    return ScoreEngine(match)


def win_set(engine, player):
    for _ in range(11):
        engine.add_point(player)


# ---------- MATCH OUTCOMES BO5 ----------

@pytest.mark.parametrize("sequence, expected", [
    (["A","A","A"], "player_a"),
    (["A","A","B","A"], "player_a"),
    (["A","B","A","B","A"], "player_a"),
    (["B","B","B"], "player_b"),
    (["A","B","B","B"], "player_b"),
    (["A","B","A","B","B"], "player_b"),
])
def test_bo5_outcomes(sequence, expected):
    engine = create_engine(5)

    for winner in sequence:
        win_set(engine, "player_a" if winner == "A" else "player_b")

    assert engine.match.winner == expected
    assert engine.match.is_finished


# ---------- BO3 ----------

def test_bo3():
    engine = create_engine(3)

    win_set(engine, "player_a")
    win_set(engine, "player_a")

    assert engine.match.winner == "player_a"
    assert engine.match.is_finished


# ---------- BO7 ----------

def test_bo7():
    engine = create_engine(7)

    for _ in range(4):
        win_set(engine, "player_a")

    assert engine.match.winner == "player_a"
    assert engine.match.is_finished


# ---------- DEUCE ----------

def test_deuce_25_23():
    engine = create_engine()

    for _ in range(23):
        engine.add_point("player_a")
        engine.add_point("player_b")

    engine.add_point("player_a")
    engine.add_point("player_a")

    s = engine.match.sets[-1]

    assert s.player_a == 25
    assert s.player_b == 23
    assert s.winner == "player_a"


def test_not_finish_11_10():
    engine = create_engine()

    for _ in range(10):
        engine.add_point("player_a")
        engine.add_point("player_b")

    engine.add_point("player_a")  # 11-10

    s = engine.match.sets[-1]
    assert s.winner is None


# ---------- LOCK ----------

def test_lock_after_finish():
    engine = create_engine()

    for _ in range(3):
        win_set(engine, "player_a")

    with pytest.raises(MatchFinishedError):
        engine.add_point("player_b")


# ---------- INVALID PLAYER ----------

def test_invalid_player():
    engine = create_engine()

    with pytest.raises(InvalidOperationError):
        engine.add_point("player_c")