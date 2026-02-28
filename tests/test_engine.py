from backend.models import MatchState
from backend.engine import ScoreEngine
from backend.config import SCHEMA_VERSION, DEFAULT_BEST_OF
from backend.exceptions import MatchFinishedError
import pytest


def create_engine():
    match = MatchState(
        schema_version=SCHEMA_VERSION,
        best_of=DEFAULT_BEST_OF
    )
    return ScoreEngine(match)


def win_set(engine, player, score=11):
    for _ in range(score):
        engine.add_point(player)


def deuce_win(engine, winner, final_a, final_b):
    for _ in range(min(final_a, final_b)):
        engine.add_point("player_a")
        engine.add_point("player_b")

    while True:
        current = engine.match.sets[-1]
        if current.winner:
            break
        engine.add_point(winner)


# ---------- MATCH RESULTS ----------

@pytest.mark.parametrize("sequence, expected_winner", [
    (["A","A","A"], "player_a"),      # 3-0
    (["A","A","B","A"], "player_a"),  # 3-1
    (["A","B","A","B","A"], "player_a"),  # 3-2
    (["B","B","B"], "player_b"),      # 0-3
    (["A","B","B","B"], "player_b"),  # 1-3
    (["A","B","A","B","B"], "player_b"),  # 2-3
])
def test_match_outcomes(sequence, expected_winner):
    engine = create_engine()

    for winner in sequence:
        win_set(engine, "player_a" if winner == "A" else "player_b")

    assert engine.match.winner == expected_winner
    assert engine.match.is_finished is True


# ---------- DEUCE ----------

def test_deuce_17_15():
    engine = create_engine()

    deuce_win(engine, "player_a", 17, 15)

    current = engine.match.sets[-1]
    assert current.player_a == 17
    assert current.player_b == 15
    assert current.winner == "player_a"
    assert engine.match.is_finished is False


# ---------- LOCK AFTER FINISH ----------

def test_lock_after_finish():
    engine = create_engine()

    for _ in range(3):
        win_set(engine, "player_a")

    assert engine.match.is_finished is True

    with pytest.raises(MatchFinishedError):
        engine.add_point("player_b")


# ---------- NO 6TH SET ----------

def test_no_sixth_set():
    engine = create_engine()

    for _ in range(3):
        win_set(engine, "player_a")

    with pytest.raises(MatchFinishedError):
        engine.add_point("player_b")