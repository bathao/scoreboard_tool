import pytest

from backend.match_session import MatchSession


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def make_events(sequence):
    """
    sequence = ["a", "b", "a", ...]
    """
    return [
        {"timestamp": i + 1, "winner": f"player_{w}"}
        for i, w in enumerate(sequence)
    ]


# ---------------------------------------------------------
# Validation branches
# ---------------------------------------------------------

def test_events_must_be_list():
    session = MatchSession(best_of=3)

    with pytest.raises(ValueError):
        session.load_events("not_a_list")


def test_invalid_event_format_missing_key():
    session = MatchSession(best_of=3)

    events = [{"timestamp": 1}]  # missing winner

    with pytest.raises(ValueError):
        session.load_events(events)


def test_invalid_winner_atomic():
    session = MatchSession(best_of=3)

    events = [
        {"timestamp": 1, "winner": "player_a"},
        {"timestamp": 2, "winner": "wrong"},
    ]

    with pytest.raises(ValueError):
        session.load_events(events)

    assert session.get_timeline() == []


# ---------------------------------------------------------
# Empty snapshot branch
# ---------------------------------------------------------

def test_get_snapshot_when_empty():
    session = MatchSession(best_of=3)

    # no crash, timeline empty
    timeline = session.get_timeline()
    assert timeline == []


# ---------------------------------------------------------
# BO3
# ---------------------------------------------------------

def test_bo3_match_completion():
    session = MatchSession(best_of=3)

    # player_a wins 2 sets
    events = make_events(["a"] * 22)  # enough to win 2 sets

    timeline = session.load_events(events)

    final = timeline[-1]

    assert final.sets_a == 2


# ---------------------------------------------------------
# BO5
# ---------------------------------------------------------

def test_bo5_full_match():
    session = MatchSession(best_of=5)

    # player_a wins 3 sets
    events = make_events(["a"] * 33)

    timeline = session.load_events(events)

    final = timeline[-1]

    assert final.sets_a == 3


# ---------------------------------------------------------
# BO7
# ---------------------------------------------------------

def test_bo7_full_match():
    session = MatchSession(best_of=7)

    # player_b wins 4 sets
    events = make_events(["b"] * 44)

    timeline = session.load_events(events)

    final = timeline[-1]

    assert final.sets_b == 4


# ---------------------------------------------------------
# Deterministic replay
# ---------------------------------------------------------

def test_replay_deterministic():
    events = make_events(["a", "b", "a", "a", "b"])

    s1 = MatchSession(best_of=3)
    s2 = MatchSession(best_of=3)

    t1 = s1.load_events(events)
    t2 = s2.load_events(events)

    assert t1[-1].score_a == t2[-1].score_a
    assert t1[-1].score_b == t2[-1].score_b


# ---------------------------------------------------------
# Reset behavior
# ---------------------------------------------------------

def test_reset_clears_state():
    session = MatchSession(best_of=3)

    session.load_events(make_events(["a", "a", "b"]))

    session.reset()

    assert session.get_timeline() == []

    # load again after reset
    timeline = session.load_events(make_events(["b", "b"]))

    assert timeline[-1].score_b == 2


# ---------------------------------------------------------
# Export events
# ---------------------------------------------------------

def test_export_events_roundtrip():
    session = MatchSession(best_of=3)

    events = make_events(["a", "b", "a"])
    session.load_events(events)

    exported = session.export_events()

    assert exported == events

def test_get_snapshot_raises_when_empty():
    session = MatchSession(best_of=3)

    with pytest.raises(RuntimeError):
        session.get_snapshot()