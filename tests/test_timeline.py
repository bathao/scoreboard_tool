import pytest

from backend.timeline import build_match_timeline
from backend.models import RallyEvent


# -------------------------------------------------
# Basic Timeline Build
# -------------------------------------------------

def test_timeline_basic_build():
    events = [
        RallyEvent(timestamp=i, winner="player_a")
        for i in range(5)
    ]

    timeline = build_match_timeline(best_of=5, events=events)

    assert len(timeline) == 5
    assert timeline[-1].score_a == 5


# -------------------------------------------------
# Timeline Stops After Match Finish
# -------------------------------------------------

def test_timeline_stops_after_match_finish():
    events = []

    # enough to win match 3 sets
    for s in range(3):
        for i in range(11):
            events.append(
                RallyEvent(timestamp=s*100+i, winner="player_a")
            )

    # add extra noise events
    for i in range(20):
        events.append(
            RallyEvent(timestamp=1000+i, winner="player_b")
        )

    timeline = build_match_timeline(best_of=5, events=events)

    last = timeline[-1]

    assert last.is_finished
    assert last.winner == "player_a"


# -------------------------------------------------
# Empty Events
# -------------------------------------------------

def test_empty_timeline():
    timeline = build_match_timeline(best_of=5, events=[])

    assert timeline == []


# -------------------------------------------------
# Mixed Winners
# -------------------------------------------------

def test_mixed_winner_progression():
    events = [
        RallyEvent(timestamp=0, winner="player_a"),
        RallyEvent(timestamp=1, winner="player_b"),
        RallyEvent(timestamp=2, winner="player_a"),
    ]

    timeline = build_match_timeline(best_of=5, events=events)

    assert timeline[0].score_a == 1
    assert timeline[1].score_b == 1
    assert timeline[2].score_a == 2


# -------------------------------------------------
# Timestamp Gaps
# -------------------------------------------------

def test_large_timestamp_gap():
    events = [
        RallyEvent(timestamp=0, winner="player_a"),
        RallyEvent(timestamp=9999, winner="player_a"),
    ]

    timeline = build_match_timeline(best_of=5, events=events)

    assert len(timeline) == 2
    assert timeline[-1].score_a == 2


# -------------------------------------------------
# Invalid Event Data
# -------------------------------------------------

def test_invalid_event_in_timeline():
    events = [
        RallyEvent(timestamp=0, winner="player_a"),
        RallyEvent(timestamp=1, winner="invalid"),
    ]

    with pytest.raises(ValueError):
        build_match_timeline(best_of=5, events=events)


# -------------------------------------------------
# Best Of Variations
# -------------------------------------------------

@pytest.mark.parametrize("best_of, required_sets", [
    (3, 2),
    (5, 3),
    (7, 4),
])
def test_best_of_variations(best_of, required_sets):

    events = []

    for s in range(required_sets):
        for i in range(11):
            events.append(
                RallyEvent(timestamp=s*100+i, winner="player_b")
            )

    timeline = build_match_timeline(best_of=best_of, events=events)

    last = timeline[-1]

    assert last.is_finished
    assert last.winner == "player_b"