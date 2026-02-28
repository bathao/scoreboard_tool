from backend.models import MatchState, RallyEvent
from backend.engine import ScoreEngine


def build_match_timeline(best_of: int, events: list[RallyEvent]):

    match = MatchState(best_of=best_of)
    engine = ScoreEngine(match)

    timeline = []

    for event in events:

        snapshot = engine.process_event(event)

        if snapshot is None:
            break

        timeline.append(snapshot)

    return timeline