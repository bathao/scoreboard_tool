from typing import List, Dict
from copy import deepcopy

from backend.engine import ScoreEngine
from backend.models import MatchState, RallyEvent, MatchSnapshot


class MatchSession:
    """
    Single local match session.

    Responsibilities:
    - Manage one ScoreEngine instance
    - Bulk replay rally events (atomic)
    - Store timeline snapshots
    - Export original rally events
    """

    def __init__(self, best_of: int):
        self._best_of = best_of
        self._engine = ScoreEngine(MatchState(best_of=best_of))
        self._timeline: List[MatchSnapshot] = []
        self._events: List[RallyEvent] = []

    # ---------------------------------------------------------
    # Core API
    # ---------------------------------------------------------

    def load_events(self, events: List[Dict]) -> List[MatchSnapshot]:
        """
        Bulk load rally events from list of dicts.
        Atomic: if any event fails -> no state mutation.
        """
        if not isinstance(events, list):
            raise ValueError("events must be a list")

        # Convert first (validation stage)
        rally_events = []
        for e in events:
            if "timestamp" not in e or "winner" not in e:
                raise ValueError("invalid event format")

            rally_events.append(
                RallyEvent(
                    timestamp=float(e["timestamp"]),
                    winner=e["winner"]
                )
            )

        # Sort by timestamp (important for AI pipeline safety)
        rally_events.sort(key=lambda x: x.timestamp)

        # Prepare temp engine for atomic replay
        temp_engine = ScoreEngine(MatchState(best_of=self._best_of))
        temp_timeline: List[MatchSnapshot] = []

        for event in rally_events:
            snapshot = temp_engine.process_event(event)
            temp_timeline.append(snapshot)

        # If everything succeeds → commit
        self._engine = temp_engine
        self._timeline = temp_timeline
        self._events = rally_events

        return deepcopy(self._timeline)

    def get_snapshot(self) -> MatchSnapshot:
        if not self._timeline:
            raise RuntimeError("No events loaded")

        return self._timeline[-1]

    def get_timeline(self) -> List[MatchSnapshot]:
        return deepcopy(self._timeline)

    def export_events(self) -> List[Dict]:
        return [
            {"timestamp": e.timestamp, "winner": e.winner}
            for e in self._events
        ]

    def reset(self):
        self._engine = ScoreEngine(MatchState(best_of=self._best_of))
        self._timeline = []
        self._events = []