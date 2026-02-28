from backend.models import MatchState, SetScore, RallyEvent, MatchSnapshot


class ScoreEngine:
    """
    Production-grade score engine.

    Responsibilities:
    - Process RallyEvent
    - Enforce domain invariants
    - Handle set & match lifecycle
    - Produce consistent MatchSnapshot
    - Remain deterministic & extensible
    """

    def __init__(self, match: MatchState):
        self.match = match
        self._validate_initial_state()
        self._last_timestamp = None

    # =========================================================
    # PUBLIC API
    # =========================================================

    def process_event(self, event: RallyEvent) -> MatchSnapshot:
        """
        Process rally event and return snapshot.
        """

        self._validate_event(event)

        # Enforce monotonic timestamp
        if self._last_timestamp is not None:
            if event.timestamp < self._last_timestamp:
                raise ValueError("Event timestamp must be non-decreasing")

        self._last_timestamp = event.timestamp

        # If match finished → do not mutate state
        if self.match.is_finished:
            return self._build_snapshot(event.timestamp)

        current_set = self.match.sets[-1]

        # Apply score
        if event.winner == "player_a":
            current_set.player_a += 1
        else:
            current_set.player_b += 1

        # Check set lifecycle
        if self._is_set_won(current_set):
            self._finalize_set(current_set)

        # Check match lifecycle
        self._update_match_status()

        return self._build_snapshot(event.timestamp)

    # =========================================================
    # VALIDATION
    # =========================================================

    def _validate_initial_state(self):
        if self.match.best_of <= 0:
            raise ValueError("best_of must be positive")

        if self.match.best_of % 2 == 0:
            raise ValueError("best_of must be odd")

        if not self.match.sets:
            self.match.sets.append(SetScore())

    def _validate_event(self, event: RallyEvent):
        if event.winner not in ("player_a", "player_b"):
            raise ValueError(f"Invalid winner: {event.winner}")

    # =========================================================
    # SET LOGIC
    # =========================================================

    def _is_set_won(self, set_score: SetScore) -> bool:
        a = set_score.player_a
        b = set_score.player_b

        return (a >= 11 or b >= 11) and abs(a - b) >= 2

    def _finalize_set(self, set_score: SetScore):
        if set_score.player_a > set_score.player_b:
            set_score.winner = "player_a"
        else:
            set_score.winner = "player_b"

        # Prepare next set if match not yet finished
        required_sets = self._required_sets_to_win()

        a_sets, b_sets = self._count_sets()

        if a_sets < required_sets and b_sets < required_sets:
            self.match.sets.append(SetScore())

    # =========================================================
    # MATCH LOGIC
    # =========================================================

    def _required_sets_to_win(self) -> int:
        return (self.match.best_of // 2) + 1

    def _count_sets(self):
        a_sets = sum(1 for s in self.match.sets if s.winner == "player_a")
        b_sets = sum(1 for s in self.match.sets if s.winner == "player_b")
        return a_sets, b_sets

    def _update_match_status(self):
        required = self._required_sets_to_win()
        a_sets, b_sets = self._count_sets()

        if a_sets >= required:
            self.match.is_finished = True
            self.match.winner = "player_a"
        elif b_sets >= required:
            self.match.is_finished = True
            self.match.winner = "player_b"
        else:
            self.match.is_finished = False
            self.match.winner = None

    # =========================================================
    # SNAPSHOT
    # =========================================================

    def _build_snapshot(self, timestamp: float) -> MatchSnapshot:
        current_set = self.match.sets[-1]
        a_sets, b_sets = self._count_sets()

        return MatchSnapshot(
            timestamp=timestamp,
            set_number=len(self.match.sets),
            score_a=current_set.player_a,
            score_b=current_set.player_b,
            sets_a=a_sets,
            sets_b=b_sets,
            is_finished=self.match.is_finished,
            winner=self.match.winner,
        )