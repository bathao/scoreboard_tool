from backend.models import MatchState, SetScore
from backend.exceptions import (
    MatchFinishedError,
    MaximumSetsReachedError,
    InvalidOperationError
)


class ScoreEngine:

    def __init__(self, match: MatchState):
        self.match = match

    def add_point(self, player: str):

        if self.match.is_finished:
            raise MatchFinishedError("Match already finished")

        if player not in ("player_a", "player_b"):
            raise InvalidOperationError("Invalid player")

        current_set = self._get_or_create_current_set()

        if player == "player_a":
            current_set.player_a += 1
        else:
            current_set.player_b += 1

        self._check_set_winner(current_set)
        self._recalculate_match_state()

    def _get_or_create_current_set(self):

        if not self.match.sets:
            self.match.sets.append(SetScore())

        current = self.match.sets[-1]

        if current.winner:
            if len(self.match.sets) >= self.match.best_of:
                raise MaximumSetsReachedError("Maximum sets reached")
            new_set = SetScore()
            self.match.sets.append(new_set)
            return new_set

        return current

    def _check_set_winner(self, set_score: SetScore):
        a = set_score.player_a
        b = set_score.player_b

        if (a >= 11 or b >= 11) and abs(a - b) >= 2:
            set_score.winner = "player_a" if a > b else "player_b"

    def _recalculate_match_state(self):

        sets_to_win = (self.match.best_of // 2) + 1

        a_sets = sum(1 for s in self.match.sets if s.winner == "player_a")
        b_sets = sum(1 for s in self.match.sets if s.winner == "player_b")

        if a_sets >= sets_to_win:
            self.match.winner = "player_a"
            self.match.is_finished = True
        elif b_sets >= sets_to_win:
            self.match.winner = "player_b"
            self.match.is_finished = True
        else:
            self.match.winner = None
            self.match.is_finished = False