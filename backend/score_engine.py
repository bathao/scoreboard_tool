class MatchValidationError(Exception):
    pass


class MatchAlreadyFinishedError(MatchValidationError):
    pass


class InvalidWinnerCodeError(MatchValidationError):
    pass


class MatchNotFinishedError(MatchValidationError):
    pass


REQUIRED_FIELDS = {
    "schema_version",
    "match_id",
    "best_of",
    "players",
    "points"
}


def validate_schema(match_data: dict):
    missing = REQUIRED_FIELDS - set(match_data.keys())
    if missing:
        raise MatchValidationError(f"Missing field(s): {missing}")

    if match_data["schema_version"] != "1.0":
        raise MatchValidationError("Unsupported schema_version")

    if match_data["best_of"] not in [3, 5, 7]:
        raise MatchValidationError("best_of must be 3, 5, or 7")

    if set(match_data["players"].keys()) != {"A", "B"}:
        raise MatchValidationError("players must contain exactly keys A and B")

    if not isinstance(match_data["points"], list):
        raise MatchValidationError("points must be list")

    for p in match_data["points"]:
        if p not in ["A", "B"]:
            raise InvalidWinnerCodeError(f"Invalid point value: {p}")


class MatchEngine:

    def __init__(self, match_data: dict):
        validate_schema(match_data)

        self.best_of = match_data["best_of"]
        self.points = match_data["points"]

        self.sets = []
        self.match_winner = None

    def process(self):
        sets_to_win = self.best_of // 2 + 1

        current_a = 0
        current_b = 0
        sets_a = 0
        sets_b = 0

        for idx, p in enumerate(self.points):

            if self.match_winner:
                raise MatchAlreadyFinishedError(
                    f"Point added after match finished at index {idx}"
                )

            if p == "A":
                current_a += 1
            elif p == "B":
                current_b += 1
            else:
                raise InvalidWinnerCodeError(f"Invalid point: {p}")

            if self._is_set_finished(current_a, current_b):
                winner = "A" if current_a > current_b else "B"

                self.sets.append({
                    "A": current_a,
                    "B": current_b,
                    "winner": winner
                })

                if winner == "A":
                    sets_a += 1
                else:
                    sets_b += 1

                current_a = 0
                current_b = 0

                if sets_a == sets_to_win:
                    self.match_winner = "A"
                elif sets_b == sets_to_win:
                    self.match_winner = "B"

        if not self.match_winner:
            raise MatchNotFinishedError("Match did not finish properly")

        return {
            "match_winner": self.match_winner,
            "sets": self.sets
        }

    @staticmethod
    def _is_set_finished(a, b):
        if (a >= 11 or b >= 11) and abs(a - b) >= 2:
            return True
        return False