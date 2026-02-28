from backend.models import MatchState
from backend.engine import ScoreEngine


def build_match_timeline(best_of: int, winner_sequence: list[str]) -> list[dict]:
    """
    Replays a match from scratch using winner_sequence.
    Returns a flattened timeline after each rally.
    Does NOT mutate external state.
    """

    match = MatchState(
        schema_version=1,
        best_of=best_of,
        sets=[]
    )

    engine = ScoreEngine(match)

    timeline: list[dict] = []

    for index, winner in enumerate(winner_sequence):

        engine.add_point(winner)

        current_set = match.sets[-1]

        sets_a = sum(1 for s in match.sets if s.winner == "player_a")
        sets_b = sum(1 for s in match.sets if s.winner == "player_b")

        snapshot = {
            "rally_index": index + 1,
            "set_number": len(match.sets),
            "score_a": current_set.player_a,
            "score_b": current_set.player_b,
            "sets_a": sets_a,
            "sets_b": sets_b,
            "is_finished": match.is_finished,
            "winner": match.winner
        }

        timeline.append(snapshot)

        if match.is_finished:
            break

    return timeline