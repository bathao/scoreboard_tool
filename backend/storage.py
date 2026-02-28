import json
from pathlib import Path
from backend.models import MatchState, SetScore

def load_match(path: Path) -> MatchState:
    with open(path, "r") as f:
        data = json.load(f)

    sets = [SetScore(**s) for s in data["sets"]]

    return MatchState(
        schema_version=data["schema_version"],
        best_of=data["best_of"],
        sets=sets,
        winner=data["winner"],
        is_finished=data["is_finished"]
    )

def save_match(path: Path, match: MatchState):
    with open(path, "w") as f:
        json.dump({
            "schema_version": match.schema_version,
            "best_of": match.best_of,
            "sets": [s.__dict__ for s in match.sets],
            "winner": match.winner,
            "is_finished": match.is_finished
        }, f, indent=4)