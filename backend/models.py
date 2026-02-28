from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SetScore:
    player_a: int = 0
    player_b: int = 0
    winner: Optional[str] = None


@dataclass
class MatchState:
    best_of: int
    sets: List[SetScore] = field(default_factory=lambda: [SetScore()])
    winner: Optional[str] = None
    is_finished: bool = False


# --- NEW DOMAIN TYPES ---

@dataclass
class RallyEvent:
    winner: str
    timestamp: float


@dataclass
class MatchSnapshot:
    timestamp: float
    set_number: int
    score_a: int
    score_b: int
    sets_a: int
    sets_b: int
    is_finished: bool
    winner: Optional[str]