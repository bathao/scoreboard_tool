from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SetScore:
    player_a: int = 0
    player_b: int = 0
    winner: Optional[str] = None

@dataclass
class MatchState:
    schema_version: int
    best_of: int
    sets: List[SetScore] = field(default_factory=list)
    winner: Optional[str] = None
    is_finished: bool = False