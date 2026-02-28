from backend.models import MatchState
from backend.engine import ScoreEngine
from backend.config import SCHEMA_VERSION, DEFAULT_BEST_OF

match = MatchState(
    schema_version=SCHEMA_VERSION,
    best_of=DEFAULT_BEST_OF
)

engine = ScoreEngine(match)

# Set 1: A
for _ in range(11):
    engine.add_point("player_a")

# Set 2: B
for _ in range(11):
    engine.add_point("player_b")

# Set 3: A
for _ in range(11):
    engine.add_point("player_a")

# Set 4: B
for _ in range(11):
    engine.add_point("player_b")

# Set 5: deuce dài 17-15
for _ in range(15):
    engine.add_point("player_a")
    engine.add_point("player_b")

engine.add_point("player_a")  # 16-15
engine.add_point("player_a")  # 17-15 -> match winner

print("Before illegal point:")
print(match)

print("\nTrying to add illegal point...")

engine.add_point("player_b")  # PHẢI raise error