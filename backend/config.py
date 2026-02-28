from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATCHES_DIR = PROJECT_ROOT / "matches"

SCHEMA_VERSION = 1
DEFAULT_BEST_OF = 5
SETS_TO_WIN = 3