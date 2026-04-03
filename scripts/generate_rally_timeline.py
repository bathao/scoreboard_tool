from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scripts.generate_draft_multistream import *  # noqa: F401,F403
from scripts.generate_draft_multistream import build_draft as build_rally_timeline
from scripts.generate_draft_multistream import main


if __name__ == "__main__":
    raise SystemExit(main())
