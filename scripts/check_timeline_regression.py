from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scripts.check_endpoint_regression import *  # noqa: F401,F403
from scripts.check_endpoint_regression import main


if __name__ == "__main__":
    raise SystemExit(main())
