from __future__ import annotations

import argparse
import sys
from pathlib import Path
from wsgiref.simple_server import make_server

sys.path.append(str(Path(__file__).parent.parent))

from backend.local_web_ui import ThreadingWSGIServer, create_local_web_app
from backend.production_pipeline import ProductionPipelineConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local scoreboard-tool web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    app = create_local_web_app(config=ProductionPipelineConfig())
    with make_server(args.host, args.port, app, server_class=ThreadingWSGIServer) as server:
        print(f"[OK] Local UI running at http://{args.host}:{args.port}")
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
