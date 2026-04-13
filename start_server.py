import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
python = sys.executable
script = str(ROOT / "scripts" / "run_local_web_ui.py")

subprocess.Popen(
    ["cmd", "/k", f'"{python}" "{script}"'],
    creationflags=subprocess.CREATE_NEW_CONSOLE,
)
print("Server starting in a new terminal window at http://127.0.0.1:8765")
