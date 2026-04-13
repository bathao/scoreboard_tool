"""Start the local scoreboard-tool server in a new console window."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
python = sys.executable
script = str(ROOT / "scripts" / "run_local_web_ui.py")

# Write a tiny bat so cmd /k can reference a file path instead of a
# command string — avoids the cmd.exe quote-escaping issue with subprocess.
bat = ROOT / "_server.bat"
bat.write_text(
    f'@echo off\n'
    f'"{python}" "{script}"\n',
    encoding="utf-8",
)

subprocess.Popen(
    ["cmd", "/k", str(bat)],
    creationflags=subprocess.CREATE_NEW_CONSOLE,
)
print("Server starting in a new terminal window at http://127.0.0.1:8765")
