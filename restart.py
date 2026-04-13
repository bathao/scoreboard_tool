"""Restart the local scoreboard-tool server.

Kills any existing process on port 8765, then opens a new console window
with the server so logs are visible.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
python = sys.executable
script = str(ROOT / "scripts" / "run_local_web_ui.py")

# Kill existing server on port 8765 (Python subprocess — avoids bash/cmd path issues)
result = subprocess.run(
    "netstat -ano | findstr :8765 | findstr LISTENING",
    shell=True, capture_output=True, text=True,
)
killed = False
for line in result.stdout.strip().splitlines():
    parts = line.split()
    if parts:
        pid = parts[-1]
        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
        print(f"Killed old server (PID {pid})")
        killed = True

if not killed:
    print("No existing server found on port 8765")

time.sleep(1)

# Start new server in a fresh independent console window
# cmd /k keeps the window open even if the server crashes (shows error output)
subprocess.Popen(
    ["cmd", "/k", f'"{python}" "{script}"'],
    creationflags=subprocess.CREATE_NEW_CONSOLE,
)
print("Server starting at http://127.0.0.1:8765")
