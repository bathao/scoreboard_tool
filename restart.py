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

# Kill existing server on port 8765
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

# Write a tiny bat so cmd /k can reference a file path instead of a
# command string — avoids the cmd.exe quote-escaping issue with subprocess.
bat = ROOT / "_server.bat"
bat.write_text(
    f'@echo off\n'
    f'"{python}" "{script}"\n',
    encoding="utf-8",
)

# cmd /k keeps the window open after the server exits (shows errors).
# CREATE_NEW_CONSOLE creates an independent window not tied to this process.
subprocess.Popen(
    ["cmd", "/k", str(bat)],
    creationflags=subprocess.CREATE_NEW_CONSOLE,
)
print("Server starting at http://127.0.0.1:8765")
