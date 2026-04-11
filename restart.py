import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
python = sys.executable
script = str(ROOT / "scripts" / "run_local_web_ui.py")

bat = ROOT / "_restart_tmp.bat"
bat.write_text(
    "@echo off\n"
    "ping -n 3 127.0.0.1 >nul\n"
    "taskkill /F /IM python.exe /T >nul 2>&1\n"
    "ping -n 2 127.0.0.1 >nul\n"
    f"start \"Scoreboard\" \"{python}\" \"{script}\"\n"
    f"del \"{bat}\"\n"
)

subprocess.Popen(
    ["cmd", "/c", str(bat)],
    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
    close_fds=True,
)
print("Restarting in ~5s... Server will be at http://127.0.0.1:8765")
