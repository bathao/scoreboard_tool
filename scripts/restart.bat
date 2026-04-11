@echo off
ping -n 2 127.0.0.1 >nul
taskkill /F /IM python.exe /T >nul 2>&1
ping -n 3 127.0.0.1 >nul
cd /d "%~dp0.."
python scripts\run_local_web_ui.py
