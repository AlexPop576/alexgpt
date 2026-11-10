@echo off
cd /d "%~dp0"
start "" http://localhost:8000
.venv\Scripts\python.exe chat_server.py
pause
