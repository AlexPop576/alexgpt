@echo off
rem On Alex's PC: uses the project environment (.venv in the LLM folder, with the GPU version of PyTorch).
cd /d "%~dp0"
start "" http://localhost:8000
"..\.venv\Scripts\python.exe" chat_server.py %*
pause
