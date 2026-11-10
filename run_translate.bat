@echo off
rem Usage: run_translate.bat sft_data\ext_en_C.txt   (Ollama must be running: run_eurollm.bat did that)
cd /d %~dp0
ollama list >nul 2>nul || (start "ollama server" /min ollama serve & timeout /t 8 >nul)
python translate_local.py %1
pause
