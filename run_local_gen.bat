@echo off
rem Everything the local model (EuroLLM on your GPU) generates. Safe to stop (Ctrl+C) and run again: it resumes.
rem Usage: run_local_gen.bat [model]      (default: EuroLLM-9B)
cd /d %~dp0
ollama list >nul 2>nul || (start "ollama server" /min ollama serve & timeout /t 8 >nul)
set MODEL=%~1
if "%MODEL%"=="" set MODEL=hf.co/lmstudio-community/EuroLLM-9B-Instruct-GGUF:Q4_K_M

echo === A. Wikipedia articles (cached ones are skipped) ===
python gen_data_wiki.py
echo === B. Fact-safe conversations from real Wikipedia text (summary / questions / "the text doesn't say") ===
python gen_grounded.py --model %MODEL%
echo === C. Casual chat, short summaries, markdown from user-given content, honest "I don't know" ===
python gen_local.py --model %MODEL% --lang ro --kind chat --count 250
python gen_local.py --model %MODEL% --lang en --kind chat --count 150
python gen_local.py --model %MODEL% --lang ro --kind markdown --count 100
python gen_local.py --model %MODEL% --lang en --kind markdown --count 100
python gen_local.py --model %MODEL% --lang ro --kind summary --count 100
python gen_local.py --model %MODEL% --lang en --kind summary --count 50
python gen_local.py --model %MODEL% --lang ro --kind idk --count 50
python gen_local.py --model %MODEL% --lang en --kind idk --count 50
echo === D. Check everything ===
python check_gen_data.py
echo ALL DONE - tell Claude "gata".
pause
