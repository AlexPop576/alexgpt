@echo off
rem ONE script: installs Ollama if missing, stores the model on this drive (F:), pulls qwen3:8b, tests 3 conversations,
rem then (after you press a key) generates ~1,150 conversations into data\sft\local_gen.jsonl.
rem Safe to run again: it skips what is already done, and generation only appends.
setlocal
cd /d %~dp0
set MODELS_DIR=%~d0\ollama_models
set MODEL=%~1
if "%MODEL%"=="" set MODEL=qwen3:8b

echo.
echo === 1/5  Is Ollama installed? ===
where ollama >nul 2>nul
if not errorlevel 1 goto have_ollama
echo Not found. Downloading the installer...
curl -L -o "%TEMP%\OllamaSetup.exe" https://ollama.com/download/OllamaSetup.exe
if errorlevel 1 goto failed
echo Installing (silent, ~1 minute)...
"%TEMP%\OllamaSetup.exe" /SILENT
set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama"
where ollama >nul 2>nul
if errorlevel 1 goto failed
:have_ollama
echo Ollama is installed.

echo.
echo === 2/5  Where are the models stored? ===
if exist "%USERPROFILE%\.ollama\models\manifests\registry.ollama.ai\library\qwen3" goto keep_default
if defined OLLAMA_MODELS goto models_already_set
mkdir "%MODELS_DIR%" 2>nul
setx OLLAMA_MODELS "%MODELS_DIR%" >nul
set "OLLAMA_MODELS=%MODELS_DIR%"
echo Models will be stored in %MODELS_DIR%
echo Restarting Ollama so it uses that folder...
taskkill /f /im "ollama app.exe" >nul 2>nul
taskkill /f /im ollama.exe >nul 2>nul
timeout /t 3 >nul
goto start_server
:keep_default
echo qwen3 is already in the default Ollama folder (%USERPROFILE%\.ollama): using it.
goto start_server
:models_already_set
echo OLLAMA_MODELS is already set to %OLLAMA_MODELS%

:start_server
echo.
echo === 3/5  Starting the Ollama server ===
ollama list >nul 2>nul
if not errorlevel 1 goto server_ok
start "ollama server" /min ollama serve
set tries=0
:wait_server
ollama list >nul 2>nul
if not errorlevel 1 goto server_ok
set /a tries+=1
if %tries% GTR 20 goto failed
timeout /t 3 >nul
goto wait_server
:server_ok
echo Server is running.

echo.
echo === 4/5  Model %MODEL% ===
ollama list | findstr /i "%MODEL%" >nul
if not errorlevel 1 goto have_model
echo Downloading %MODEL% (~5 GB, a few minutes)...
ollama pull %MODEL%
if errorlevel 1 goto failed
:have_model
echo Model is ready.

echo.
echo === 5/5  Test: 3 Romanian conversations ===
del data\sft\local_test.jsonl 2>nul
python gen_local.py --model %MODEL% --count 3 --lang ro --kind chat --out data\sft\local_test.jsonl
if errorlevel 1 goto failed
python -c "import json; [print(json.dumps(json.loads(line)['messages'], ensure_ascii=False, indent=1)) for line in open('data/sft/local_test.jsonl', encoding='utf-8')]"
echo.
echo Quick look at the 3 conversations above. The long generation starts in 20 seconds (Ctrl+C to cancel).
timeout /t 20
call run_local_gen.bat "%MODEL%"
exit /b 0

:failed
echo.
echo FAILED - copy the messages above and send them to Claude.
pause
exit /b 1
