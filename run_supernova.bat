@echo off
rem ================================================================
rem  AlexGPT Supernova: everything from zero to training, by itself.
rem  Double-click this file. You can close the window or press Ctrl+C
rem  any time; double-click it again and it continues where it was.
rem ================================================================
cd /d F:\Code\PersonalProjects\LLM
call .venv\Scripts\activate.bat

echo.
echo [1/4] installing the streaming library ...
pip install --quiet datasets
if errorlevel 1 goto failed

echo.
echo [2/4] tokenizer (skipped if it already exists) ...
if not exist checkpoints\supernova_tokenizer.json (
    python prepare_tokenizer_supernova.py
    if errorlevel 1 goto failed
)

echo.
echo [3/4] preparing the data (continues where it stopped) ...
python prepare_supernova.py
if errorlevel 1 goto failed

echo.
echo [4/4] training Supernova (Ctrl+C once = save and stop) ...
python train_supernova.py
if errorlevel 1 goto failed

echo.
echo === all done ===
pause
exit /b 0

:failed
echo.
echo === something went wrong: the error is above. Send it to Claude. ===
pause
exit /b 1
