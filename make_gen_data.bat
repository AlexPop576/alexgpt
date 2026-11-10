@echo off
rem Builds all code-made SFT data. Run from the project folder:  make_gen_data.bat
rem 1) offline (no internet, no AI) 2) Wikipedia (internet) 3) check
cd /d %~dp0
python gen_data_tools.py || goto :error
python gen_data_markdown.py || goto :error
python gen_data_idk.py || goto :error
python gen_data_wiki.py || goto :error
python check_gen_data.py
pause
exit /b 0
:error
echo FAILED - read the message above
pause
