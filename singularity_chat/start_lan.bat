@echo off
rem Like start.bat, but phones/laptops on the same Wi-Fi can open it too: http://IP-ul-acestui-PC:8000
cd /d "%~dp0"
ipconfig | findstr IPv4
.venv\Scripts\python.exe chat_server.py --lan
pause
