@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" windows_service.py remove
pause
