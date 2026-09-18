@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" windows_service.py install
if errorlevel 1 (
    echo Ejecuta este archivo como administrador. Revisa el error anterior.
) else (
    echo Consulta http://127.0.0.1:8001/health para comprobar que la IA esta lista.
)
pause
