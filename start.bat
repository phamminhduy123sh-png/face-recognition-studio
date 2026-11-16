@echo off
setlocal
cd /d "%~dp0"
echo Face Studio - Keep this window open while using the app.
if exist ".venv\Scripts\python.exe" goto dependencies
where py >nul 2>nul
if errorlevel 1 (
  python -m venv .venv
) else (
  py -3 -m venv .venv
)
if errorlevel 1 goto failed

:dependencies
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m faceid init
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m faceid serve --open-browser
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo Startup failed. Read or copy the error above before closing this window.
echo Python 3.10-3.12 is recommended. Run start.bat from the extracted project folder.
pause
exit /b 1
