@echo off
REM ============================================================
REM  CandyEcho launcher - double-click to run (no terminal needed).
REM
REM  First-time setup (run once in a terminal):
REM    uv sync
REM  Windows also needs a shared FFmpeg 8 build on PATH - see README.
REM ============================================================
cd /d "%~dp0"

echo Starting CandyEcho on http://localhost:8100
echo Your browser will open in a few seconds. Close this window to stop the server.
echo.

REM Open the browser a few seconds after the server has had time to start.
start "" /min cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:8100"

uv run uvicorn longecho.main:app --host 127.0.0.1 --port 8100

echo.
echo CandyEcho has stopped.
pause
