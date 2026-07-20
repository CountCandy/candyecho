@echo off
REM ============================================================
REM  CandyEcho launcher - double-click to run (no terminal needed).
REM
REM  First-time setup (run once in a terminal):
REM    uv sync
REM  Windows also needs a shared FFmpeg 8 build on PATH - see README.
REM ============================================================
cd /d "%~dp0"

echo ============================================================
echo   CandyEcho is starting up...
echo.
echo   Web interface:      http://localhost:8100
echo   OpenAI TTS endpoint: http://localhost:8100/v1/audio/speech
echo     ^(model: candyecho  -  point SillyTavern's "OpenAI Compatible"
echo      TTS provider at the endpoint above^)
echo.
echo   The web interface opens in your browser automatically.
echo   Close this window to stop the server ^(both the web UI and the
echo   OpenAI endpoint shut down together^).
echo ============================================================
echo.

REM Auto-launch the web interface a few seconds after the server binds.
start "" /min cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:8100"

REM Serving the web UI at / and the OpenAI-compatible endpoint at
REM /v1/audio/speech from the same process, so both come up at once.
uv run uvicorn longecho.main:app --host 127.0.0.1 --port 8100

echo.
echo CandyEcho has stopped.
pause
