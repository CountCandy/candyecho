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
echo   The web interface opens automatically once the backend has finished
echo   loading ^(the first run can take a while as the models load^).
echo   Close this window to stop the server ^(both the web UI and the
echo   OpenAI endpoint shut down together^).
echo ============================================================
echo.

REM Wait until the backend has actually finished loading, then open the web
REM interface. A hidden background PowerShell polls /health and only launches
REM the browser once the server answers 200 -- so it never opens too early
REM (model loading can take a while, the first run most of all). Gives up
REM quietly after ~10-25 min if the server never comes up.
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 300;$i++){try{if((Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 'http://localhost:8100/health').StatusCode -eq 200){Start-Process 'http://localhost:8100';exit}}catch{}; Start-Sleep -Seconds 2}"

REM Serving the web UI at / and the OpenAI-compatible endpoint at
REM /v1/audio/speech from the same process, so both come up at once.
uv run uvicorn longecho.main:app --host 127.0.0.1 --port 8100

echo.
echo CandyEcho has stopped.
pause
