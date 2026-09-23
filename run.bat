@echo off
REM One-command launcher (Windows): virtualenv + deps (once) + frontend build + server on :8000
setlocal
cd /d "%~dp0"

if not exist .venv (
  echo ==^> Creating virtual environment
  python -m venv .venv || goto :nopython
)
call .venv\Scripts\activate.bat

if not exist .venv\.deps-installed (
  echo ==^> Installing Python packages ^(first run takes a few minutes: torch is large^)
  python -m pip install --upgrade pip
  python -m pip install -r backend\requirements.txt || goto :error
  echo done> .venv\.deps-installed
)

where npm >nul 2>nul
if %errorlevel%==0 (
  if not exist frontend\node_modules (
    echo ==^> Installing frontend packages
    pushd frontend & call npm install & popd
  )
  echo ==^> Building frontend
  pushd frontend & call npm run build & popd
) else (
  echo ==^> npm not found - using the prebuilt frontend in frontend\dist
)

echo.
echo   RL Rebalancing Lab is starting at  http://localhost:8000
echo.
start "" http://localhost:8000
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
goto :eof

:nopython
echo Python 3.10+ was not found. Install it from https://www.python.org/downloads/ (tick "Add to PATH").
pause
goto :eof

:error
echo Installation failed - see the messages above.
pause
