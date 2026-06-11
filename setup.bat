@echo off
REM One-shot setup for QuPath YOLO Active Learning extension
REM Run from the repo root: setup.bat

set YOLO_AL_DIR=%APPDATA%\QuPath\yolo-al
set VENV_DIR=%YOLO_AL_DIR%\yolo-venv
set EXTENSIONS_DIR=%USERPROFILE%\QuPath\v0.7\extensions

echo === QuPath YOLO-AL Setup ===

echo [1/4] Installing Python server to %YOLO_AL_DIR%
if not exist "%YOLO_AL_DIR%" mkdir "%YOLO_AL_DIR%"
copy /Y python\yolo_server.py   "%YOLO_AL_DIR%\"
copy /Y python\requirements.txt "%YOLO_AL_DIR%\"

echo [2/4] Creating Python venv at %VENV_DIR%
python -m venv "%VENV_DIR%"

echo [3/4] Installing Python packages (may take a few minutes)...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip -q
"%VENV_DIR%\Scripts\python.exe" -m pip install -r "%YOLO_AL_DIR%\requirements.txt" -q

echo [4/4] Building and installing QuPath extension JAR...
if exist gradlew.bat (
    call gradlew.bat installExtension
) else (
    gradle installExtension
)

echo.
echo Setup complete.
echo.
echo Next steps:
echo   1. Start QuPath 0.7
echo   2. Extensions ^> YOLO Active Learning ^> Open Panel...
echo   3. Set dataset directory, annotate slides, then Extract ^> Train ^> Infer
