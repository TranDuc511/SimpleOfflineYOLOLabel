@echo off
cd /d "%~dp0"
if exist "%USERPROFILE%\anaconda3\python.exe" (
    "%USERPROFILE%\anaconda3\python.exe" app.py
) else (
    python app.py
)
if errorlevel 1 pause
