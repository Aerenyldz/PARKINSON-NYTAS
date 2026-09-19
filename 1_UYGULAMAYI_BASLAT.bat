@echo off
chcp 65001 >nul
title NYTAS-PARKINSON Baslatıcı
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

python main.py

if errorlevel 1 (
    echo.
    echo [UYARI] Bir hata olustu veya program kapandi.
    pause
)
