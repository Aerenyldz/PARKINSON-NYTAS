@echo off
chcp 65001 >nul
echo =========================================================================
echo NYTAS-PARKINSON | Canli Yuruyus Analiz Modulu Baslatiliyor...
echo =========================================================================

cd /d "%~dp0.."

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

python kaynak_kodlar\nytas_parkinson.py %*

if errorlevel 1 (
    echo.
    echo [BILGI] Program sonlandi veya bir hata olustu.
    pause
)
