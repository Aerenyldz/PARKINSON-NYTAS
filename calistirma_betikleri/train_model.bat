@echo off
chcp 65001 >nul
echo =========================================================================
echo NYTAS-PARKINSON | Makine Ogrenmesi Model Egitimi
echo =========================================================================

cd /d "%~dp0.."

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

python kaynak_kodlar\model_egitim_parkinson.py %*

echo.
echo =========================================================================
echo Egitim Islemi Tamamlandi.
echo =========================================================================
pause
