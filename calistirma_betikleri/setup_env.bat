@echo off
chcp 65001 >nul
echo =========================================================================
echo NYTAS-PARKINSON | Sanal Ortam Kurulum Betigi (Windows)
echo =========================================================================

cd /d "%~dp0.."

if not exist ".venv" (
    echo [.VENV] Sanal ortam olusturuluyor...
    python -m venv .venv
    if errorlevel 1 (
        echo [HATA] Python sanal ortam olusturulamadi! Python3'un yuklu oldugundan emin olun.
        pause
        exit /b 1
    )
) else (
    echo [.VENV] Sanal ortam zaten mevcut (.venv).
)

echo [.VENV] Sanal ortam aktif ediliyor...
call .venv\Scripts\activate.bat

echo [PIP] Paketler guncelleniyor ve bagimliliklar yukleniyor...
python -m pip install --upgrade pip
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo [HATA] requirements.txt bulunamadi!
)

echo.
echo =========================================================================
echo Kurulum Tamamlandi!
echo Uygulamayi baslatmak icin main.py veya scripts\run_nytas.bat calistirabilirsiniz.
echo =========================================================================
pause
