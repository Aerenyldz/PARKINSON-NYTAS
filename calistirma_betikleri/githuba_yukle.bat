@echo off
chcp 65001 >nul
title NYTAS-PARKINSON - GitHub'a Yükleme Aracı
color 0B

echo =========================================================================
echo             NYTAS-PARKINSON — GITHUB YÜKLEME VE SENKRONİZASYON
echo =========================================================================
echo.

cd /d "%~dp0\.."

REM Git kurulu mu kontrol et
git --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [HATA] Sisteminizde Git kurulu bulunamadi!
    echo Lutfen https://git-scm.com adresinden Git indirip kurunuz.
    echo.
    pause
    exit /b 1
)

REM Git reposu baslatilmis mi kontrol et
if not exist ".git" (
    echo [*] Git deposu baslatiliyor...
    git init
    git branch -M main
)

REM Uzak depo (remote origin) var mi kontrol et
git remote get-url origin >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [DIKKAT] Henuz bir GitHub uzak depo (remote origin) baglanmamis!
    echo.
    echo Lutfen GitHub'da actiginiz reponun URL adresini giriniz:
    echo (Ornek: https://github.com/KULLANICI_ADINIZ/PARKINSON-NYTAS.git)
    echo.
    set /p REPO_URL="GitHub Repo URL: "
    if "%REPO_URL%"=="" (
        echo [HATA] Gecerli bir URL girmediniz. Islem iptal edildi.
        pause
        exit /b 1
    )
    git remote add origin %REPO_URL%
    git branch -M main
    echo [*] Uzak depo baglandi: %REPO_URL%
) else (
    for /f "tokens=*" %%i in ('git remote get-url origin') do set MEVCUT_URL=%%i
    echo [*] Bagli GitHub Deposu: %MEVCUT_URL%
)

echo.
echo [*] Dosyalar hazirlaniyor (git add)...
git add .

echo.
set /p COMMIT_MSG="Commit Mesaji (Bos birakirsaniz varsayilan kullanilir): "
if "%COMMIT_MSG%"=="" (
    set COMMIT_MSG=NYTAS-PARKINSON v1.3: Klinik Raporlama ve Biyomekanik Iyilestirmeler
)

echo.
echo [*] Degisiklikler kaydediliyor (git commit)...
git commit -m "%COMMIT_MSG%"

echo.
echo [*] GitHub'a yukleniyor (git push -u origin main)...
git push -u origin main

if %errorlevel% equ 0 (
    color 0A
    echo.
    echo =========================================================================
    echo [BASARILI] Projeniz GitHub'a basariyla yuklendi!
    echo =========================================================================
) else (
    color 0E
    echo.
    echo [NOT] Eger ilk push sirasinda hata aldiysaniz:
    echo 1. GitHub kullanici adi / Personal Access Token veya SSH anahtarinizi kontrol edin.
    echo 2. Eger repoda README onceden olustuysa, su komutu calistirin:
    echo    git pull origin main --rebase
    echo    git push -u origin main
)

echo.
pause
