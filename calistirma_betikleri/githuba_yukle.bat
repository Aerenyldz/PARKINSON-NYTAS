@echo off
setlocal
cd /d "%~dp0\.."

title NYTAS-PARKINSON - GitHub Yukleme Araci
color 0B

echo =========================================================================
echo             NYTAS-PARKINSON - GITHUB YUKLEME ARACI
echo =========================================================================
echo.

REM 1. Git kurulu mu?
git --version >nul 2>&1
if %errorlevel% neq 0 goto :HATA_GIT

REM 2. Git repo kontrolu
if not exist ".git" (
    echo [*] Git deposu baslatiliyor...
    git init
    git branch -M main
)

REM 3. Uzak depo kontrolu
git remote get-url origin >nul 2>&1
if %errorlevel% equ 0 goto :REMOTE_VAR

:REMOTE_YOK
echo [*] Henuz bagli bir GitHub deposu bulunamadi.
echo.
echo Lutfen GitHub'da olusturdugunuz deponun URL adresini yapistirin:
echo Ornek: https://github.com/kullanici_adiniz/PARKINSON-NYTAS.git
echo.
set "REPO_URL="
set /p REPO_URL="Depo URL adresi: "

if "%REPO_URL%"=="" goto :HATA_URL

git remote add origin %REPO_URL%
git branch -M main
echo [*] Uzak depo basariyla baglandi.
goto :COMMIT_VE_PUSH

:REMOTE_VAR
for /f "tokens=*" %%a in ('git remote get-url origin') do set "MEVCUT_URL=%%a"
echo [*] Bagli GitHub Deposu: %MEVCUT_URL%
echo.

:COMMIT_VE_PUSH
echo [*] Dosyalar hazirlaniyor...
git add .

set "COMMIT_MSG="
set /p COMMIT_MSG="Commit Mesaji (Enter'a basarsaniz otomatik mesaj kullanilir): "
if "%COMMIT_MSG%"=="" set COMMIT_MSG=NYTAS-PARKINSON v1.3: Klinik Raporlama ve Biyomekanik Iyilestirmeler

git commit -m "%COMMIT_MSG%"

echo.
echo [*] GitHub'a gonderiliyor (git push -u origin main)...
echo.
git push -u origin main
if %errorlevel% equ 0 goto :BASARILI

goto :HATA_PUSH

:BASARILI
color 0A
echo.
echo =========================================================================
echo [TEBRIKLER] Projeniz GitHub'a basariyla yuklendi!
echo =========================================================================
echo.
goto :SON

:HATA_GIT
color 0C
echo [HATA] Sisteminizde Git kurulu bulunamadi!
echo Lutfen https://git-scm.com adresinden Git indirip kurunuz.
echo.
goto :SON

:HATA_URL
color 0C
echo [HATA] URL girmediniz. Islem iptal edildi.
echo.
goto :SON

:HATA_PUSH
color 0E
echo.
echo =========================================================================
echo [UYARI] Push islemi tamamlanamadi!
echo =========================================================================
echo Olasi Nedenler ve Cozumler:
echo 1. GitHub hesabinizla oturum acmaniz istenebilir (Tarayicida acilan onay ekranini tamamlayin).
echo 2. Eger GitHub'da depo olustururken README veya License eklediyseniz:
echo    Konsolda su komutlari calistirin:
echo      git pull origin main --allow-unrelated-histories
echo      git push -u origin main
echo =========================================================================
echo.
goto :SON

:SON
echo Devam etmek icin bir tusa basiniz...
pause >nul
