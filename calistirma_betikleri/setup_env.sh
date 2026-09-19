#!/usr/bin/env bash
echo "========================================================================="
echo "NYTAS-PARKINSON | Sanal Ortam Kurulum Betigi (Linux/macOS)"
echo "========================================================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( dirname "$SCRIPT_DIR" )"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
    echo "[.VENV] Sanal ortam olusturuluyor..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[HATA] Python3 venv oluşturulamadı!"
        exit 1
    fi
else
    echo "[.VENV] Sanal ortam zaten mevcut (.venv)."
fi

echo "[.VENV] Sanal ortam aktif ediliyor..."
source .venv/bin/activate

echo "[PIP] Bagimliliklar yukleniyor..."
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "[HATA] requirements.txt bulunamadı!"
fi

echo "========================================================================="
echo "Kurulum Tamamlandi!"
echo "Uygulamayi baslatmak icin 'python3 main.py' komutunu kullanabilirsiniz."
echo "========================================================================="
