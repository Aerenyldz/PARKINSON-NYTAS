"""
=============================================================================
NYTAS-PARKINSON — Proje Yapılandırma ve Yol Tanımları
=============================================================================
"""

import os
from pathlib import Path

# Proje Kök Dizini (PARKİNSON-NYTAS)
BASE_DIR = Path(__file__).resolve().parent.parent

# Klasör Yolları (Türkçe Yapılandırma)
AYARLAR_DIR = BASE_DIR / "ayarlar"
VERILER_DIR = BASE_DIR / "veriler"
HAM_VERI_DIR = VERILER_DIR / "raw"
ISLENMIS_VERI_DIR = VERILER_DIR / "processed" / "nytas_parkinson_veri"
MODELLER_DIR = BASE_DIR / "modeller"
KAYNAK_KODLAR_DIR = BASE_DIR / "kaynak_kodlar"
DOKUMANLAR_DIR = BASE_DIR / "dokumanlar"
BETIKLER_DIR = BASE_DIR / "calistirma_betikleri"

# Geriye Dönük Uyumluluk Takma Adları (Backward Compatibility Aliases)
CONFIG_DIR = AYARLAR_DIR
DATA_DIR = VERILER_DIR
RAW_DATA_DIR = HAM_VERI_DIR
PROCESSED_DATA_DIR = ISLENMIS_VERI_DIR
MODELS_DIR = MODELLER_DIR
SRC_DIR = KAYNAK_KODLAR_DIR
DOCS_DIR = DOKUMANLAR_DIR
SCRIPTS_DIR = BETIKLER_DIR

# Model ve Landmark Dosya Yolları
POSE_LANDMARKER_PATH = MODELLER_DIR / "pose_landmarker.task"
PARKINSON_MODEL_PATH = MODELLER_DIR / "parkinson_model.pkl"

# Veri Seti Dosya Yolları
DATASET_NORMAL_PATH = HAM_VERI_DIR / "dataset_normal_parkinson.csv"
DATASET_PARKINSON_PATH = HAM_VERI_DIR / "dataset_parkinson_parkinson.csv"
MASTER_CSV_PATH = ISLENMIS_VERI_DIR / "parkinson_master.csv"
SEANSLAR_OZET_PATH = ISLENMIS_VERI_DIR / "seanslar_ozet.csv"

# Klasörlerin varlığını garanti et
for _dir in [AYARLAR_DIR, VERILER_DIR, HAM_VERI_DIR, ISLENMIS_VERI_DIR, MODELLER_DIR, KAYNAK_KODLAR_DIR, DOKUMANLAR_DIR, BETIKLER_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

