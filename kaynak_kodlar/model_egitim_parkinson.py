"""
=============================================================================
NYTAS-PARKINSON — Model Egitim Scripti
TUBITAK 1002 Hizli Destek Programi
=============================================================================
Bu script, "nytas_parkinson.py" modulunun topladigi verilere uygun
bir makine ogrenmesi modeli egitir.

Kullanim:
  1. Oncelikle "dataset_normal.csv" ve "dataset_parkinson.csv" dosyalarini
     olusturmaniz gerekir. Bunlar asagidaki 8 sutunu icermelidir:
     - kol_asimetri       : Kol salinim asimetrisi (%)
     - adim_uzunlugu_cm   : Adim uzunlugu (cm)
     - govde_egimi        : Govde one egimi (derece)
     - kadans_spm         : Kadans (adim/dakika)
     - fog_skoru           : Freezing of Gait skoru (0.0 / 0.5 / 1.0)
     - yuruyus_hizi_cms   : Yuruyus hizi (cm/s)
     - tremor_hz          : Tremor frekansi (Hz)
     - kol_genlik_ort     : Ortalama kol salinim genligi (cm)
     - label              : 0 = Normal, 1 = Parkinson

  2. Alternatif olarak, nytas_parkinson.py ile toplanan
     "nytas_parkinson_veri/parkinson_master.csv" dosyasindan
     verileri otomatik cekebilirsiniz (--master-csv argumani).

  3. Egitilen model "parkinson_model.pkl" olarak kaydedilir.
     nytas_parkinson.py bunu otomatik yukler.
=============================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.pipeline import Pipeline
import joblib

# Path Ayarları
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from ayarlar.settings import (
        PARKINSON_MODEL_PATH, DATASET_NORMAL_PATH, DATASET_PARKINSON_PATH, MASTER_CSV_PATH, RAW_DATA_DIR
    )
    MODEL_CIKTI = str(PARKINSON_MODEL_PATH)
    DEFAULT_NORMAL_CSV = str(DATASET_NORMAL_PATH)
    DEFAULT_PARKINSON_CSV = str(DATASET_PARKINSON_PATH)
    DEFAULT_MASTER_CSV = str(MASTER_CSV_PATH)
    DEFAULT_RAW_DIR = str(RAW_DATA_DIR)
except Exception:
    try:
        from config.settings import (
            PARKINSON_MODEL_PATH, DATASET_NORMAL_PATH, DATASET_PARKINSON_PATH, MASTER_CSV_PATH, RAW_DATA_DIR
        )
        MODEL_CIKTI = str(PARKINSON_MODEL_PATH)
        DEFAULT_NORMAL_CSV = str(DATASET_NORMAL_PATH)
        DEFAULT_PARKINSON_CSV = str(DATASET_PARKINSON_PATH)
        DEFAULT_MASTER_CSV = str(MASTER_CSV_PATH)
        DEFAULT_RAW_DIR = str(RAW_DATA_DIR)
    except Exception:
        MODEL_CIKTI = str(BASE_DIR / "modeller" / "parkinson_model.pkl")
        DEFAULT_NORMAL_CSV = str(BASE_DIR / "veriler" / "raw" / "dataset_normal_parkinson.csv")
        DEFAULT_PARKINSON_CSV = str(BASE_DIR / "veriler" / "raw" / "dataset_parkinson_parkinson.csv")
        DEFAULT_MASTER_CSV = str(BASE_DIR / "veriler" / "processed" / "nytas_parkinson_veri" / "parkinson_master.csv")
        DEFAULT_RAW_DIR = str(BASE_DIR / "veriler" / "raw")

# =========================================================================
# Konfigurasyon
# =========================================================================
# nytas_parkinson.py'nin ML modeline gonderdigi 8 parametre
PARKINSON_FEATURES = [
    "kol_asimetri",         # Kol salinim asimetrisi (%)
    "adim_uzunlugu_cm",     # Adim uzunlugu (cm)
    "govde_egimi",          # Govde one egimi (derece)
    "kadans_spm",           # Kadans (adim/dakika)
    "fog_skoru",            # Freezing of Gait (0/0.5/1.0)
    "yuruyus_hizi_cms",     # Yuruyus hizi (cm/s)
    "tremor_hz",            # El tremor frekansi (Hz)
    "kol_genlik_ort",       # Ortalama kol salinim genligi (cm)
]

RANDOM_STATE = 42


# =========================================================================
# Veri Yukleme: Manuel CSV'ler veya Master CSV
# =========================================================================
def verileri_yukle_csv(normal_csv: str, parkinson_csv: str) -> pd.DataFrame:
    """dataset_normal.csv ve dataset_parkinson.csv dosyalarindan yukler."""
    print("=" * 55)
    print("  NYTAS-PARKINSON | Model Egitim Aracı")
    print("=" * 55)

    try:
        df_normal = pd.read_csv(normal_csv)
        df_parkinson = pd.read_csv(parkinson_csv)
        print(f"  Normal veri     : {len(df_normal)} satir  ({normal_csv})")
        print(f"  Parkinson veri  : {len(df_parkinson)} satir  ({parkinson_csv})")
    except FileNotFoundError as e:
        print(f"\nHATA: CSV dosyasi bulunamadi -> {e}")
        print(f"Beklenen dosyalar: '{normal_csv}' ve '{parkinson_csv}'")
        print("\nBu dosyalar su sutunlari icermeli:")
        for f in PARKINSON_FEATURES:
            print(f"  - {f}")
        print("  - label  (0=Normal, 1=Parkinson)")
        sys.exit(1)

    # Birlestir ve karistir
    df = pd.concat([df_normal, df_parkinson], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def verileri_yukle_master(master_csv: str) -> pd.DataFrame:
    """nytas_parkinson.py'nin topladigi parkinson_master.csv'den yukler.
    Bu dosyada 'tahmin' sutunundan label cikarilir (kullanici isaretlemis olmali)."""
    print("=" * 55)
    print("  NYTAS-PARKINSON | Master CSV'den Model Egitimi")
    print("=" * 55)

    try:
        df = pd.read_csv(master_csv)
        print(f"  Toplam satir: {len(df)}  ({master_csv})")
    except FileNotFoundError:
        print(f"HATA: {master_csv} bulunamadi!")
        sys.exit(1)

    # Gerekli sutunlarin var olup olmadigini kontrol et
    mevcut_sutunlar = set(df.columns)
    # Master CSV'deki sutun isimleri eslestirmesi:
    master_mapping = {
        "kol_asimetri_ort": "kol_asimetri",
        "kol_asimetri":     "kol_asimetri",
        "adim_uzunlugu_ort": "adim_uzunlugu_cm",
        "adim_uzunlugu_cm": "adim_uzunlugu_cm",
        "govde_egimi_ort":  "govde_egimi",
        "govde_egimi":      "govde_egimi",
        "kadans_spm":       "kadans_spm",
        "fog_skoru":        "fog_skoru",
        "yuruyus_hizi_cms": "yuruyus_hizi_cms",
        "tremor_hz_max":    "tremor_hz",       # Bilateral en yuksek el tremoru
        "tremor_hz_L":      "tremor_hz",
        "kol_genlik_min":   "kol_genlik_ort",   # En cok kisitlanan taraf salinimi
        "kol_genlik_ort":   "kol_genlik_ort",
        "kol_genlik_L_ort": "kol_genlik_ort",
    }

    # Sutunlari yeniden isimlendir
    rename_map = {}
    for src, dst in master_mapping.items():
        if src in mevcut_sutunlar and src != dst:
            rename_map[src] = dst
    df = df.rename(columns=rename_map)

    # label olustur: Eger 'label' sutunu yoksa, kullanici 'tahmin' sutunundan cikart
    if "label" not in df.columns:
        if "tahmin" in df.columns:
            print("  'label' sutunu yok, 'tahmin' sutunundan olusturuluyor...")
            df["label"] = df["tahmin"].apply(
                lambda x: 1 if "PARKINSON" in str(x).upper() or "RISK" in str(x).upper() else 0
            )
        else:
            print("HATA: CSV'de ne 'label' ne de 'tahmin' sutunu var!")
            sys.exit(1)

    # NaN temizligi
    for col in PARKINSON_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[c for c in PARKINSON_FEATURES if c in df.columns])

    print(f"  Temizlenmis satir: {len(df)}")
    print(f"  Label dagilimi:\n{df['label'].value_counts().to_string()}")
    return df


# =========================================================================
# Veri Analizi ve On Isleme
# =========================================================================
def veri_analizi(df: pd.DataFrame):
    """Verinin genel ozeti ve Parkinson'a ozgu istatistikler."""
    print("\n" + "-" * 55)
    print("  VERI ANALIZI")
    print("-" * 55)

    mevcut = [f for f in PARKINSON_FEATURES if f in df.columns]
    eksik  = [f for f in PARKINSON_FEATURES if f not in df.columns]

    if eksik:
        print(f"\n  UYARI: Su sutunlar eksik: {eksik}")
        print("  Eksik sutunlar 0 ile doldurulacak.\n")
        for e in eksik:
            df[e] = 0.0

    print(f"\n  Toplam ornek: {len(df)}")
    print(f"  Normal (0)  : {len(df[df['label'] == 0])}")
    print(f"  Parkinson(1): {len(df[df['label'] == 1])}")

    # Her bir biyobelirtecin ortalama degerleri (Normal vs Parkinson)
    print(f"\n  {'Biyobelirtec':<22} {'Normal':>10} {'Parkinson':>12}")
    print("  " + "-" * 46)
    for feat in mevcut:
        ort_n = df[df['label'] == 0][feat].mean()
        ort_p = df[df['label'] == 1][feat].mean()
        print(f"  {feat:<22} {ort_n:>10.2f} {ort_p:>12.2f}")

    return df


# =========================================================================
# Model Egitimi
# =========================================================================
def model_egit(df: pd.DataFrame):
    """Parkinson siniflandirma modelini egitir, degerlendirir ve kaydeder."""
    print("\n" + "=" * 55)
    print("  MODEL EGITIMI BASLIYOR")
    print("=" * 55)

    # Giris (X) ve cikis (y)
    mevcut = [f for f in PARKINSON_FEATURES if f in df.columns]
    X = df[mevcut].values
    y = df["label"].values

    # Egitim / Test bolumu (%80 / %20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Egitim seti : {len(X_train)} ornek")
    print(f"  Test seti   : {len(X_test)} ornek")

    # Pipeline: Olcekleme + Model
    # Random Forest (az veriyle robust) + Gradient Boosting (yuksek dogruluk)
    modeller = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                class_weight="balanced",    # Dengesiz veri icin onemli
                random_state=RANDOM_STATE
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=RANDOM_STATE
            ))
        ]),
    }

    en_iyi_model = None
    en_iyi_skor = 0.0
    en_iyi_isim = ""

    for isim, pipeline in modeller.items():
        print(f"\n  --- {isim} ---")

        # Capraz dogrulama (5-Fold Stratified)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_skorlar = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        print(f"  5-Fold CV Ort: %{cv_skorlar.mean()*100:.2f} (+/- {cv_skorlar.std()*100:.2f})")

        # Egit
        pipeline.fit(X_train, y_train)

        # Test
        tahminler = pipeline.predict(X_test)
        dogruluk = accuracy_score(y_test, tahminler)
        print(f"  Test Dogruluk: %{dogruluk*100:.2f}")

        # AUC (eger predict_proba varsa)
        if hasattr(pipeline, "predict_proba"):
            probalar = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probalar)
            print(f"  AUC-ROC     : {auc:.4f}")

        if dogruluk > en_iyi_skor:
            en_iyi_skor = dogruluk
            en_iyi_model = pipeline
            en_iyi_isim = isim

    # En iyi modeli sec
    print("\n" + "=" * 55)
    print(f"  EN IYI MODEL: {en_iyi_isim}")
    print(f"  DOGRULUK    : %{en_iyi_skor*100:.2f}")
    print("=" * 55)

    # Detayli rapor
    tahminler = en_iyi_model.predict(X_test)
    print("\n  Siniflandirma Raporu:")
    print(classification_report(y_test, tahminler,
                                target_names=["Normal", "Parkinson"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, tahminler)
    print("  Karisiklik Matrisi (Confusion Matrix):")
    print(f"                  Tahmin Normal  Tahmin Parkinson")
    print(f"  Gercek Normal        {cm[0][0]:>5}          {cm[0][1]:>5}")
    print(f"  Gercek Parkinson     {cm[1][0]:>5}          {cm[1][1]:>5}")

    # Feature Importance (Ozellik Onemliligi)
    if hasattr(en_iyi_model.named_steps["clf"], "feature_importances_"):
        print("\n  Ozellik Onemliligi (Parkinson Tanisinda En Etkili Parametreler):")
        print("  " + "-" * 46)
        importances = en_iyi_model.named_steps["clf"].feature_importances_
        idxs = np.argsort(importances)[::-1]
        for rank, i in enumerate(idxs, 1):
            bar = "#" * int(importances[i] * 40)
            print(f"  {rank}. {mevcut[i]:<22} {importances[i]:.4f}  {bar}")

    # Modeli kaydet
    joblib.dump(en_iyi_model, MODEL_CIKTI)
    print(f"\n  Model kaydedildi -> '{MODEL_CIKTI}'")
    print(f"  nytas_parkinson.py bu dosyayi otomatik yukleyecektir.\n")


# =========================================================================
# Sentetik Veri Uretici (Veriniz yoksa test amacli)
# =========================================================================
def sentetik_veri_uret(n_normal: int = 200, n_parkinson: int = 200):
    """Gercek klinik veriye erisim yoksa, test amacli sentetik veri olusturur.
    Degerler literatur ortalamalarina ve MDS klinik calismalarina dayalidir.

    v1.3 Guncelleme: Parkinson parametreleri erken-orta evre (Hoehn & Yahr 1-2)
    klinik bulgularina gore ayarlandi. Eski degerler sadece ileri evre hastaları
    yakaliyordu, yeni degerler erken taniya duyarli."""
    print("\n  Sentetik veri uretiliyor (v1.3 — klinik literatur referanslariyla)...")

    np.random.seed(RANDOM_STATE)

    # =====================================================================
    # Normal yuruyus parametreleri (saglikli kontrol grubu)
    # Kaynak: Gait & Posture meta-analiz, MDS-UPDRS referans degerleri
    # =====================================================================
    normal = pd.DataFrame({
        "kol_asimetri":     np.random.normal(5.0,  2.5,  n_normal).clip(0.0, 12.0),
        "adim_uzunlugu_cm": np.random.normal(70.0, 6.0,  n_normal).clip(55.0, 90.0),
        "govde_egimi":      np.random.normal(3.0,  1.5,  n_normal).clip(0.0, 6.0),
        "kadans_spm":       np.random.normal(112.0, 8.0, n_normal).clip(90.0, 135.0),
        "fog_skoru":        np.zeros(n_normal),
        "yuruyus_hizi_cms": np.random.normal(125.0, 12.0, n_normal).clip(95.0, 155.0),
        "tremor_hz":        np.zeros(n_normal),
        "kol_genlik_ort":   np.random.normal(24.0, 4.0,  n_normal).clip(15.0, 38.0),
        "label":            np.zeros(n_normal, dtype=int),
    })

    # =====================================================================
    # Parkinson yuruyus parametreleri (Erken-Orta Evre / H&Y 1-2)
    # Kaynak: MDS Task Force, Movement Disorders, Gait & Posture
    #
    # Degisiklik Ozeti (v1.2 -> v1.3):
    #   kol_asimetri:     mean 38 -> 18  (MDS: PD ort %13.9±7.9)
    #   adim_uzunlugu_cm: mean 38 -> 45  (erken PD 35-65 cm arasi)
    #   govde_egimi:      mean 14 -> 10  (10-25° tipik PD araligi)
    #   kadans_spm:       mean 85 -> 95  (erken PD hafif dusuk)
    #   yuruyus_hizi_cms: mean 65 -> 80  (erken PD 60-110 cm/s)
    #   kol_genlik_ort:   mean 7  -> 12  (daha az kisitlanmis)
    #   tremor orani:     %75 -> %50     (erken evrede daha az siklik)
    #   fog_skoru:        esit dagilim -> erken evrede nadir
    # =====================================================================

    # Erken evre PD'de istirahat tremoru yaklasik %50-60 hastada gorulur
    park_tremor = np.where(
        np.random.rand(n_parkinson) < 0.50,
        np.random.normal(4.8, 0.8, n_parkinson).clip(3.5, 7.5),
        0.0
    )

    parkinson = pd.DataFrame({
        "kol_asimetri":     np.random.normal(18.0, 8.0,  n_parkinson).clip(8.0, 45.0),
        "adim_uzunlugu_cm": np.random.normal(45.0, 10.0, n_parkinson).clip(25.0, 65.0),
        "govde_egimi":      np.random.normal(10.0, 3.5,  n_parkinson).clip(4.0, 22.0),
        "kadans_spm":       np.random.normal(95.0, 12.0, n_parkinson).clip(60.0, 125.0),
        "fog_skoru":        np.random.choice([0.0, 0.5, 1.0], n_parkinson, p=[0.60, 0.25, 0.15]),
        "yuruyus_hizi_cms": np.random.normal(80.0, 15.0, n_parkinson).clip(40.0, 115.0),
        "tremor_hz":        park_tremor,
        "kol_genlik_ort":   np.random.normal(12.0, 3.5,  n_parkinson).clip(4.0, 22.0),
        "label":            np.ones(n_parkinson, dtype=int),
    })

    df = pd.concat([normal, parkinson], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # CSV olarak da kaydet
    normal.to_csv(DEFAULT_NORMAL_CSV, index=False)
    parkinson.to_csv(DEFAULT_PARKINSON_CSV, index=False)
    print(f"  Normal: {n_normal}, Parkinson: {n_parkinson} ornek uretildi.")
    print(f"  Kaydedildi: {DEFAULT_NORMAL_CSV}, {DEFAULT_PARKINSON_CSV}")

    return df


# =========================================================================
# Ana Program
# =========================================================================
if __name__ == "__main__":
    print()

    # Mod secimi
    if "--sentetik" in sys.argv:
        # Test amacli sentetik veri ile egit
        df = sentetik_veri_uret()
    elif "--master-csv" in sys.argv:
        # nytas_parkinson.py'nin topladigi verilerden egit
        df = verileri_yukle_master(DEFAULT_MASTER_CSV)
    else:
        # Manuel olusturulmus CSV dosyalarindan egit (varsayilan)
        df = verileri_yukle_csv(DEFAULT_NORMAL_CSV, DEFAULT_PARKINSON_CSV)

    # Analiz ve egitim
    df = veri_analizi(df)
    model_egit(df)

    print("  Kullanim Notlari:")
    print("  -" * 28)
    print("  python model_egitim_parkinson.py                -> Manuel CSV'lerden egit")
    print("  python model_egitim_parkinson.py --master-csv   -> Toplanan verilerden egit")
    print("  python model_egitim_parkinson.py --sentetik     -> Test amacli sentetik veri")
    print()
