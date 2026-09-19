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
        PARKINSON_MODEL_PATH, DATASET_NORMAL_PATH, DATASET_PARKINSON_PATH, MASTER_CSV_PATH, SEANSLAR_OZET_PATH, RAW_DATA_DIR
    )
    MODEL_CIKTI = str(PARKINSON_MODEL_PATH)
    DEFAULT_NORMAL_CSV = str(DATASET_NORMAL_PATH)
    DEFAULT_PARKINSON_CSV = str(DATASET_PARKINSON_PATH)
    DEFAULT_MASTER_CSV = str(MASTER_CSV_PATH)
    DEFAULT_OZET_CSV = str(SEANSLAR_OZET_PATH)
    DEFAULT_RAW_DIR = str(RAW_DATA_DIR)
except Exception:
    try:
        from config.settings import (
            PARKINSON_MODEL_PATH, DATASET_NORMAL_PATH, DATASET_PARKINSON_PATH, MASTER_CSV_PATH, SEANSLAR_OZET_PATH, RAW_DATA_DIR
        )
        MODEL_CIKTI = str(PARKINSON_MODEL_PATH)
        DEFAULT_NORMAL_CSV = str(DATASET_NORMAL_PATH)
        DEFAULT_PARKINSON_CSV = str(DATASET_PARKINSON_PATH)
        DEFAULT_MASTER_CSV = str(MASTER_CSV_PATH)
        DEFAULT_OZET_CSV = str(SEANSLAR_OZET_PATH)
        DEFAULT_RAW_DIR = str(RAW_DATA_DIR)
    except Exception:
        MODEL_CIKTI = str(BASE_DIR / "modeller" / "parkinson_model.pkl")
        DEFAULT_NORMAL_CSV = str(BASE_DIR / "veriler" / "raw" / "dataset_normal_parkinson.csv")
        DEFAULT_PARKINSON_CSV = str(BASE_DIR / "veriler" / "raw" / "dataset_parkinson_parkinson.csv")
        DEFAULT_MASTER_CSV = str(BASE_DIR / "veriler" / "processed" / "nytas_parkinson_veri" / "parkinson_master.csv")
        DEFAULT_OZET_CSV = str(BASE_DIR / "veriler" / "processed" / "nytas_parkinson_veri" / "seanslar_ozet.csv")
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


def verileri_yukle_master(master_csv: str = DEFAULT_MASTER_CSV, ozet_csv: str = DEFAULT_OZET_CSV) -> pd.DataFrame:
    """
    Canlı seans verilerini yükler ve model eğitimine hazırlar.

    Bilimsel & Klinik Öğrenme İlkesi:
    - Sadece hekim tarafından 'Klinik Ön Tanı / Etiket' atanmış (0: Sağlıklı, 1: Parkinson)
      seanslar gerçek 'Ground Truth' kabul edilir.
    - 'Bilinmiyor / Rutin Tarama (-1)' statüsündeki kayıtlar, modelin kendi yanlış tahminlerini
      ezberlemesini (döngüsel öğrenme) önlemek amacıyla eğitime doğrudan dahil edilmez.
    - Eğer henüz teyitli canlı veri yoksa veya az sayıda ise, referans klinik veri seti ile
      desteklenerek modelin kararlı kalması sağlanır.
    """
    print("=" * 55)
    print("  NYTAS-PARKINSON | Canlı Seans Verileri ile Model Eğitimi")
    print("=" * 55)

    df_canli_temiz = None

    # 1. Öncelik: Kararlı Seans Özet Havuzunu (seanslar_ozet.csv) Kontrol Et
    if os.path.exists(ozet_csv):
        try:
            df_ozet = pd.read_csv(ozet_csv)
            if "klinik_etiket" in df_ozet.columns:
                df_teyitli = df_ozet[df_ozet["klinik_etiket"].isin([0, 1])].copy()
                if not df_teyitli.empty:
                    print(f"  [BİLGİ] 'seanslar_ozet.csv' havuzunda {len(df_teyitli)} doğrulanmış klinik seans bulundu.")
                    df_canli_temiz = pd.DataFrame()
                    for f in PARKINSON_FEATURES:
                        df_canli_temiz[f] = pd.to_numeric(df_teyitli.get(f, 0.0), errors="coerce").fillna(0.0)
                    df_canli_temiz["label"] = df_teyitli["klinik_etiket"].astype(int)
        except Exception as e:
            print(f"  [UYARI] Seans özet dosyası okunurken hata: {e}")

    # 2. İkinci Öncelik: Frame Bazlı Master CSV'yi (parkinson_master.csv) Kontrol Et
    if df_canli_temiz is None or df_canli_temiz.empty:
        if os.path.exists(master_csv):
            try:
                df_master = pd.read_csv(master_csv)
                if "klinik_etiket" in df_master.columns:
                    if "tahmin" in df_master.columns:
                        df_master = df_master[~df_master["tahmin"].str.contains("Toplaniyor", na=False, case=False)]
                    df_teyitli = df_master[df_master["klinik_etiket"].isin([0, 1])].copy()
                    if not df_teyitli.empty:
                        print(f"  [BİLGİ] 'parkinson_master.csv' içinde {len(df_teyitli)} doğrulanmış kare bulundu.")
                        df_canli_temiz = pd.DataFrame()
                        df_canli_temiz["kol_asimetri"] = pd.to_numeric(df_teyitli.get("kol_asimetri_ort", df_teyitli.get("kol_asimetri", 0.0)), errors="coerce")
                        df_canli_temiz["adim_uzunlugu_cm"] = pd.to_numeric(df_teyitli.get("adim_uzunlugu_ort", df_teyitli.get("adim_uzunlugu_cm", 0.0)), errors="coerce")
                        df_canli_temiz["govde_egimi"] = pd.to_numeric(df_teyitli.get("govde_egimi_ort", df_teyitli.get("govde_egimi", 0.0)), errors="coerce")
                        df_canli_temiz["kadans_spm"] = pd.to_numeric(df_teyitli.get("kadans_spm", 0.0), errors="coerce")
                        df_canli_temiz["fog_skoru"] = pd.to_numeric(df_teyitli.get("fog_skoru", 0.0), errors="coerce")
                        df_canli_temiz["yuruyus_hizi_cms"] = pd.to_numeric(df_teyitli.get("yuruyus_hizi_cms", 0.0), errors="coerce")
                        df_canli_temiz["tremor_hz"] = pd.to_numeric(df_teyitli.get("tremor_hz_max", df_teyitli.get("tremor_hz_L", 0.0)), errors="coerce")
                        df_canli_temiz["kol_genlik_ort"] = pd.to_numeric(df_teyitli.get("kol_genlik_min", df_teyitli.get("kol_genlik_ort", 0.0)), errors="coerce")
                        df_canli_temiz["label"] = df_teyitli["klinik_etiket"].astype(int)
            except Exception as e:
                print(f"  [UYARI] Master CSV okunurken hata: {e}")

    # 3. Sonuç Değerlendirme & Hibrit Birleştirme (Klinik Güvenlik)
    if df_canli_temiz is None or df_canli_temiz.empty:
        print("\n" + "!" * 55)
        print("  [KLİNİK BİLGİLENDİRME]")
        print("  Canlı seanslarda hekim tarafından doğrulanmış etiket (0 veya 1) bulunamadı!")
        print("  Mevcut seanslar 'Bilinmiyor / Rutin Tarama (-1)' statüsündedir.")
        print("  Modelin kendi tahminlerini ezberlemesini (döngüsel öğrenme hatası)")
        print("  önlemek için referans klinik veri seti (dataset_normal + dataset_parkinson)")
        print("  kullanılarak temel model güncelleniyor.")
        print("  (İpucu: Yeni test başlatırken 'Klinik Durum' seçeneğinden hastanın tanısını")
        print("  seçerek modele gerçek klinik tecrübe kazandırabilirsiniz.)")
        print("!" * 55 + "\n")
        return verileri_yukle_csv(DEFAULT_NORMAL_CSV, DEFAULT_PARKINSON_CSV)

    # NaN temizliği
    df_canli_temiz = df_canli_temiz.dropna(subset=PARKINSON_FEATURES)

    # Eğer canlı veride sadece tek sınıf varsa veya örnek sayısı az ise,
    # modelin çökmemesi ve genel geçerliliğini koruması için referans veriyle takviye et
    siniflar = df_canli_temiz["label"].unique()
    if len(siniflar) < 2 or len(df_canli_temiz) < 50:
        print(f"  [GÜVENCE] Canlı veri ({len(df_canli_temiz)} örnek, sınıflar: {siniflar.tolist()}) referans veri setiyle harmanlanıyor...")
        df_ref = verileri_yukle_csv(DEFAULT_NORMAL_CSV, DEFAULT_PARKINSON_CSV)
        df_final = pd.concat([df_ref, df_canli_temiz], ignore_index=True)
        df_final = df_final.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        print(f"  [BAŞARILI] {len(df_canli_temiz)} adet doğrulanmış canlı veriyle bağımsız eğitim yapılıyor.")
        df_final = df_canli_temiz.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"  Toplam Eğitim Örneği: {len(df_final)}")
    print(f"  Sınıf Dağılımı:\n{df_final['label'].value_counts().to_string()}")
    return df_final


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
        df = verileri_yukle_master(DEFAULT_MASTER_CSV, DEFAULT_OZET_CSV)
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
