"""
=============================================================================
NYTAS-PARKINSON — Klinik Rapor Oluşturucu Modülü (Faz 2)
=============================================================================
Açıklama:
  Tamamlanan yürüyüş analizi seans verilerini (parkinson_master.csv veya
  seans_verileri.csv) işleyerek hekimlere sunulabilir standart medikal
  klinik raporlar (HTML/PDF) ve görsel grafikler (Radar & Zaman Serisi) üretir.
=============================================================================
"""

import base64
import os
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # GUI thread çakışmalarını önlemek için arka plan render
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Proje Kök Yolu
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from ayarlar.settings import PROCESSED_DATA_DIR
    DEFAULT_VERI_DIZINI = str(PROCESSED_DATA_DIR)
except Exception:
    DEFAULT_VERI_DIZINI = str(BASE_DIR / "veriler" / "processed" / "nytas_parkinson_veri")


# =========================================================================
# Klinik Referans Değerleri ve Eşikler (Literatür MDS-UPDRS Normları)
# =========================================================================
KLINIK_PARAMETRELER = [
    {
        "id": "kol_asimetri_ort",
        "ad": "Kol Salınım Asimetrisi",
        "ad_en": "Arm Swing Asymmetry",
        "birim": "%",
        "normal_aralik": "< %12.0",
        "risk_esik": 12.0,
        "risk_yonu": "buyuk",  # eşikten büyükse riskli
        "aciklama": "Erken evre Parkinson'da en karakteristik tek taraflı azalma belirtisi."
    },
    {
        "id": "yuruyus_hizi_cms",
        "ad": "Yürüyüş Hızı (Z-Derinlik)",
        "ad_en": "Gait Velocity",
        "birim": "cm/s",
        "normal_aralik": "90 - 140 cm/s",
        "risk_esik": 90.0,
        "risk_yonu": "kucuk",  # eşikten küçükse riskli
        "aciklama": "Bradikinezi ve adımlarda genel yavaşlama göstergesi."
    },
    {
        "id": "govde_egimi_ort",
        "ad": "Gövde Öne Eğimi (Sagittal)",
        "ad_en": "Trunk Flexion / Camptocormia",
        "birim": "°",
        "normal_aralik": "< 7.0°",
        "risk_esik": 7.0,
        "risk_yonu": "buyuk",
        "aciklama": "Postüral instabilite ve fleksör kas tonusu artışı (öne eğilme)."
    },
    {
        "id": "adim_uzunlugu_ort",
        "ad": "Adım Uzunluğu",
        "ad_en": "Stride Length",
        "birim": "cm",
        "normal_aralik": "55 - 85 cm",
        "risk_esik": 50.0,
        "risk_yonu": "kucuk",
        "aciklama": "Hipokinetik yürüyüş ve kısa adımlama (shuffling gait)."
    },
    {
        "id": "kadans_spm",
        "ad": "Kadans (Adım Sıklığı)",
        "ad_en": "Cadence",
        "birim": "adım/dk",
        "normal_aralik": "95 - 130 adım/dk",
        "risk_esik": 80.0,
        "risk_yonu": "kucuk",
        "aciklama": "Adım frekansında düzensizlik veya belirgin azalma."
    },
    {
        "id": "kol_genlik_ort",
        "ad": "Ortalama Kol Salınım Genliği",
        "ad_en": "Mean Arm Swing Amplitude",
        "birim": "cm",
        "normal_aralik": "15 - 35 cm",
        "risk_esik": 14.0,
        "risk_yonu": "kucuk",
        "aciklama": "Üst ekstremitede genel hareket kısıtlılığı."
    },
    {
        "id": "tremor_hz_max",
        "ad": "İstirahat Tremoru (3.5-7.5 Hz)",
        "ad_en": "Resting Hand Tremor",
        "birim": "Hz",
        "normal_aralik": "Tremor Yok",
        "risk_esik": 3.0,
        "risk_yonu": "buyuk",
        "aciklama": "Bilateral bilek ivmelenmesinde Parkinson bandı piki."
    },
    {
        "id": "fog_skoru",
        "ad": "Yürüyüş Donması (FOG)",
        "ad_en": "Freezing of Gait",
        "birim": "Skor",
        "normal_aralik": "Donma Yok (0.0)",
        "risk_esik": 0.4,
        "risk_yonu": "buyuk",
        "aciklama": "Ayakların yere yapışması ve adımlamanın kilitlenmesi."
    },
]


def _guvenli_float(deger, varsayilan=0.0) -> float:
    try:
        val = float(deger)
        return varsayilan if np.isnan(val) else val
    except (ValueError, TypeError):
        return varsayilan


def seans_verisi_yukle(seans_no: int, veri_dizini: str = DEFAULT_VERI_DIZINI) -> Tuple[Optional[pd.DataFrame], dict]:
    """
    Seansa ait verileri yükler. Önce seans klasöründeki seans_verileri.csv'ye bakar,
    yoksa ana parkinson_master.csv'den filtreler.
    """
    seans_kl = os.path.join(veri_dizini, f"seans_{seans_no:03d}")
    ozel_csv = os.path.join(seans_kl, "seans_verileri.csv")
    master_csv = os.path.join(veri_dizini, "parkinson_master.csv")

    df = None
    if os.path.exists(ozel_csv) and os.path.getsize(ozel_csv) > 100:
        try:
            df = pd.read_csv(ozel_csv)
        except Exception:
            df = None

    if df is None and os.path.exists(master_csv):
        try:
            df_master = pd.read_csv(master_csv)
            if "seans_no" in df_master.columns:
                df = df_master[df_master["seans_no"] == seans_no].copy()
        except Exception:
            df = None

    if df is None or df.empty:
        return None, {}

    # İlk kalibrasyon aşamasını (Veri Toplanıyor...) ayrı tutarak aktif kareleri al
    if "tahmin" in df.columns:
        df_aktif = df[~df["tahmin"].str.contains("Toplaniyor", na=False, case=False)].copy()
        if df_aktif.empty:
            df_aktif = df.copy()
    else:
        df_aktif = df.copy()

    # Seans meta bilgileri
    kullanici_id = str(df["kullanici_id"].iloc[0]) if "kullanici_id" in df.columns and not df.empty else "hasta_001"
    tarih_saat = str(df["tarih_saat"].iloc[0]) if "tarih_saat" in df.columns and not df.empty else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    toplam_kare = len(df)
    
    # Süre hesabı (timestamp varsa)
    sure_sn = 0.0
    if "timestamp" in df.columns and len(df) > 1:
        try:
            t0 = float(df["timestamp"].iloc[0])
            t1 = float(df["timestamp"].iloc[-1])
            sure_sn = max(0.0, t1 - t0)
        except Exception:
            sure_sn = toplam_kare / 30.0
    else:
        sure_sn = toplam_kare / 30.0

    # Kararlılık derecelendirmesi (MDS-UPDRS Yürüyüş Standardı: 30-40 sn ideal)
    if sure_sn < 15.0:
        kararlilik_derecesi = "Düşük (Kısa Seans - Öneri: ≥30s)"
        kararlilik_renk = "#dc2626"
        kararlilik_bg = "#fef2f2"
    elif sure_sn < 25.0:
        kararlilik_derecesi = "Kabul Edilebilir (%70 Kararlılık)"
        kararlilik_renk = "#d97706"
        kararlilik_bg = "#fffbeb"
    else:
        kararlilik_derecesi = "İdeal Kararlı Rejim (MDS Standardı)"
        kararlilik_renk = "#16a34a"
        kararlilik_bg = "#f0fdf4"

    meta = {
        "seans_no": seans_no,
        "kullanici_id": kullanici_id,
        "tarih_saat": tarih_saat,
        "toplam_kare": toplam_kare,
        "sure_sn": round(sure_sn, 1),
        "kararlilik_derecesi": kararlilik_derecesi,
        "kararlilik_renk": kararlilik_renk,
        "kararlilik_bg": kararlilik_bg,
        "seans_klasoru": seans_kl,
    }

    return df_aktif, meta


def biyobelirtec_analizi_yap(df: pd.DataFrame) -> Tuple[List[dict], dict]:
    """
    DataFrame üzerinden 8 biyobelirtecin ortalamalarını, risk durumlarını ve
    genel nihai teşhisi hesaplar.
    
    Kararlılık Optimizasyonu:
    İlk 2-3 saniyelik (yaklaşık 75 kare) ivmelenme ve kadraja alışma evresini
    istatistiklerden eler; böylece ortalamalar sadece kararlı (steady-state)
    yürüyüş evresine dayanır.
    """
    sonuclar = []
    toplam_riskli_sayisi = 0

    # İvmelenme / geçici rejim filtresi
    if len(df) > 150:
        df_hesap = df.iloc[75:].copy()
    else:
        df_hesap = df.copy()

    for param in KLINIK_PARAMETRELER:
        p_id = param["id"]
        ad = param["ad"]
        ad_en = param["ad_en"]
        birim = param["birim"]
        esik = param["risk_esik"]
        yon = param["risk_yonu"]

        # Değer hesaplama (ilgili sütun yoksa alternatiflere bak — df_hesap kullanılır)
        deger = 0.0
        if p_id in df_hesap.columns and not df_hesap[p_id].dropna().empty:
            deger = float(df_hesap[p_id].dropna().mean())
        elif p_id == "kol_asimetri_ort" and "kol_asimetri" in df_hesap.columns:
            deger = float(df_hesap["kol_asimetri"].dropna().mean())
        elif p_id == "adim_uzunlugu_ort" and "adim_uzunlugu_cm" in df_hesap.columns:
            deger = float(df_hesap["adim_uzunlugu_cm"].dropna().mean())
        elif p_id == "govde_egimi_ort" and "govde_egimi" in df_hesap.columns:
            deger = float(df_hesap["govde_egimi"].dropna().mean())
        elif p_id == "kol_genlik_ort" and "kol_genlik_min" in df_hesap.columns:
            deger = float(df_hesap["kol_genlik_min"].dropna().mean())
        elif p_id == "tremor_hz_max":
            t_col = "tremor_hz_max" if "tremor_hz_max" in df_hesap.columns else ("tremor_hz_L" if "tremor_hz_L" in df_hesap.columns else None)
            deger = float(df_hesap[t_col].dropna().max()) if t_col and not df_hesap[t_col].dropna().empty else 0.0

        # Risk kontrolü
        is_risk = False
        if yon == "buyuk":
            is_risk = (deger > esik)
        elif yon == "kucuk":
            is_risk = (0 < deger < esik)  # 0 ölçüm yoksa doğrudan risk sayma

        durum_metni = "Normal"
        durum_renk = "yesil"
        if is_risk:
            durum_metni = "YÜKSEK RİSK"
            durum_renk = "kirmizi"
            toplam_riskli_sayisi += 1
        else:
            # Sınıra yakınlık kontrolü (%15 tolerans)
            if yon == "buyuk" and deger > esik * 0.8:
                durum_metni = "Sınırda (Hafif)"
                durum_renk = "turuncu"
            elif yon == "kucuk" and deger < esik * 1.2:
                durum_metni = "Sınırda (Hafif)"
                durum_renk = "turuncu"

        sonuclar.append({
            "id": p_id,
            "ad": ad,
            "ad_en": ad_en,
            "birim": birim,
            "deger": round(deger, 2),
            "normal_aralik": param["normal_aralik"],
            "durum": durum_metni,
            "durum_renk": durum_renk,
            "aciklama": param["aciklama"],
            "is_risk": is_risk,
        })

    # Genel Seans Teşhisi ve Güven Skoru
    tahminler = df["tahmin"].value_counts(normalize=True) if "tahmin" in df.columns else {}
    parkinson_orani = float(tahminler.get("PARKINSON RISKI", 0.0))

    if "guven_pct" in df.columns:
        ortalama_guven = float(df["guven_pct"].dropna().mean())
    else:
        ortalama_guven = 85.0

    # Hibrit Karar: Model çoğunlukla Risk dediyse VEYA 3+ parametre klinik eşikteyse
    if parkinson_orani > 0.35 or toplam_riskli_sayisi >= 3:
        nihai_teshis = "PARKINSON RİSKİ TESPİT EDİLDİ"
        teshis_renk = "kirmizi"
        teshis_aciklama = (
            f"Hastanın yürüyüş biyomekaniğinde {toplam_riskli_sayisi}/8 kritik dijital biyobelirteç "
            "Parkinson klinik eşik aralığındadır. Nörolojik değerlendirme önerilir."
        )
    else:
        nihai_teshis = "NORMAL YÜRÜYÜŞ ÖRÜNTÜSÜ"
        teshis_renk = "yesil"
        teshis_aciklama = (
            f"Analiz edilen parametrelerin {8 - toplam_riskli_sayisi}/8'i sağlıklı kontrol referans "
            "aralıkları içerisindedir. Belirgin bir motor asimetri veya yavaşlama gözlenmemiştir."
        )

    genel_ozet = {
        "nihai_teshis": nihai_teshis,
        "teshis_renk": teshis_renk,
        "teshis_aciklama": teshis_aciklama,
        "guven_pct": round(ortalama_guven, 1),
        "klinik_risk_sayisi": toplam_riskli_sayisi,
        "parkinson_orani": round(parkinson_orani * 100, 1),
    }

    return sonuclar, genel_ozet


def radar_grafigi_ciz(biyobelirtecler: List[dict], cikti_yolu: str):
    """
    8 Biyobelirteç için normalize edilmiş Radar (Örümcek Ağı) grafiği çizer.
    Sağlıklı referans normu ile hastanın durumunu kıyaslar.
    """
    etiketler = [b["ad"].split("(")[0].strip() for b in biyobelirtecler]
    n_vars = len(etiketler)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]  # Çemberi kapat

    # Normalize değerler (0-100 arası ölçek)
    # Sağlıklı referans çizgisi: ortalama 50 puan
    norm_degerler = []
    for b in biyobelirtecler:
        deg = b["deger"]
        p_id = b["id"]
        # Parametreye göre 0-100 skorlama
        if p_id == "kol_asimetri_ort":
            skor = min(100, (deg / 30.0) * 100)
        elif p_id == "yuruyus_hizi_cms":
            skor = max(0, min(100, 100 - (deg / 130.0) * 60))
        elif p_id == "govde_egimi_ort":
            skor = min(100, (deg / 20.0) * 100)
        elif p_id == "adim_uzunlugu_ort":
            skor = max(0, min(100, 100 - (deg / 75.0) * 60))
        elif p_id == "kadans_spm":
            skor = max(0, min(100, 100 - (deg / 120.0) * 60))
        elif p_id == "kol_genlik_ort":
            skor = max(0, min(100, 100 - (deg / 25.0) * 60))
        elif p_id == "tremor_hz_max":
            skor = min(100, (deg / 6.0) * 100) if deg > 0 else 10
        elif p_id == "fog_skoru":
            skor = min(100, deg * 100)
        else:
            skor = 50
        norm_degerler.append(round(skor, 1))

    norm_degerler += norm_degerler[:1]
    referans_degerler = [35] * n_vars + [35]  # Sağlıklı kontrol alanı

    fig, ax = plt.subplots(figsize=(6.5, 6.0), subplot_kw=dict(polar=True), dpi=130)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    # Açı ve etiket ayarları
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(etiketler, fontsize=8.5, fontweight="bold", color="#1e293b")
    ax.set_rlabel_position(0)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["%25", "%50 (Eşik)", "%75", "%100"], fontsize=7, color="#64748b")
    ax.set_ylim(0, 100)
    ax.grid(color="#cbd5e1", linestyle="--", linewidth=0.7)

    # Sağlıklı Referans Bölgesi (Yeşil Gölgeli)
    ax.plot(angles, referans_degerler, color="#10b981", linewidth=1.8, linestyle="--", label="Sağlıklı Referans")
    ax.fill(angles, referans_degerler, color="#10b981", alpha=0.15)

    # Hasta Değerleri (Kırmızı/Mavi Alan)
    ax.plot(angles, norm_degerler, color="#ef4444", linewidth=2.2, label="Hasta Biyomekanik Ölçümü")
    ax.fill(angles, norm_degerler, color="#ef4444", alpha=0.25)
    ax.scatter(angles, norm_degerler, color="#b91c1c", s=30, zorder=5)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8, frameon=True)
    plt.title("Parkinson Biyobelirteç Risk Profili (Radar Analizi)", fontsize=11, fontweight="bold", color="#0f172a", pad=20)
    plt.tight_layout()
    plt.savefig(cikti_yolu, format="png", bbox_inches="tight")
    plt.close(fig)


def zaman_serisi_grafigi_ciz(df: pd.DataFrame, cikti_yolu: str):
    """
    Yürüyüş Hızı, Kol Salınımı Sol/Sağ ve Gövde Açısının seans boyunca
    nasıl değiştiğini gösteren 3 panelli zaman serisi grafiği çizer.
    """
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 6.2), dpi=130, sharex=True)
    fig.patch.set_facecolor("#ffffff")

    kareler = df["frame"].values if "frame" in df.columns else np.arange(len(df))

    # 1. Panel: Yürüyüş Hızı
    ax1 = axes[0]
    ax1.set_facecolor("#f8fafc")
    hiz_col = "yuruyus_hizi_cms" if "yuruyus_hizi_cms" in df.columns else None
    if hiz_col:
        ax1.plot(kareler, df[hiz_col].values, color="#0284c7", linewidth=1.8, label="Yürüyüş Hızı (cm/s)")
        ax1.axhline(90.0, color="#ef4444", linestyle="--", linewidth=1.2, label="Klinik Eşik (90 cm/s)")
    ax1.set_ylabel("Hız (cm/s)", fontsize=8, fontweight="bold", color="#1e293b")
    ax1.legend(loc="upper right", fontsize=7.5)
    ax1.grid(color="#e2e8f0", linestyle=":")

    # 2. Panel: Sol vs Sağ Kol Salınım Genliği (Asimetri Göstergesi)
    ax2 = axes[1]
    ax2.set_facecolor("#f8fafc")
    l_col = "kol_genlik_L_ort" if "kol_genlik_L_ort" in df.columns else ("arm_rel_L" if "arm_rel_L" in df.columns else None)
    r_col = "kol_genlik_R_ort" if "kol_genlik_R_ort" in df.columns else ("arm_rel_R" if "arm_rel_R" in df.columns else None)
    if l_col and r_col:
        ax2.plot(kareler, df[l_col].values, color="#8b5cf6", linewidth=1.5, label="Sol Kol")
        ax2.plot(kareler, df[r_col].values, color="#f59e0b", linewidth=1.5, label="Sağ Kol")
    ax2.set_ylabel("Kol Genlik (cm)", fontsize=8, fontweight="bold", color="#1e293b")
    ax2.legend(loc="upper right", fontsize=7.5)
    ax2.grid(color="#e2e8f0", linestyle=":")

    # 3. Panel: Gövde Eğimi (Sagittal Öne Eğilme Açısı)
    ax3 = axes[2]
    ax3.set_facecolor("#f8fafc")
    govde_col = "govde_egimi_ort" if "govde_egimi_ort" in df.columns else ("govde_egimi" if "govde_egimi" in df.columns else None)
    if govde_col:
        ax3.plot(kareler, df[govde_col].values, color="#059669", linewidth=1.8, label="Gövde Öne Eğimi (°)")
        ax3.axhline(7.0, color="#ef4444", linestyle="--", linewidth=1.2, label="Klinik Eşik (7.0°)")
    ax3.set_ylabel("Gövde Açısı (°)", fontsize=8, fontweight="bold", color="#1e293b")
    ax3.set_xlabel("Seans Süresi (Video Kare Sayısı)", fontsize=8.5, fontweight="bold", color="#1e293b")
    ax3.legend(loc="upper right", fontsize=7.5)
    ax3.grid(color="#e2e8f0", linestyle=":")

    fig.suptitle("Yürüyüş Dinamikleri & Biyomekanik Zaman Serisi", fontsize=11, fontweight="bold", color="#0f172a", y=0.98)
    plt.tight_layout()
    plt.savefig(cikti_yolu, format="png", bbox_inches="tight")
    plt.close(fig)


def html_rapor_derle(meta: dict, genel_ozet: dict, biyobelirtecler: List[dict],
                      radar_yolu: str, zaman_yolu: str, cikti_html: str) -> str:
    """
    Tam teşekküllü, modern ve medikal standartlarda HTML/PDF raporu derler.
    Grafikler sayfa içine doğrudan base64 veya dosya bağlantısı olarak gömülür.
    """
    # Grafikleri base64'e dönüştür (bağımsız tek dosya rapor için)
    radar_b64 = ""
    if os.path.exists(radar_yolu):
        with open(radar_yolu, "rb") as f:
            radar_b64 = base64.b64encode(f.read()).decode("utf-8")

    zaman_b64 = ""
    if os.path.exists(zaman_yolu):
        with open(zaman_yolu, "rb") as f:
            zaman_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Renk temaları
    teshis_renk_kodu = "#dc2626" if genel_ozet["teshis_renk"] == "kirmizi" else "#16a34a"
    teshis_bg_kodu = "#fef2f2" if genel_ozet["teshis_renk"] == "kirmizi" else "#f0fdf4"
    teshis_border = "#f87171" if genel_ozet["teshis_renk"] == "kirmizi" else "#86efac"

    # Tablo satırları
    tablo_satirlari_html = ""
    for b in biyobelirtecler:
        renk_css = "#dc2626" if b["durum_renk"] == "kirmizi" else ("#d97706" if b["durum_renk"] == "turuncu" else "#16a34a")
        bg_css = "#fef2f2" if b["durum_renk"] == "kirmizi" else ("#fffbeb" if b["durum_renk"] == "turuncu" else "#f0fdf4")
        badge_icon = "⚠️" if b["durum_renk"] == "kirmizi" else ("⚡" if b["durum_renk"] == "turuncu" else "✓")

        tablo_satirlari_html += f"""
        <tr>
            <td style="font-weight: 600; color: #1e293b;">
                {b['ad']}
                <div style="font-size: 10px; color: #64748b; font-weight: normal;">{b['ad_en']}</div>
            </td>
            <td style="font-size: 13px; font-weight: 700; color: #0f172a; text-align: right;">
                {b['deger']} {b['birim']}
            </td>
            <td style="color: #475569; text-align: center; font-size: 11px;">
                {b['normal_aralik']}
            </td>
            <td style="text-align: center;">
                <span style="background: {bg_css}; color: {renk_css}; padding: 3px 8px; border-radius: 9999px; font-weight: 700; font-size: 11px; display: inline-block; border: 1px solid {renk_css}40;">
                    {badge_icon} {b['durum']}
                </span>
            </td>
            <td style="font-size: 11px; color: #475569;">
                {b['aciklama']}
            </td>
        </tr>
        """

    html_icerik = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>NYTAS-PARKINSON Klinik Analiz Raporu | Seans #{meta['seans_no']:03d}</title>
    <style>
        @page {{
            size: A4;
            margin: 12mm 15mm;
        }}
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
            background-color: #f1f5f9;
            color: #0f172a;
            margin: 0;
            padding: 20px;
            font-size: 12px;
            line-height: 1.4;
        }}
        .rapor-kapsayici {{
            max-width: 900px;
            margin: 0 auto;
            background: #ffffff;
            padding: 32px 40px;
            border-radius: 8px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        }}
        .ust-banner {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #0284c7;
            padding-bottom: 14px;
            margin-bottom: 20px;
        }}
        .logo-alan h1 {{
            margin: 0;
            font-size: 20px;
            color: #0284c7;
            letter-spacing: -0.5px;
        }}
        .logo-alan p {{
            margin: 3px 0 0 0;
            font-size: 11px;
            color: #64748b;
        }}
        .seans-rozet {{
            text-align: right;
            font-size: 11px;
            color: #334155;
        }}
        .hasta-kart {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 12px 18px;
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px;
            margin-bottom: 18px;
        }}
        .hasta-kart .bilgi-ogesi span {{
            display: block;
            font-size: 10px;
            color: #64748b;
            text-transform: uppercase;
            font-weight: 600;
        }}
        .hasta-kart .bilgi-ogesi strong {{
            font-size: 13px;
            color: #0f172a;
        }}
        .teshis-kutusu {{
            background: {teshis_bg_kodu};
            border: 1.5px solid {teshis_border};
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 22px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .teshis-kutusu h2 {{
            margin: 0;
            font-size: 17px;
            color: {teshis_renk_kodu};
        }}
        .teshis-kutusu p {{
            margin: 4px 0 0 0;
            font-size: 12px;
            color: #334155;
        }}
        .skor-kutusu {{
            text-align: right;
            min-width: 140px;
        }}
        .skor-kutusu .skor-sayi {{
            font-size: 22px;
            font-weight: 800;
            color: {teshis_renk_kodu};
        }}
        .skor-kutusu .skor-etiket {{
            font-size: 10px;
            color: #64748b;
            text-transform: uppercase;
        }}
        table.medikal-tablo {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 24px;
        }}
        table.medikal-tablo th {{
            background: #f1f5f9;
            color: #334155;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 8px 10px;
            border-top: 1px solid #cbd5e1;
            border-bottom: 1px solid #cbd5e1;
            text-align: left;
        }}
        table.medikal-tablo td {{
            padding: 9px 10px;
            border-bottom: 1px solid #f1f5f9;
            vertical-align: middle;
        }}
        .grafikler-kutusu {{
            display: flex;
            gap: 16px;
            margin-bottom: 24px;
            page-break-inside: avoid;
        }}
        .grafik-karti {{
            flex: 1;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 10px;
            text-align: center;
        }}
        .grafik-karti img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .imza-alani {{
            display: flex;
            justify-content: space-between;
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #e2e8f0;
            font-size: 11px;
            color: #475569;
            page-break-inside: avoid;
        }}
        .imza-kutu {{
            width: 220px;
            text-align: center;
            border-top: 1px dashed #94a3b8;
            padding-top: 6px;
            margin-top: 35px;
        }}
        .yazdir-butonu {{
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: #0284c7;
            color: #ffffff;
            padding: 12px 20px;
            border-radius: 50px;
            font-weight: 700;
            font-size: 13px;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4);
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
        }}
        .yazdir-butonu:hover {{
            background: #0369a1;
            transform: translateY(-2px);
        }}
        @media print {{
            body {{
                background: #ffffff;
                padding: 0;
            }}
            .rapor-kapsayici {{
                box-shadow: none;
                padding: 0;
                max-width: 100%;
            }}
            .yazdir-butonu {{
                display: none !important;
            }}
        }}
    </style>
</head>
<body>

    <button class="yazdir-butonu" onclick="window.print()">
        🖨️ Raporu Yazdır / PDF Kaydet
    </button>

    <div class="rapor-kapsayici">
        <!-- Üst Başlık -->
        <div class="ust-banner">
            <div class="logo-alan">
                <h1>🧬 NYTAS-PARKINSON™</h1>
                <p>Parkinson Hastalığı Dijital Biyobelirteç & Yürüyüş Analiz Sistemi | TÜBİTAK 1002 Hızlı Destek Programı</p>
            </div>
            <div class="seans-rozet">
                <strong style="font-size: 14px; color: #0284c7;">KLİNİK SEANS RAPORU</strong>
                <div>Seans Kodu: <strong>#SEANS-{meta['seans_no']:03d}</strong></div>
                <div>Rapor Tarihi: {meta['tarih_saat']}</div>
            </div>
        </div>

        <!-- Hasta & Ölçüm Bilgileri Kartı -->
        <div class="hasta-kart">
            <div class="bilgi-ogesi">
                <span>Hasta / Kullanıcı ID</span>
                <strong>{meta['kullanici_id']}</strong>
            </div>
            <div class="bilgi-ogesi">
                <span>Test Tarihi & Saati</span>
                <strong>{meta['tarih_saat']}</strong>
            </div>
            <div class="bilgi-ogesi">
                <span>Seans Süresi</span>
                <strong>{meta['sure_sn']} Saniye</strong>
            </div>
            <div class="bilgi-ogesi">
                <span>İşlenen Kare Sayısı</span>
                <strong>{meta['toplam_kare']} Kare (~30 FPS)</strong>
            </div>
            <div class="bilgi-ogesi">
                <span>Veri Kararlılık Düzeyi</span>
                <strong style="color: {meta['kararlilik_renk']}; font-size: 11px;">{meta['kararlilik_derecesi']}</strong>
            </div>
        </div>

        <!-- Teşhis Özeti Kutusu -->
        <div class="teshis-kutusu">
            <div>
                <h2>{genel_ozet['nihai_teshis']}</h2>
                <p>{genel_ozet['teshis_aciklama']}</p>
            </div>
            <div class="skor-kutusu">
                <div class="skor-sayi">%{genel_ozet['guven_pct']}</div>
                <div class="skor-etiket">ML Model Güveni</div>
                <div style="font-size: 11px; margin-top: 4px; font-weight: 700; color: #475569;">
                    Risk Skoru: {genel_ozet['klinik_risk_sayisi']}/8
                </div>
            </div>
        </div>

        <!-- 8 Biyobelirteç Klinik Tablosu -->
        <h3 style="font-size: 13px; color: #1e293b; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">
            📋 8 Temel Dijital Biyobelirteç Değerlendirmesi
        </h3>
        <table class="medikal-tablo">
            <thead>
                <tr>
                    <th style="width: 25%;">Biyobelirteç Parametresi</th>
                    <th style="width: 15%; text-align: right;">Ölçülen Ortalama</th>
                    <th style="width: 18%; text-align: center;">Klinik Norm Aralığı</th>
                    <th style="width: 18%; text-align: center;">Durum</th>
                    <th style="width: 24%;">Klinik Yorum / Anlam</th>
                </tr>
            </thead>
            <tbody>
                {tablo_satirlari_html}
            </tbody>
        </table>

        <!-- Klinik Grafikler (Radar ve Zaman Serisi) -->
        <div class="grafikler-kutusu">
            <div class="grafik-karti" style="flex: 0.9;">
                <h4 style="margin: 0 0 6px 0; font-size: 11px; color: #334155;">Risk Profili (Radar Analizi)</h4>
                {f'<img src="data:image/png;base64,{radar_b64}" alt="Radar Grafiği">' if radar_b64 else '<p>Grafik yüklenemedi</p>'}
            </div>
            <div class="grafik-karti" style="flex: 1.3;">
                <h4 style="margin: 0 0 6px 0; font-size: 11px; color: #334155;">Zaman Serisi (Hız, Kol Salınımı, Gövde Açısı)</h4>
                {f'<img src="data:image/png;base64,{zaman_b64}" alt="Zaman Serisi Grafiği">' if zaman_b64 else '<p>Grafik yüklenemedi</p>'}
            </div>
        </div>

        <!-- Hekim Notu ve İmza Alanı -->
        <div class="imza-alani">
            <div style="max-width: 450px;">
                <strong>Hekim / Araştırmacı Notları:</strong>
                <p style="margin: 4px 0 0 0; color: #64748b; font-size: 10px;">
                    Bu rapor, yapay zeka tabanlı temassız biyomekanik tarama sonuçlarını içermektedir.
                    Kesin tanı için nörolojik klinik muayene, MDS-UPDRS skorlaması ve gerektiğinde medikal görüntüleme ile desteklenmelidir.
                </p>
            </div>
            <div class="imza-kutu">
                Uzm. Dr. / Nörolog<br>
                Kaşe & İmza
            </div>
        </div>
    </div>

</body>
</html>
"""
    with open(cikti_html, "w", encoding="utf-8") as f:
        f.write(html_icerik)

    return cikti_html


def rapor_olustur(seans_no: int, veri_dizini: str = DEFAULT_VERI_DIZINI, otomatik_ac: bool = True) -> Optional[str]:
    """
    Belirtilen seans numarası için tüm analizi yapar, grafikleri çizer,
    HTML raporu derler ve isteğe bağlı olarak tarayıcıda açar.
    """
    df, meta = seans_verisi_yukle(seans_no, veri_dizini)
    if df is None or df.empty:
        print(f"[UYARI] Seans #{seans_no} için işlenebilir veri bulunamadı!")
        return None

    seans_kl = meta["seans_klasoru"]
    os.makedirs(seans_kl, exist_ok=True)

    # 1. Biyobelirteç İstatistiklerini Çıkar
    biyobelirtecler, genel_ozet = biyobelirtec_analizi_yap(df)

    # 2. Grafikleri Üret
    radar_yolu = os.path.join(seans_kl, "radar_biyobelirtec.png")
    zaman_yolu = os.path.join(seans_kl, "zaman_serisi_yuruyus.png")
    html_yolu = os.path.join(seans_kl, f"klinik_rapor_seans_{seans_no:03d}.html")

    try:
        radar_grafigi_ciz(biyobelirtecler, radar_yolu)
    except Exception as e:
        print(f"[HATA] Radar grafiği çizilemedi: {e}")

    try:
        zaman_serisi_grafigi_ciz(df, zaman_yolu)
    except Exception as e:
        print(f"[HATA] Zaman serisi grafiği çizilemedi: {e}")

    # 3. HTML Raporunu Oluştur
    try:
        html_rapor_derle(meta, genel_ozet, biyobelirtecler, radar_yolu, zaman_yolu, html_yolu)
        print(f"[BAŞARILI] Klinik Rapor Oluşturuldu -> {html_yolu}")
    except Exception as e:
        print(f"[HATA] HTML raporu derlenemedi: {e}")
        return None

    # 4. Tarayıcıda Aç
    if otomatik_ac and os.path.exists(html_yolu):
        try:
            webbrowser.open(f"file:///{os.path.abspath(html_yolu)}")
        except Exception:
            pass

    return html_yolu


def tum_seanslari_raporla(veri_dizini: str = DEFAULT_VERI_DIZINI) -> List[str]:
    """
    Mevcut veriler altındaki tüm seanslar için toplu olarak klinik rapor üretir.
    """
    master_csv = os.path.join(veri_dizini, "parkinson_master.csv")
    uretilenler = []
    if not os.path.exists(master_csv):
        print("[HATA] Master CSV bulunamadı!")
        return uretilenler

    try:
        df = pd.read_csv(master_csv, usecols=["seans_no"])
        seanslar = sorted(df["seans_no"].dropna().unique())
    except Exception as e:
        print(f"[HATA] Seanslar okunamadı: {e}")
        return uretilenler

    for s_no in seanslar:
        try:
            s_int = int(s_no)
            p = rapor_olustur(s_int, veri_dizini, otomatik_ac=False)
            if p:
                uretilenler.append(p)
        except Exception as e:
            print(f"[HATA] Seans #{s_no} raporlanırken hata: {e}")

    print(f"\n[TAMAMLANDI] Toplam {len(uretilenler)} seans için klinik rapor üretildi.")
    return uretilenler


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        rapor_olustur(int(sys.argv[1]), otomatik_ac=True)
    elif "--hepsi" in sys.argv:
        tum_seanslari_raporla()
    else:
        print("Kullanım:")
        print("  python rapor_olusturucu.py [seans_no]   -> Belirli seansın raporunu üret ve aç")
        print("  python rapor_olusturucu.py --hepsi     -> Tüm geçmiş seansların raporlarını üret")
