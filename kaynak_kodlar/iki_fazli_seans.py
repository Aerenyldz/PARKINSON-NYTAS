"""
=============================================================================
NYTAS-PARKINSON — İki Fazlı Otomatik Klinik Seans Modülü
=============================================================================
Açıklama:
  Hekim ve hasta için tam yönergeli iki fazlı klinik test motoru:

  FAZ 1 — İSTİRAHAT TREMOR TESTİ (varsayılan: 45 saniye)
    • Hasta hareketsiz oturur/ayakta durur
    • Sadece tremor FFT + FOG verisi anlamlı
    • Yönergeli görsel sayaç + uyarı bandı

  FAZ 2 — YÜRÜYÜŞ ANALİZ TESTİ (varsayılan: 60 saniye)
    • Hasta kamera önünde yürür
    • Tüm 8 biyobelirteç işlenir
    • Süre dolunca otomatik kayıt + klinik rapor

Kullanım (CLI):
  python kaynak_kodlar/iki_fazli_seans.py --id hasta_001 --boy 175 --kamera 0
  python kaynak_kodlar/iki_fazli_seans.py --id hasta_001 --boy 175 --kamera 0 --faz1-sure 45 --faz2-sure 60
=============================================================================
"""

import argparse
import os
import sys
import time
import math
import logging
import threading
import urllib.request
import csv
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk

# Proje kök yolu
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from ayarlar.settings import POSE_LANDMARKER_PATH, PARKINSON_MODEL_PATH, PROCESSED_DATA_DIR
    MODEL_DOSYASI = str(POSE_LANDMARKER_PATH)
    DEFAULT_ML_MODEL = str(PARKINSON_MODEL_PATH)
    DEFAULT_VERI_DIZINI = str(PROCESSED_DATA_DIR)
except Exception:
    MODEL_DOSYASI = str(BASE_DIR / "modeller" / "pose_landmarker.task")
    DEFAULT_ML_MODEL = str(BASE_DIR / "modeller" / "parkinson_model.pkl")
    DEFAULT_VERI_DIZINI = str(BASE_DIR / "veriler" / "processed" / "nytas_parkinson_veri")

# Mevcut analiz bileşenlerini içe aktar (çakışma yok — ayrı process)
from kaynak_kodlar.nytas_parkinson import (
    Config, VideoStream, ParkinsonAnalizor, ParkinsonTemporal,
    VeriLogger, ParkinsonHUD, skeleton_ciz, yuz_gizle,
    model_hazirla, _sonraki_seans_no, _log_kur,
    POSE_CONNECTIONS, MODEL_URL
)


# =========================================================================
# İki Fazlı Seans Konfigürasyonu
# =========================================================================
@dataclass
class IkiFazliConfig:
    # Hasta bilgileri
    KULLANICI_ID: str = "hasta_001"
    KULLANICI_BOYU_CM: float = 175.0
    KAMERA_KAYNAK: object = 0
    KAMERA_DONDUR: bool = False
    YUZU_GIZLE: bool = True
    KLINIK_ETIKET: int = -1

    # Faz süreleri
    FAZ1_SURE_SN: float = 45.0    # İstirahat tremor testi süresi
    FAZ2_SURE_SN: float = 60.0    # Yürüyüş analizi süresi

    # Teknik
    FRAME_GENISLIK: int = 640
    FRAME_YUKSEKLIK: int = 480
    MIN_DETECTION_CONF: float = 0.6
    MIN_TRACKING_CONF: float = 0.6
    BUFFER_BOYUTU: int = 120
    VERI_DIZINI: str = DEFAULT_VERI_DIZINI
    VIDEO_KAYDET: bool = True
    CSV_KAYDET: bool = True
    ML_MODEL_DOSYASI: str = DEFAULT_ML_MODEL
    KAMERA_FPS: int = 30
    KAMERA_COZUNURLUK_W: int = 640
    KAMERA_COZUNURLUK_H: int = 480
    KAMERA_BUFFER: int = 1


# =========================================================================
# İki Fazlı HUD Overlay
# =========================================================================
class IkiFazliHUD:
    """İki fazlı seans için özel HUD bileşeni."""

    # Renkler (BGR)
    CYAN    = (200, 200, 0)
    YESIL   = (50, 220, 50)
    KIRMIZI = (50, 50, 220)
    SARI    = (0, 200, 255)
    TURUNCU = (0, 140, 255)
    MOR     = (200, 50, 200)
    BEYAZ   = (255, 255, 255)
    GRI     = (140, 140, 140)
    KOYU    = (18, 18, 18)
    LACIVERT = (80, 30, 10)
    ACIK_MAVI = (220, 180, 50)

    @staticmethod
    def faz_bantı_ciz(img: np.ndarray, faz: int, gecen: float, toplam: float,
                       seans_no: int, hasta_id: str) -> np.ndarray:
        """Üst bant: Faz göstergesi + sayaç + ilerleme çubuğu."""
        h, w = img.shape[:2]
        C = IkiFazliHUD

        # Üst arka plan
        cv2.rectangle(img, (0, 0), (w, 110), C.KOYU, -1)

        # Faz göstergesi
        if faz == 1:
            faz_renk = C.MOR
            faz_metin = "FAZ 1 / 2 — ISTIRAHAT TREMOR TESTI"
            faz_ikon  = "[ HAREKETSIZ KALIN ]"
        else:
            faz_renk = C.ACIK_MAVI
            faz_metin = "FAZ 2 / 2 — YURUYUS ANALIZ TESTI"
            faz_ikon  = "[ KAMERAYA DIK YURUYIN ]"

        cv2.putText(img, "NYTAS-PARKINSON | IKI FAZLI KLINIK SEANS",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C.CYAN, 1)
        cv2.putText(img, faz_metin,
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, faz_renk, 1)
        cv2.putText(img, f"Seans #{seans_no:03d} | {hasta_id} | {faz_ikon}",
                    (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C.BEYAZ, 1)

        # Geri sayaç
        kalan = max(0.0, toplam - gecen)
        pct   = min(100, int((gecen / toplam) * 100)) if toplam > 0 else 100
        sayac_renk = C.KIRMIZI if kalan < 10 else (C.SARI if kalan < 20 else C.YESIL)
        cv2.putText(img, f"Kalan: {kalan:.0f}s  ({pct}%)",
                    (w - 190, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, sayac_renk, 1)

        # İlerleme çubuğu
        cv2.rectangle(img, (0, 106), (w, 110), (35, 35, 35), -1)
        bar_w = int((w * pct) / 100)
        cv2.rectangle(img, (0, 106), (bar_w, 110), faz_renk, -1)

        return img

    @staticmethod
    def yonerge_bandı_ciz(img: np.ndarray, faz: int,
                           gecen: float, toplam: float) -> np.ndarray:
        """Alt yönerge bandı: Hasta ve hekim için talimatlar."""
        h, w = img.shape[:2]
        C = IkiFazliHUD

        cv2.rectangle(img, (0, h - 55), (w, h), C.KOYU, -1)

        if faz == 1:
            kalan = max(0.0, toplam - gecen)
            if kalan > 30:
                metin = "YONERGE: Ellerinizi dizinize koyun. Hareketsiz, rahat ve duz durun."
            elif kalan > 15:
                metin = "YONERGE: Hareketsiz kalmaya devam edin. Tremor olcumuyor..."
            else:
                metin = f"FAZ 1 {kalan:.0f}s'de bitiyor. Yuruyuse hazirlanin!"
        else:
            kalan = max(0.0, toplam - gecen)
            if kalan > 40:
                metin = "YONERGE: Dogal yuruyusunuzle kameraya dogru ilerleyin, kollarinizi serbest birakın."
            elif kalan > 20:
                metin = "YONERGE: Yuruyusunuzu surdurun. Veri kararlasıyor..."
            else:
                metin = f"YONERGE: {kalan:.0f}s kaldi. Tamamlanıyor, surdurun..."

        cv2.putText(img, metin, (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C.SARI, 1)
        cv2.putText(img, "Cikis: Q tus | Faz Atla: SPACE",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C.GRI, 1)
        return img

    @staticmethod
    def faz_gecis_overlay(img: np.ndarray, sn_kaldi: float) -> np.ndarray:
        """Faz geçiş animasyonu (3 saniyelik ara ekran)."""
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 40), -1)
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        cv2.putText(img, "FAZ 1 TAMAMLANDI!",
                    (w // 2 - 180, h // 2 - 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (50, 220, 50), 2)
        cv2.putText(img, "SIMDI YURUYUSE BASLAYIN",
                    (w // 2 - 200, h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 1)
        cv2.putText(img, f"FAZ 2 {sn_kaldi:.0f}s'de basliyor...",
                    (w // 2 - 160, h // 2 + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 140, 50), 1)
        return img


# =========================================================================
# İki Fazlı Ana Analiz Döngüsü
# =========================================================================
def iki_fazli_analiz_baslat(cfg: IkiFazliConfig):
    """İki fazlı klinik seans akışını başlatır ve yönetir."""

    seans_no = _sonraki_seans_no(cfg.VERI_DIZINI)
    log_kl   = os.path.join(cfg.VERI_DIZINI, f"seans_{seans_no:03d}")
    log      = _log_kur(seans_no, log_kl)

    log.info("=" * 60)
    log.info("  NYTAS-PARKINSON | IKI FAZLI KLINIK SEANS")
    log.info("=" * 60)
    log.info(f"  Hasta       : {cfg.KULLANICI_ID}")
    log.info(f"  Boy         : {cfg.KULLANICI_BOYU_CM} cm")
    log.info(f"  Kamera      : {cfg.KAMERA_KAYNAK}")
    log.info(f"  Faz1 Sure   : {cfg.FAZ1_SURE_SN}s (Istirahat Tremor)")
    log.info(f"  Faz2 Sure   : {cfg.FAZ2_SURE_SN}s (Yuruyus Analiz)")
    log.info(f"  Seans       : #{seans_no:03d}")
    log.info(f"  Tarih       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    # MediaPipe model hazırlığı
    if not model_hazirla(log):
        messagebox.showerror("Model Hatası",
                             f"MediaPipe modeli indirilemedi:\n{MODEL_DOSYASI}")
        return

    # ML modeli
    ml_model = None
    try:
        ml_model = joblib.load(cfg.ML_MODEL_DOSYASI)
        log.info(f"ML model yuklendi: {type(ml_model).__name__}")
    except FileNotFoundError:
        log.warning("ML model bulunamadi -> sadece veri toplanacak")
    except Exception as e:
        log.error(f"ML model hatasi: {e}")

    # Kamera / VideoStream
    # Config nesnesini geçici olarak global CFG ile senkronize et
    import kaynak_kodlar.nytas_parkinson as nytas_mod
    nytas_mod.CFG.KAMERA_BUFFER      = cfg.KAMERA_BUFFER
    nytas_mod.CFG.KAMERA_FPS         = cfg.KAMERA_FPS
    nytas_mod.CFG.KAMERA_COZUNURLUK_W = cfg.KAMERA_COZUNURLUK_W
    nytas_mod.CFG.KAMERA_COZUNURLUK_H = cfg.KAMERA_COZUNURLUK_H

    try:
        cap = VideoStream(cfg.KAMERA_KAYNAK).start()
    except Exception as e:
        log.error(f"Kamera baslatilamadi: {e}")
        messagebox.showerror("Kamera Hatası", f"Kamera akışı başlatılamadı:\n{e}")
        return

    # Kamera hazır mı?
    kamera_ok = False
    for _ in range(40):
        time.sleep(0.1)
        if cap.read() is not None:
            kamera_ok = True
            break
    if not kamera_ok:
        cap.stop()
        messagebox.showerror("Kamera Hatası",
                             f"Kamera görüntüsü alınamadı!\nKaynak: {cfg.KAMERA_KAYNAK}")
        return

    # Analiz bileşenleri — FAZ başına sıfırlanacak
    def yeni_bilesenler():
        nytas_mod.CFG.KULLANICI_BOYU_CM = cfg.KULLANICI_BOYU_CM
        nytas_mod.CFG.YUZU_GIZLE = cfg.YUZU_GIZLE
        return (
            ParkinsonAnalizor(),
            ParkinsonTemporal(buf=cfg.BUFFER_BOYUTU, fps=cfg.KAMERA_FPS),
        )

    # VeriLogger (tek seans, iki faz aynı CSV'ye yazılır)
    veri_logger = VeriLogger(seans_no, cfg.VERI_DIZINI, cfg.VIDEO_KAYDET)

    # MediaPipe PoseLandmarker
    with open(MODEL_DOSYASI, "rb") as f:
        model_buffer = f.read()
    base_opts = mp.tasks.BaseOptions(model_asset_buffer=model_buffer)
    pose_opts = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_pose_detection_confidence=cfg.MIN_DETECTION_CONF,
        min_pose_presence_confidence=cfg.MIN_TRACKING_CONF,
        num_poses=1,
    )

    # ---- Seans Döngüsü ----
    faz = 1
    faz1_bitis = None          # Faz 1'in bitiş zamanı (gecis animasyonu için)
    faz_baslangic = time.time()
    seans_baslangic = time.time()

    analizor, temporal = yeni_bilesenler()

    tahmin_sonucu = "Veri Toplaniyor..."
    tahmin_guveni = 0.0
    uyari = ""
    fps_sayaci, fps_ref, anlik_fps = 0, time.time(), 0.0
    frame_ts_ms = 0

    # Faz 1'e özel tremor özeti (rapor için)
    faz1_tremor_ozet: Dict = {}

    GECIS_SURESI = 4.0  # saniye, geçiş animasyonu

    with mp.tasks.vision.PoseLandmarker.create_from_options(pose_opts) as landmarker:
        while True:
            frame = cap.read()
            if frame is None:
                time.sleep(0.01)
                continue

            if cfg.KAMERA_DONDUR:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (cfg.FRAME_GENISLIK, cfg.FRAME_YUKSEKLIK))
            h, w = frame.shape[:2]

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts_ms += 1

            try:
                results = landmarker.detect_for_video(mp_img, frame_ts_ms)
            except Exception as e:
                log.error(f"Pose hatasi: {e}")
                continue

            now_t        = time.time()
            gecen        = now_t - faz_baslangic
            toplam_sure  = cfg.FAZ1_SURE_SN if faz == 1 else cfg.FAZ2_SURE_SN
            gecen_global = now_t - seans_baslangic

            img = frame.copy()
            fv: Optional[dict]  = None
            tm: Optional[dict]  = None

            # ---- FAZ GEÇİŞ ANİMASYONU ----
            if faz == 1 and faz1_bitis is not None:
                kalan_gecis = GECIS_SURESI - (now_t - faz1_bitis)
                if kalan_gecis > 0:
                    img = IkiFazliHUD.faz_gecis_overlay(img, kalan_gecis)
                    cv2.imshow(f"NYTAS-Parkinson | İki Fazlı Seans #{seans_no:03d}", img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    continue
                else:
                    # Faz 2'ye geç
                    faz = 2
                    faz1_bitis = None
                    faz_baslangic = time.time()
                    analizor, temporal = yeni_bilesenler()
                    log.info("FAZ 2 baslatildi: Yuruyus Analiz Testi")
                    continue

            # ---- FAZ 1 BİTİŞ KONTROLÜ ----
            if faz == 1 and gecen >= cfg.FAZ1_SURE_SN:
                # Tremor özetini kaydet
                m = temporal.metrikler()
                faz1_tremor_ozet = {
                    "tremor_hz_L":   m.get("tremor_hz_L", 0.0),
                    "tremor_hz_R":   m.get("tremor_hz_R", 0.0),
                    "tremor_hz_max": m.get("tremor_hz_max", 0.0),
                    "fog_skoru_f1":  m.get("fog_skoru", 0.0),
                }
                log.info(f"FAZ 1 tamamlandi. Tremor ozeti: {faz1_tremor_ozet}")
                faz1_bitis = now_t
                continue

            # ---- FAZ 2 BİTİŞ KONTROLÜ ----
            if faz == 2 and gecen >= cfg.FAZ2_SURE_SN:
                log.info("FAZ 2 tamamlandi. Seans kaydediliyor...")
                break

            # ---- POSE ANALIZ ----
            if results.pose_landmarks and results.pose_world_landmarks:
                lm       = results.pose_landmarks[0]
                world_lm = results.pose_world_landmarks[0]

                if cfg.YUZU_GIZLE:
                    yuz_gizle(img, lm, w, h)

                ts       = now_t
                fv_raw, uyari = analizor.hesapla(lm, world_lm, h, w, ts)

                if fv_raw is not None:
                    fv = fv_raw

                    # FAZ 1: Yürüyüş metriklerini sıfırla (sadece tremor/FOG anlamlı)
                    if faz == 1:
                        fv["adim_uzunlugu_cm"] = 0.0
                        fv["govde_egimi"]       = 0.0
                        fv["kalca_merkez_z"]    = 0.0

                    temporal.guncelle(fv, ts)
                    tm = temporal.metrikler()

                    # ML + Klinik Kural (sadece FAZ 2'de)
                    klinik_risk = 0
                    if faz == 2 and ml_model is not None and tm:
                        try:
                            df_pred = pd.DataFrame([[
                                tm.get("kol_asimetri_ort", 0.0),
                                tm.get("adim_uzunlugu_ort", 0.0),
                                tm.get("govde_egimi_ort",   0.0),
                                tm.get("kadans_spm",        0.0),
                                tm.get("fog_skoru",         0.0),
                                tm.get("yuruyus_hizi_cms",  0.0),
                                tm.get("tremor_hz_max",     0.0),
                                tm.get("kol_genlik_min",    0.0),
                            ]], columns=[
                                "kol_asimetri", "adim_uzunlugu_cm",
                                "govde_egimi",  "kadans_spm",
                                "fog_skoru",    "yuruyus_hizi_cms",
                                "tremor_hz",    "kol_genlik_ort",
                            ])
                            kod = ml_model.predict(df_pred)[0]
                            if hasattr(ml_model, "predict_proba"):
                                tahmin_guveni = float(ml_model.predict_proba(df_pred)[0][kod]) * 100
                            ml_sonuc = "NORMAL" if kod == 0 else "PARKINSON RISKI"

                            # Klinik kural motoru
                            if tm.get("kol_asimetri_ort",  0) > 12.0: klinik_risk += 1
                            if tm.get("adim_uzunlugu_ort", 0) < 50.0: klinik_risk += 1
                            if tm.get("govde_egimi_ort",   0) > 7.0:  klinik_risk += 1
                            if 0 < tm.get("kadans_spm",    0) < 80.0: klinik_risk += 1
                            if 0 < tm.get("yuruyus_hizi_cms", 0) < 90.0: klinik_risk += 1
                            if 0 < tm.get("kol_genlik_min", 0) < 14.0: klinik_risk += 1
                            if tm.get("tremor_hz_max",     0) > 3.0:  klinik_risk += 1
                            if tm.get("fog_skoru",         0) > 0.4:  klinik_risk += 1

                            tahmin_sonucu = ("PARKINSON RISKI"
                                             if ml_sonuc == "PARKINSON RISKI" or klinik_risk >= 3
                                             else "NORMAL")
                        except Exception as e:
                            log.error(f"Tahmin hatasi: {e}")
                    elif faz == 1:
                        tahmin_sonucu = "FAZ 1 — Tremor Testi"

                    if tm:
                        tm["klinik_risk_sayisi"] = klinik_risk

                    veri_logger.kaydet(img, fv, tm if tm else {},
                                       tahmin_sonucu, tahmin_guveni, now_t)

                skeleton_ciz(img, lm, w, h)
            else:
                uyari = "Pose tespit edilemiyor"

            # ---- HUD ÇİZ ----
            # Standart Parkinson HUD (alt panel metrikler)
            img = ParkinsonHUD.ciz(img, fv, tm,
                                    tahmin_sonucu, tahmin_guveni,
                                    seans_no, uyari,
                                    gecen_sure_sn=gecen_global)

            # İki Fazlı özel üst bant (standart HUD üzerine yazar)
            img = IkiFazliHUD.faz_bantı_ciz(img, faz, gecen, toplam_sure,
                                              seans_no, cfg.KULLANICI_ID)
            img = IkiFazliHUD.yonerge_bandı_ciz(img, faz, gecen, toplam_sure)

            # FPS
            fps_sayaci += 1
            elapsed_fps = now_t - fps_ref
            if elapsed_fps >= 2.0:
                anlik_fps = fps_sayaci / elapsed_fps
                fps_sayaci, fps_ref = 0, now_t
            cv2.putText(img, f"FPS:{anlik_fps:.0f}", (w - 70, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1)

            cv2.imshow(f"NYTAS-Parkinson | İki Fazlı Seans #{seans_no:03d}", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                log.info("Kullanici Q ile cikti.")
                break
            elif key == ord(" ") and faz == 1:
                # SPACE ile Faz 1'i erken atla
                log.info("SPACE: Faz 1 erken tamamlandi, Faz 2'ye geciliyor.")
                m = temporal.metrikler()
                faz1_tremor_ozet = {
                    "tremor_hz_L":   m.get("tremor_hz_L", 0.0),
                    "tremor_hz_R":   m.get("tremor_hz_R", 0.0),
                    "tremor_hz_max": m.get("tremor_hz_max", 0.0),
                    "fog_skoru_f1":  m.get("fog_skoru", 0.0),
                }
                faz1_bitis = now_t

    # ---- Seans Sonu ----
    cap.stop()
    veri_logger.kapat()
    cv2.destroyAllWindows()
    log.info(f"İki Fazlı Seans #{seans_no:03d} tamamlandi.")
    if faz1_tremor_ozet:
        log.info(f"FAZ 1 Tremor Ozeti: {faz1_tremor_ozet}")

    # Otomatik klinik rapor
    try:
        sys.path.insert(0, str(BASE_DIR))
        from kaynak_kodlar.rapor_olusturucu import rapor_olustur
        log.info(f"Klinik rapor derleniyor (Seans #{seans_no:03d})...")
        rapor_dosyasi = rapor_olustur(seans_no, cfg.VERI_DIZINI, otomatik_ac=True)
        if rapor_dosyasi:
            log.info(f"Klinik rapor acildi: {rapor_dosyasi}")
    except Exception as e:
        log.error(f"Klinik rapor olusturulamadi: {e}")


# =========================================================================
# Tkinter Giriş Ekranı (İki Fazlı Mod)
# =========================================================================
def iki_fazli_giris_ekrani():
    root = tk.Tk()
    root.title("NYTAS-Parkinson | İki Fazlı Klinik Seans Yapılandırması")
    root.geometry("560x680")
    root.eval("tk::PlaceWindow . center")
    root.configure(bg="#0f172a")

    # Başlık
    tk.Label(root, text="🔬 İKİ FAZLI KLİNİK SEANS",
             font=("Segoe UI", 15, "bold"), bg="#0f172a", fg="#38bdf8").pack(pady=(18, 2))
    tk.Label(root, text="FAZ 1: İstirahat Tremor  |  FAZ 2: Yürüyüş Analizi",
             font=("Segoe UI", 9, "italic"), bg="#0f172a", fg="#94a3b8").pack()
    tk.Label(root, text="TÜBİTAK 1002 Hızlı Destek Programı",
             font=("Segoe UI", 8), bg="#0f172a", fg="#475569").pack(pady=(0, 10))

    frm = tk.Frame(root, bg="#1e293b", padx=18, pady=14,
                   highlightthickness=1, highlightbackground="#334155")
    frm.pack(padx=20, fill="x")

    def satir(row, etiket, default, hint=""):
        tk.Label(frm, text=etiket, font=("Segoe UI", 9, "bold"),
                 bg="#1e293b", fg="#e2e8f0").grid(row=row, column=0, pady=5, sticky="w")
        ent = tk.Entry(frm, font=("Segoe UI", 9), width=22,
                       bg="#0f172a", fg="#f1f5f9", insertbackground="white")
        ent.insert(0, default)
        ent.grid(row=row, column=1, pady=5, padx=10, sticky="w")
        if hint:
            tk.Label(frm, text=hint, font=("Segoe UI", 7),
                     bg="#1e293b", fg="#64748b").grid(row=row, column=2, sticky="w")
        return ent

    id_ent     = satir(0, "Hasta ID:",        "hasta_001")
    boy_ent    = satir(1, "Boy (cm):",         "175.0")
    cam_ent    = satir(2, "Kamera Kaynağı:",   "0", "(0: dahili, 1: USB)")
    faz1_ent   = satir(3, "Faz 1 Süresi (s):", "45", "İstirahat tremor")
    faz2_ent   = satir(4, "Faz 2 Süresi (s):", "60", "Yürüyüş analizi")

    def dosya_sec():
        from tkinter import filedialog
        d = filedialog.askopenfilename(
            title="Video Dosyası Seç",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("Tümü", "*.*")]
        )
        if d:
            cam_ent.delete(0, tk.END)
            cam_ent.insert(0, d)

    tk.Button(frm, text="📁 Video Seç", font=("Segoe UI", 8), bg="#334155",
              fg="white", relief="flat", cursor="hand2",
              command=dosya_sec).grid(row=2, column=2, padx=4)

    # Klinik durum
    tk.Label(frm, text="Klinik Durum:", font=("Segoe UI", 9, "bold"),
             bg="#1e293b", fg="#38bdf8").grid(row=5, column=0, pady=5, sticky="w")
    durum_cb = ttk.Combobox(frm, values=[
        "Bilinmiyor / Rutin Tarama",
        "Sağlıklı Kontrol Grubu (Referans)",
        "Tanı Almış Parkinson Hastası"
    ], state="readonly", font=("Segoe UI", 8), width=28)
    durum_cb.current(0)
    durum_cb.grid(row=5, column=1, columnspan=2, pady=5, padx=10, sticky="w")

    # Checkboxlar
    rot_var  = tk.IntVar(value=0)
    blur_var = tk.IntVar(value=1)
    tk.Checkbutton(frm, text="Telefon Dikey Çekim Modu (90° Döndür)",
                   variable=rot_var, font=("Segoe UI", 8),
                   bg="#1e293b", fg="#e2e8f0", selectcolor="#0f172a",
                   activebackground="#1e293b").grid(row=6, column=0, columnspan=3, pady=3, sticky="w")
    tk.Checkbutton(frm, text="Yüzü Gizle / Bulanıklaştır (KVKK)",
                   variable=blur_var, font=("Segoe UI", 8),
                   bg="#1e293b", fg="#e2e8f0", selectcolor="#0f172a",
                   activebackground="#1e293b").grid(row=7, column=0, columnspan=3, pady=3, sticky="w")

    # Faz açıklamaları
    aciklama = (
        "📋 SEANS AKIŞI:\n"
        "• FAZ 1: Hasta hareketsiz durur — İstirahat tremor FFT ölçümü yapılır.\n"
        "  SPACE tuşu ile FAZ 1'i erken bitirebilirsiniz.\n"
        "• FAZ 2: Hasta yürür — 8 dijital biyobelirteç tam olarak hesaplanır.\n"
        "• Seans sonu otomatik HTML klinik raporu açılır.\n"
        "• Her fazda Q tuşu ile çıkılabilir."
    )
    tk.Label(root, text=aciklama, fg="#94a3b8", bg="#0f172a",
             font=("Segoe UI", 8), justify="left").pack(pady=8, padx=20, fill="x")

    def baslat():
        k_id   = id_ent.get().strip()
        k_boy  = boy_ent.get().strip()
        k_cam  = cam_ent.get().strip()
        f1_str = faz1_ent.get().strip()
        f2_str = faz2_ent.get().strip()

        if not k_id or not k_boy:
            messagebox.showerror("Hata", "Hasta ID ve Boy zorunludur!")
            return
        try:
            boy_f = float(k_boy)
            if boy_f < 50 or boy_f > 250:
                raise ValueError
        except ValueError:
            messagebox.showerror("Hata", "Geçerli boy girin (50–250 cm)!")
            return

        cfg = IkiFazliConfig()
        cfg.KULLANICI_ID      = k_id
        cfg.KULLANICI_BOYU_CM = boy_f
        cfg.KAMERA_KAYNAK     = int(k_cam) if k_cam.isdigit() else (k_cam or 0)
        cfg.KAMERA_DONDUR     = bool(rot_var.get())
        cfg.YUZU_GIZLE        = bool(blur_var.get())
        try:
            cfg.FAZ1_SURE_SN = max(10.0, float(f1_str))
            cfg.FAZ2_SURE_SN = max(15.0, float(f2_str))
        except ValueError:
            pass

        durum_str = durum_cb.get()
        if "Sağlıklı" in durum_str:
            cfg.KLINIK_ETIKET = 0
        elif "Parkinson" in durum_str:
            cfg.KLINIK_ETIKET = 1
        else:
            cfg.KLINIK_ETIKET = -1

        root.destroy()
        iki_fazli_analiz_baslat(cfg)

    tk.Button(root, text="🔬  İKİ FAZLI SEANSI BAŞLAT",
              font=("Segoe UI", 11, "bold"), bg="#0284c7", fg="white",
              activebackground="#0369a1", relief="flat", cursor="hand2",
              pady=10, command=baslat).pack(fill="x", padx=20, pady=(4, 6))
    tk.Button(root, text="✖  İptal",
              font=("Segoe UI", 9), bg="#334155", fg="white",
              relief="flat", cursor="hand2",
              command=root.destroy).pack(fill="x", padx=20, pady=(0, 12))

    root.mainloop()


# =========================================================================
# CLI Giriş Noktası
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYTAS-Parkinson İki Fazlı Klinik Seans")
    parser.add_argument("--id",        default="hasta_001",    help="Hasta ID")
    parser.add_argument("--boy",       type=float, default=175.0, help="Boy (cm)")
    parser.add_argument("--kamera",    default="0",            help="Kamera kaynağı (0/1/URL/dosya)")
    parser.add_argument("--dondur",    action="store_true",    help="Görüntüyü 90° döndür")
    parser.add_argument("--no-blur",   action="store_true",    help="Yüz bulanıklaştırmayı kapat")
    parser.add_argument("--etiket",    type=int, default=-1, choices=[-1, 0, 1])
    parser.add_argument("--faz1-sure", type=float, default=45.0, help="Faz 1 süresi (saniye)")
    parser.add_argument("--faz2-sure", type=float, default=60.0, help="Faz 2 süresi (saniye)")
    parser.add_argument("--gui",       action="store_true",    help="Tkinter giriş ekranını aç")
    args = parser.parse_args()

    if args.gui or len(sys.argv) == 1:
        iki_fazli_giris_ekrani()
    else:
        cfg = IkiFazliConfig()
        cfg.KULLANICI_ID      = args.id
        cfg.KULLANICI_BOYU_CM = args.boy
        cfg.KAMERA_KAYNAK     = int(args.kamera) if str(args.kamera).isdigit() else args.kamera
        cfg.KAMERA_DONDUR     = args.dondur
        cfg.YUZU_GIZLE        = not args.no_blur
        cfg.KLINIK_ETIKET     = args.etiket
        cfg.FAZ1_SURE_SN      = args.faz1_sure
        cfg.FAZ2_SURE_SN      = args.faz2_sure
        iki_fazli_analiz_baslat(cfg)
