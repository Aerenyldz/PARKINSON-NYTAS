"""
=============================================================================
NYTAS-PARKINSON — Parkinson Hastaligi Odakli Yuruyus Analiz Modulu
TUBITAK 1002 Hizli Destek Programi
=============================================================================
Surum: v1.0
Aciklama:
  Genel NYTAS (v4) sisteminin Parkinson Hastaligina ozel sadelesilmis ve
  odaklanmis versiyonudur. 18 genel parametre yerine Parkinson literaturunde
  en guclu kanita sahip 8 dijital biyobelirteci olcer:

  1. Kol Salinim Genligi (Arm Swing Amplitude)
  2. Kol Salinim Asimetrisi (Arm Swing Asymmetry)
  3. Adim Uzunlugu (Stride Length)
  4. Kadans / Adim Sikliginda Duzensizlik (Cadence Variability)
  5. Donma Skoru (Freezing of Gait - FOG)
  6. Govde One Egimi (Trunk Flexion / Camptocormia)
  7. Yuruyus Hizi (Gait Velocity)
  8. El Tremor Frekansi (Hand Tremor Frequency via FFT)

  Sinyal Filtreleme: One-Euro Filter (literatur standardi, EMA'dan ustun)
  Gizlilik: KVKK uyumlu yuz bulaniklastirma (opsiyonel)

Gelistirici Notlari:
  - mediapipe 0.10.30+ Tasks API kullanir
  - pose_landmarker.task modeli gereklidir (~5MB, otomatik indirilir)
=============================================================================
"""

import csv
import logging
import math
import os
import shutil
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, filedialog, ttk

# Path Ayarları
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from ayarlar.settings import POSE_LANDMARKER_PATH, PARKINSON_MODEL_PATH, PROCESSED_DATA_DIR
    MODEL_DOSYASI = str(POSE_LANDMARKER_PATH)
    DEFAULT_ML_MODEL = str(PARKINSON_MODEL_PATH)
    DEFAULT_VERI_DIZINI = str(PROCESSED_DATA_DIR)
except Exception:
    try:
        from config.settings import POSE_LANDMARKER_PATH, PARKINSON_MODEL_PATH, PROCESSED_DATA_DIR
        MODEL_DOSYASI = str(POSE_LANDMARKER_PATH)
        DEFAULT_ML_MODEL = str(PARKINSON_MODEL_PATH)
        DEFAULT_VERI_DIZINI = str(PROCESSED_DATA_DIR)
    except Exception:
        MODEL_DOSYASI = str(BASE_DIR / "modeller" / "pose_landmarker.task")
        DEFAULT_ML_MODEL = str(BASE_DIR / "modeller" / "parkinson_model.pkl")
        DEFAULT_VERI_DIZINI = str(BASE_DIR / "veriler" / "processed" / "nytas_parkinson_veri")


# =========================================================================
# BlazePose 33 Landmark Baglantilari
# =========================================================================
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12),
    (11, 13), (13, 15), (15, 17), (17, 19), (19, 15), (15, 21),
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

# =========================================================================
# Loglama
# =========================================================================
class _RenkliFormatter(logging.Formatter):
    _RENKLER = {
        logging.DEBUG: "\033[37m", logging.INFO: "\033[96m",
        logging.WARNING: "\033[93m", logging.ERROR: "\033[91m",
    }
    _RESET = "\033[0m"

    def format(self, record):
        renk = self._RENKLER.get(record.levelno, self._RESET)
        r = logging.makeLogRecord(record.__dict__)
        r.levelname = f"{renk}{record.levelname}{self._RESET}"
        r.msg = f"{renk}{record.msg}{self._RESET}"
        return super().format(r)


def _log_kur(seans_no: int, log_kl: str) -> logging.Logger:
    os.makedirs(log_kl, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    fh = logging.FileHandler(os.path.join(log_kl, f"seans_{seans_no:03d}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    ch = logging.StreamHandler()
    ch.setFormatter(_RenkliFormatter(fmt))
    logger = logging.getLogger(f"NYTAS_PD.s{seans_no}")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# =========================================================================
# Seans Numaralama
# =========================================================================
def _sonraki_seans_no(veri_dizini: str) -> int:
    csv_yol = os.path.join(veri_dizini, "parkinson_master.csv")
    if not os.path.exists(csv_yol):
        return 1
    try:
        df = pd.read_csv(csv_yol, usecols=["seans_no"])
        return 1 if df.empty else int(df["seans_no"].max()) + 1
    except Exception:
        return 1


# =========================================================================
# Model Indirme
# =========================================================================
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

def model_hazirla(log: logging.Logger) -> bool:
    if os.path.exists(MODEL_DOSYASI):
        log.info(f"Pose modeli hazir: {MODEL_DOSYASI}")
        return True
    log.info("Pose modeli indiriliyor (~5MB)...")
    try:
        os.makedirs(os.path.dirname(MODEL_DOSYASI), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_DOSYASI)
        log.info(f"Pose modeli indirildi: {MODEL_DOSYASI}")
        return True
    except Exception as e:
        log.error(f"Model indirilemedi: {e}")
        return False


# =========================================================================
# Konfigurasyon
# =========================================================================
@dataclass
class Config:
    KULLANICI_BOYU_CM: float  = 175.0
    KULLANICI_ID: str         = "hasta_001"
    YUZU_GIZLE: bool          = True

    KAMERA_KAYNAK: object     = 0       # 0: Dahili webcam, 1: USB cam, "http://...": IP cam veya video.mp4
    KAMERA_DONDUR: bool       = False   # Telefon dikey çekiminde True, normal webcamde False
    FRAME_GENISLIK: int       = 640
    FRAME_YUKSEKLIK: int      = 480
    MIN_DETECTION_CONF: float = 0.6
    MIN_TRACKING_CONF: float  = 0.6
    BUFFER_BOYUTU: int        = 120     # ~4 saniye @ 30fps
    VERI_DIZINI: str          = DEFAULT_VERI_DIZINI
    VIDEO_KAYDET: bool        = True
    CSV_KAYDET: bool          = True
    ML_MODEL_DOSYASI: str     = DEFAULT_ML_MODEL
    KLINIK_ETIKET: int        = -1      # -1: Bilinmiyor/Rutin Tarama, 0: Saglikli Kontrol, 1: Parkinson Hastasi

    KAMERA_FPS: int           = 30
    KAMERA_COZUNURLUK_W: int  = 640
    KAMERA_COZUNURLUK_H: int  = 480
    KAMERA_BUFFER: int        = 1

CFG = Config()


# =========================================================================
# Video Akisi (Threaded)
# =========================================================================
class VideoStream:
    def __init__(self, src):
        # Eğer kaynak sayısal metin ise integer yap (ör: "0" -> 0)
        if isinstance(src, str):
            if src.strip().isdigit():
                self.src = int(src.strip())
            else:
                self.src = src.strip()
        else:
            self.src = src

        self.stream = cv2.VideoCapture(self.src)
        if isinstance(self.src, str) and not str(self.src).isdigit():
            self.stream.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4000)
            self.stream.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 4000)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, CFG.KAMERA_BUFFER)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.KAMERA_COZUNURLUK_W)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.KAMERA_COZUNURLUK_H)
        self.stream.set(cv2.CAP_PROP_FPS, CFG.KAMERA_FPS)
        self.grabbed, self.frame = False, None
        self.stopped = False
        self._lock = threading.Lock()
        self._evt = threading.Event()

    def start(self) -> "VideoStream":
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            ok, frm = self.stream.read()
            if ok:
                with self._lock:
                    self.grabbed, self.frame = ok, frm
                self._evt.set()
            else:
                time.sleep(0.001)

    def read(self) -> Optional[np.ndarray]:
        self._evt.wait(timeout=0.05)
        self._evt.clear()
        with self._lock:
            return self.frame.copy() if self.grabbed and self.frame is not None else None

    def stop(self):
        self.stopped = True
        time.sleep(0.1)
        self.stream.release()


# =========================================================================
# One-Euro Filter (Literatur standardi - EMA'dan ustun)
# Kaynak: Casiez et al. 2012  "1€ Filter"
# Adaptif: Durgunken cok filtreler, hareket edince az filtreler (lag yok)
# =========================================================================
class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        te = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + te / dt) if dt > 0 else 1.0

    def __call__(self, x: float, t: float) -> float:
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x
        dt = t - self.t_prev
        if dt <= 0:
            dt = 1e-6

        # Turev (hiz) hesabi
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # Adaptif cutoff: hareket hizliysa cutoff artar (az filtre = az gecikme)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


# =========================================================================
# Yardimci Matematik
# =========================================================================
def aci_hesapla(a, b, c) -> float:
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    aci = abs(np.degrees(rad))
    return 360.0 - aci if aci > 180.0 else aci


def piksel_cm_orani(lm, fh: int) -> Optional[float]:
    nose_y = lm[0].y
    heel_y = max(lm[29].y, lm[30].y)
    boy_n = abs(heel_y - nose_y)
    if boy_n < 0.25:
        return None
    return CFG.KULLANICI_BOYU_CM / (boy_n * fh)


# =========================================================================
# PARKINSON ANALIZOR — 8 Biyobelirtec
# =========================================================================
class ParkinsonAnalizor:
    """Her karede Parkinson'a ozgu biyomekanik ham verileri hesaplar."""

    def __init__(self):
        # One-Euro filtreler (anlik gurultuyu engeller)
        self._f_arm_L = OneEuroFilter(min_cutoff=1.5, beta=0.01)
        self._f_arm_R = OneEuroFilter(min_cutoff=1.5, beta=0.01)
        self._f_trunk = OneEuroFilter(min_cutoff=1.0, beta=0.005)
        self._f_step  = OneEuroFilter(min_cutoff=1.0, beta=0.005)

    def hesapla(self, lm, world_lm, fh: int, fw: int, t: float) -> Tuple[Optional[dict], str]:
        ratio = piksel_cm_orani(lm, fh)
        uyari = ""

        if ratio is None:
            return None, "Kisi tam gorunmuyor"

        try:
            # --- Kamera acisi kontrolu (Z-derinlik farki) ---
            omuz_z_fark = abs(world_lm[11].z - world_lm[12].z)
            if omuz_z_fark > 0.18:
                uyari = "Uyari: Kameraya dik yuruyun!"

            # ====== 1 & 2: KOL SALINIM HAM VERILERI (Omuza Gore Bagil Bilek Pozisyonu) ======
            # Omuz (11, 12) ile Bilek (15, 16) arasi bagil dikey deplasman (cm)
            # Bu sinyalin temporal penceredeki tepe-cukur farki (Peak-to-Peak) salinim genligini verir.
            arm_rel_L = (lm[15].y - lm[11].y) * fh * ratio
            arm_rel_R = (lm[16].y - lm[12].y) * fh * ratio

            arm_rel_L = self._f_arm_L(arm_rel_L, t)
            arm_rel_R = self._f_arm_R(arm_rel_R, t)

            # ====== 3: ADIM UZUNLUGU (3D World Landmarks) ======
            w_l, w_r = world_lm[27], world_lm[28]  # sol/sag ayak bilegi
            adim_uzunlugu = math.sqrt(
                (w_l.x - w_r.x)**2 + (w_l.y - w_r.y)**2 + (w_l.z - w_r.z)**2
            ) * 100  # metre -> cm
            adim_uzunlugu = self._f_step(adim_uzunlugu, t)

            # ====== 6: GOVDE ONE EGIMI (Trunk Flexion / Camptocormia — 3D Sagittal) ======
            # MediaPipe world_landmarks 3D metre koordinatlari kullanilarak
            # omurga vektoru ile dikey eksen arasindaki aci hesaplanir.
            # Bu sayede kameraya dogru yururken one egim dogrudan olculur.
            omuz_3d = np.array([
                (world_lm[11].x + world_lm[12].x) / 2.0,
                (world_lm[11].y + world_lm[12].y) / 2.0,
                (world_lm[11].z + world_lm[12].z) / 2.0])
            kalca_3d = np.array([
                (world_lm[23].x + world_lm[24].x) / 2.0,
                (world_lm[23].y + world_lm[24].y) / 2.0,
                (world_lm[23].z + world_lm[24].z) / 2.0])
            govde_vek = omuz_3d - kalca_3d
            dikey_vek = np.array([0.0, -1.0, 0.0])  # Y-yukari (world_lm koordinat sistemi)
            govde_norm = np.linalg.norm(govde_vek)
            if govde_norm > 1e-6:
                cos_aci = np.dot(govde_vek, dikey_vek) / govde_norm
                govde_egimi = np.degrees(np.arccos(np.clip(cos_aci, -1.0, 1.0)))
            else:
                govde_egimi = 0.0
            govde_egimi = self._f_trunk(govde_egimi, t)

            # ====== 7: YURUYUS HIZI (Kalca Merkezi Z-derinlik deplasmanı) ======
            # Kameraya dogru/geriye yururken ilerleme Z eksenindedir
            kalca_merkez_z = (world_lm[23].z + world_lm[24].z) / 2.0

            # ====== 8: EL TREMOR (Govdeden Bagimsiz Omuza Gore Bagil Bilek Hareketi) ======
            # Govde adim sarsintisini elimine etmek icin omuza gore bagil piksel farki alinir
            bilek_rel_L_px = (lm[15].y - lm[11].y) * fh
            bilek_rel_R_px = (lm[16].y - lm[12].y) * fh

            # ====== 5: FREEZING (Ayak bilegi Y - FOG ham verisi) ======
            ayak_L_y = lm[27].y
            ayak_R_y = lm[28].y

            return {
                "arm_rel_L":        arm_rel_L,
                "arm_rel_R":        arm_rel_R,
                "adim_uzunlugu_cm": adim_uzunlugu,
                "govde_egimi":      govde_egimi,
                "kalca_merkez_z":   kalca_merkez_z,
                "bilek_rel_L_px":   bilek_rel_L_px,
                "bilek_rel_R_px":   bilek_rel_R_px,
                "ayak_L_y":         ayak_L_y,
                "ayak_R_y":         ayak_R_y,
            }, uyari

        except Exception as e:
            return None, f"Hata: {e}"


# =========================================================================
# PARKINSON TEMPORAL — Zamana Bagli Analizler (Peak-to-Peak Genlik, FOG, Tremor FFT, Kadans)
# =========================================================================
class ParkinsonTemporal:
    def __init__(self, buf: int = 120, fps: int = 30):
        self.fps = fps
        # Kol salınım geçmişi (anlık bağıl deplasmanlar)
        self.arm_L = deque(maxlen=buf)
        self.arm_R = deque(maxlen=buf)
        # Adim & Govde
        self.adim = deque(maxlen=buf)
        self.govde = deque(maxlen=buf)
        # Yuruyus hizi & Zaman damgalari (Z-derinlik ekseni)
        self.kalca_z = deque(maxlen=buf)
        self.kalca_t = deque(maxlen=buf)
        # Tremor icin govdeden arindirilmis bagil bilek Y
        self.bilek_rel_L = deque(maxlen=buf)
        self.bilek_rel_R = deque(maxlen=buf)
        self.t_kayit = deque(maxlen=buf)
        # Freezing icin ayak bilegi Y
        self.ayak_L = deque(maxlen=buf)
        self.ayak_R = deque(maxlen=buf)
        # Kadans (topuk vuruşu zamanlari)
        self._adim_t: deque = deque(maxlen=30)
        self._son_adim = 0.0
        self._prev_ayak_y = 0.0

    def guncelle(self, veri: dict, t: float):
        self.arm_L.append(veri.get("arm_rel_L", 0.0))
        self.arm_R.append(veri.get("arm_rel_R", 0.0))
        self.adim.append(veri.get("adim_uzunlugu_cm", 0.0))
        self.govde.append(veri.get("govde_egimi", 0.0))
        self.kalca_z.append(veri.get("kalca_merkez_z", 0.0))
        self.kalca_t.append(t)
        self.bilek_rel_L.append(veri.get("bilek_rel_L_px", 0.0))
        self.bilek_rel_R.append(veri.get("bilek_rel_R_px", 0.0))
        self.t_kayit.append(t)
        self.ayak_L.append(veri.get("ayak_L_y", 0.0))
        self.ayak_R.append(veri.get("ayak_R_y", 0.0))

        # Topuk vurusu tespiti (kadans)
        ayak_y = veri.get("ayak_L_y", 0.0)
        if (self._prev_ayak_y > 0
                and ayak_y < self._prev_ayak_y - 0.01
                and t - self._son_adim > 0.3):
            self._adim_t.append(t)
            self._son_adim = t
        self._prev_ayak_y = ayak_y

    def metrikler(self) -> dict:
        if len(self.arm_L) < 20:
            return {}

        m: Dict[str, float] = {}

        # --- 1 & 2: KOL SALINIM GENLIGI (Peak-to-Peak) & ASIMETRISI ---
        # Son 60 kareden (yaklasik 2 saniyelik 1 tam yuruyus dongusu) genlik hesabi
        pencere_L = list(self.arm_L)[-60:]
        pencere_R = list(self.arm_R)[-60:]
        genlik_L = float(np.ptp(pencere_L)) if len(pencere_L) >= 15 else 0.0
        genlik_R = float(np.ptp(pencere_R)) if len(pencere_R) >= 15 else 0.0

        m["kol_genlik_L_ort"] = genlik_L
        m["kol_genlik_R_ort"] = genlik_R
        m["kol_genlik_ort"]   = (genlik_L + genlik_R) / 2.0
        m["kol_genlik_min"]   = min(genlik_L, genlik_R)  # En cok kisitlanan taraf

        # Asimetri indeksi (%)
        ort_genlik = (genlik_L + genlik_R) / 2.0
        if ort_genlik > 1.0:
            m["kol_asimetri_ort"] = float(abs(genlik_L - genlik_R) / ort_genlik * 100.0)
        else:
            m["kol_asimetri_ort"] = 0.0

        # --- 3: ADIM UZUNLUGU ---
        m["adim_uzunlugu_ort"] = float(np.mean(self.adim))
        m["adim_varyabilite"]  = float(np.std(self.adim))

        # --- 4: KADANS (adim/dakika) ---
        if len(self._adim_t) >= 3:
            aralar = np.diff(list(self._adim_t))
            m["kadans_spm"]          = 60.0 / float(np.mean(aralar))
            m["kadans_varyabilite"]  = float(np.std(aralar))
        else:
            m["kadans_spm"]          = 0.0
            m["kadans_varyabilite"]  = 0.0

        # --- 6: GOVDE EGIMI ---
        m["govde_egimi_ort"]  = float(np.mean(self.govde))

        # --- 7: YURUYUS HIZI (cm/s) — Z-derinlik ekseni ---
        if len(self.kalca_z) >= 10 and len(self.kalca_t) >= 10:
            dz = abs(self.kalca_z[-1] - self.kalca_z[0]) * 100  # metre -> cm
            dt = self.kalca_t[-1] - self.kalca_t[0]
            m["yuruyus_hizi_cms"] = dz / dt if dt > 0.1 else 0.0
        else:
            m["yuruyus_hizi_cms"] = 0.0

        # --- 5: FREEZING OF GAIT (FOG) SKORU ---
        if len(self.ayak_L) >= 30:
            son_30_L = list(self.ayak_L)[-30:]
            son_30_R = list(self.ayak_R)[-30:]
            hareket_L = np.std(son_30_L)
            hareket_R = np.std(son_30_R)
            fog_esik = 0.005
            if hareket_L < fog_esik and hareket_R < fog_esik:
                m["fog_skoru"] = 1.0   # Tam donma
            elif hareket_L < fog_esik or hareket_R < fog_esik:
                m["fog_skoru"] = 0.5   # Kısmi donma
            else:
                m["fog_skoru"] = 0.0   # Normal yuruyus
        else:
            m["fog_skoru"] = 0.0

        # --- 8: BILATERAL EL TREMOR FREKANSI (Dinamik FFT & 3.5 - 7.5 Hz Bandi) ---
        m["tremor_hz_L"]   = self._tremor_fft(self.bilek_rel_L, self.t_kayit)
        m["tremor_hz_R"]   = self._tremor_fft(self.bilek_rel_R, self.t_kayit)
        m["tremor_hz_max"] = max(m["tremor_hz_L"], m["tremor_hz_R"])
        m["tremor_var"]    = (m["tremor_hz_max"] > 0)

        return m

    def _tremor_fft(self, sinyal_deque: deque, t_deque: deque) -> float:
        """Govdeden bagimsizlestirilmis bagil bilek sinyalinin dinamik FFT analizini yapar."""
        if len(sinyal_deque) < 45 or len(t_deque) < 45:
            return 0.0

        sinyal = np.array(list(sinyal_deque), dtype=float)
        ts = np.array(list(t_deque), dtype=float)

        # Dinamik FPS tespiti (kamera dalgalanmalarina karsi)
        dt_ort = (ts[-1] - ts[0]) / (len(ts) - 1)
        if dt_ort <= 0 or dt_ort > 0.2:  # Gecersiz veya cok dusuk FPS
            return 0.0
        fs = 1.0 / dt_ort

        # Nyquist siniri kontrolu (3.5-7.5 Hz icin en az 15 FPS gerekir)
        if fs < 15.0:
            return 0.0

        # DC bileseni ve lineer trendi cikar
        sinyal = sinyal - np.mean(sinyal)
        std_val = np.std(sinyal)
        if std_val < 1.0:  # Fiziksel hareket yoksa gurultuyu ele
            return 0.0

        n = len(sinyal)
        fft_vals = np.abs(np.fft.rfft(sinyal))
        freqs = np.fft.rfftfreq(n, d=dt_ort)

        # 3.5 - 7.5 Hz bandi (Parkinson tremor bandi)
        mask = (freqs >= 3.5) & (freqs <= 7.5)
        if not np.any(mask):
            return 0.0

        fft_band = fft_vals[mask]
        freq_band = freqs[mask]

        peak_idx = np.argmax(fft_band)
        peak_amp = fft_band[peak_idx]

        # Sinyal-gurultu orani kontrolu (Peak, bant ortalamasinin en az 2.2 kati olmali)
        ortalama_genlik = np.mean(fft_vals[1:])
        if peak_amp > ortalama_genlik * 2.2:
            return float(freq_band[peak_idx])
        return 0.0


# =========================================================================
# Iskelet Cizimi
# =========================================================================
def skeleton_ciz(img: np.ndarray, lm, w: int, h: int):
    pts = [(int(l.x * w), int(l.y * h)) for l in lm]
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(img, pts[a], pts[b], (0, 160, 255), 2)
    for pt in pts:
        cv2.circle(img, pt, 3, (0, 230, 230), -1)


# =========================================================================
# Parkinson HUD (Gorsellestiricici)
# =========================================================================
class ParkinsonHUD:
    CYAN    = (200, 200, 0)
    YESIL   = (50, 220, 50)
    KIRMIZI = (50, 50, 220)
    SARI    = (0, 200, 255)
    TURUNCU = (0, 140, 255)
    BEYAZ   = (255, 255, 255)
    GRI     = (140, 140, 140)
    KOYU    = (18, 18, 18)

    @staticmethod
    def ciz(img: np.ndarray, fv: Optional[dict], tm: Optional[dict],
            tahmin: str, guven: float, seans_no: int, uyari: str = "",
            gecen_sure_sn: float = 0.0) -> np.ndarray:
        if fv is None: fv = {}
        if tm is None: tm = {}
        h, w = img.shape[:2]
        C = ParkinsonHUD

        # --- Baslik ---
        cv2.rectangle(img, (0, 0), (w, 90), C.KOYU, -1)
        cv2.putText(img, "NYTAS-PARKINSON", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C.CYAN, 1)

        # Sure ve Kararlilik Gostergesi (30 sn hedefli klinik standart)
        HEDEF_SURE = 30.0
        pct = min(100, int((gecen_sure_sn / HEDEF_SURE) * 100)) if HEDEF_SURE > 0 else 100
        if pct >= 100:
            durum_metni = f"Sure: {gecen_sure_sn:.0f}s - [IDEAL KARARLI REJIM] Cikis icin 'Q'"
            durum_renk = C.YESIL
        elif pct >= 33:
            durum_metni = f"Sure: {gecen_sure_sn:.0f}s / 30s (%{pct}) - [Veri Kararlasiyor...]"
            durum_renk = C.SARI
        else:
            durum_metni = f"Sure: {gecen_sure_sn:.0f}s / 30s (%{pct}) - [Isinma / Ivmelenme]"
            durum_renk = C.TURUNCU

        cv2.putText(img, f"Seans #{seans_no:03d} | {CFG.KULLANICI_ID} | {durum_metni}",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, durum_renk, 1)

        # Kararlilik Ilerleme Cubugu (Baslik altinda ince serit)
        cv2.rectangle(img, (0, 86), (w, 90), (35, 35, 35), -1)
        bar_w = int((w * pct) / 100)
        cv2.rectangle(img, (0, 86), (bar_w, 90), durum_renk, -1)

        renk = C.YESIL if tahmin == "NORMAL" else C.KIRMIZI
        cv2.putText(img, tahmin, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, renk, 2)
        if guven > 0:
            cv2.putText(img, f"%{guven:.0f}", (w - 60, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C.BEYAZ, 1)

        # --- Anlik Veriler ---
        if fv:
            y0 = 120
            satirlar = [
                f"Kol Sal. Sol: {tm.get('kol_genlik_L_ort', 0):.1f} cm" if tm else f"Kol Poz. Sol: {fv.get('arm_rel_L', 0):.1f} cm",
                f"Kol Sal. Sag: {tm.get('kol_genlik_R_ort', 0):.1f} cm" if tm else f"Kol Poz. Sag: {fv.get('arm_rel_R', 0):.1f} cm",
                f"Kol Asimetri: {tm.get('kol_asimetri_ort', 0):.1f} %" if tm else "Kol Asimetri: Hesaplanıyor...",
                f"Adim Uzun.: {fv.get('adim_uzunlugu_cm', 0):.1f} cm",
                f"Govde Egimi: {fv.get('govde_egimi', 0):.1f} derece",
            ]
            for i, s in enumerate(satirlar):
                cv2.putText(img, s, (10, y0 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, C.BEYAZ, 1)

        # --- Temporal Metrikler (Alt Panel — Klinik Eşik Renkli) ---
        if tm:
            cv2.rectangle(img, (0, h - 215), (w, h), (22, 22, 22), -1)
            fog = tm.get("fog_skoru", 0.0)
            fog_renk = C.KIRMIZI if fog >= 1.0 else (C.TURUNCU if fog >= 0.5 else C.YESIL)
            fog_metin = "DONMA!" if fog >= 1.0 else ("Kismi" if fog >= 0.5 else "Yok")

            tremor_max = tm.get("tremor_hz_max", 0.0)
            tremor_L = tm.get("tremor_hz_L", 0.0)
            tremor_R = tm.get("tremor_hz_R", 0.0)
            if tremor_max > 0:
                detay = f"{tremor_max:.1f} Hz ("
                detay += f"Sol:{tremor_L:.1f} " if tremor_L > 0 else ""
                detay += f"Sag:{tremor_R:.1f}" if tremor_R > 0 else ""
                detay += ")"
                tremor_metin = detay
                tremor_renk = C.KIRMIZI if tremor_max > 3.0 else C.TURUNCU
            else:
                tremor_metin = "Yok"
                tremor_renk = C.YESIL

            # Klinik esik bazli renklendirme (riskli = kirmizi, normal = yesil/sari)
            kadans_v = tm.get("kadans_spm", 0.0)
            hiz_v    = tm.get("yuruyus_hizi_cms", 0.0)
            asim_v   = tm.get("kol_asimetri_ort", 0.0)
            egim_v   = tm.get("govde_egimi_ort", 0.0)

            kadans_renk = C.KIRMIZI if (0 < kadans_v < 80) else C.SARI
            hiz_renk    = C.KIRMIZI if (0 < hiz_v < 90) else C.SARI
            asim_renk   = C.KIRMIZI if asim_v > 12 else C.SARI
            egim_renk   = C.KIRMIZI if egim_v > 7 else C.SARI
            klinik_risk = tm.get("klinik_risk_sayisi", 0)

            satirlar = [
                ("Kadans",           f"{kadans_v:.0f} adim/dk",                       kadans_renk),
                ("Kadans Varyab.",   f"{tm.get('kadans_varyabilite', 0):.3f} s",       C.SARI),
                ("Hiz",              f"{hiz_v:.1f} cm/s",                              hiz_renk),
                ("Adim Varyab.",     f"{tm.get('adim_varyabilite', 0):.2f} cm",        C.SARI),
                ("FOG Durumu",       fog_metin,                                        fog_renk),
                ("Tremor (3.5-7Hz)", tremor_metin,                                     tremor_renk),
                ("Kol Asimetri",     f"{asim_v:.1f} %",                                asim_renk),
                ("Govde Egimi",      f"{egim_v:.1f} derece",                           egim_renk),
                ("Klinik Risk",      f"{klinik_risk}/8 parametre",
                    C.KIRMIZI if klinik_risk >= 3 else (C.TURUNCU if klinik_risk >= 1 else C.YESIL)),
            ]
            for i, (baslik, deger, renk) in enumerate(satirlar):
                cv2.putText(img, f"{baslik}: {deger}",
                            (10, h - 200 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, renk, 1)

        if uyari:
            cv2.putText(img, f"! {uyari}", (10, h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C.KIRMIZI, 1)
        return img


# =========================================================================
# Veri Logger (CSV + Video)
# =========================================================================
class VeriLogger:
    CSV_ALANLAR = [
        "seans_no", "kullanici_id", "timestamp", "tarih_saat", "frame",
        "tahmin", "guven_pct",
        # Anlik (ParkinsonAnalizor.hesapla() ciktilari)
        "arm_rel_L", "arm_rel_R", "adim_uzunlugu_cm", "govde_egimi",
        "kalca_merkez_z",
        # Temporal & Klinik (ParkinsonTemporal.metrikler() ciktilari)
        "kol_genlik_L_ort", "kol_genlik_R_ort", "kol_genlik_ort", "kol_genlik_min",
        "kol_asimetri_ort", "adim_uzunlugu_ort", "adim_varyabilite",
        "kadans_spm", "kadans_varyabilite",
        "govde_egimi_ort", "yuruyus_hizi_cms",
        "fog_skoru", "tremor_hz_L", "tremor_hz_R", "tremor_hz_max",
        # Hibrit Klinik Karar
        "klinik_risk_sayisi",
        # Gercek Klinik Etiket (Ground Truth: -1: Bilinmiyor, 0: Saglikli, 1: Parkinson)
        "klinik_etiket",
    ]

    def __init__(self, seans_no: int, veri_dizini: str, video_kaydet: bool):
        self.seans_no = seans_no
        self.video_kaydet = video_kaydet
        self.frame_sayaci = 0
        os.makedirs(veri_dizini, exist_ok=True)
        self._csv_yolu = os.path.join(veri_dizini, "parkinson_master.csv")
        self._header_yazildi = os.path.exists(self._csv_yolu) and os.path.getsize(self._csv_yolu) > 0
        self._csv_f = open(self._csv_yolu, "a", newline="", encoding="utf-8")
        self._csv_w: Optional[csv.DictWriter] = None
        seans_kl = os.path.join(veri_dizini, f"seans_{seans_no:03d}")
        os.makedirs(seans_kl, exist_ok=True)
        self._seans_csv_yolu = os.path.join(seans_kl, "seans_verileri.csv")
        self._seans_csv_f = open(self._seans_csv_yolu, "w", newline="", encoding="utf-8")
        self._seans_csv_w: Optional[csv.DictWriter] = None

        self._video_yolu = os.path.join(seans_kl, "video.mp4")
        self._video_w: Optional[cv2.VideoWriter] = None

    def kaydet(self, frame: np.ndarray, fv: dict, tm: dict,
               tahmin: str, guven: float, ts: float):
        self.frame_sayaci += 1
        if self.video_kaydet:
            fh, fw = frame.shape[:2]
            if self._video_w is None:
                self._video_w = cv2.VideoWriter(
                    self._video_yolu, cv2.VideoWriter_fourcc(*"mp4v"), 20, (fw, fh))
            self._video_w.write(frame)

        if not CFG.CSV_KAYDET:
            return
        satir = {
            "seans_no": self.seans_no, "kullanici_id": CFG.KULLANICI_ID,
            "timestamp": f"{ts:.3f}",
            "tarih_saat": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
            "frame": self.frame_sayaci, "tahmin": tahmin, "guven_pct": f"{guven:.1f}",
            "klinik_etiket": getattr(CFG, "KLINIK_ETIKET", -1),
        }
        for k, v in {**fv, **tm}.items():
            satir[k] = f"{v:.4f}" if isinstance(v, float) else v

        if self._csv_w is None:
            self._csv_w = csv.DictWriter(self._csv_f, fieldnames=self.CSV_ALANLAR, extrasaction="ignore")
            if not self._header_yazildi:
                self._csv_w.writeheader()
                self._header_yazildi = True
        self._csv_w.writerow(satir)

        # Seans bazli ozel CSV'ye de kaydet
        if self._seans_csv_w is None:
            self._seans_csv_w = csv.DictWriter(self._seans_csv_f, fieldnames=self.CSV_ALANLAR, extrasaction="ignore")
            self._seans_csv_w.writeheader()
        self._seans_csv_w.writerow(satir)

    def kapat(self):
        self._csv_f.flush()
        self._csv_f.close()
        self._seans_csv_f.flush()
        self._seans_csv_f.close()
        if self._video_w:
            self._video_w.release()


# =========================================================================
# Yuz Bulaniklastirma (KVKK)
# =========================================================================
def yuz_gizle(frame: np.ndarray, lm, w: int, h: int):
    xs = [int(lm[i].x * w) for i in range(11)]
    ys = [int(lm[i].y * h) for i in range(11)]
    kw, kh = max(xs) - min(xs), max(ys) - min(ys)
    x1 = max(0, min(xs) - int(kw * 0.5) - 20)
    x2 = min(w, max(xs) + int(kw * 0.5) + 20)
    y1 = max(0, min(ys) - int(kh * 0.5) - 20)
    y2 = min(h, max(ys) + int(kh * 1.5) + 20)
    if y2 > y1 and x2 > x1:
        roi = frame[y1:y2, x1:x2]
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (61, 61), 0)


# =========================================================================
# Tkinter Giris Ekrani
# =========================================================================
# =========================================================================
# Tkinter Giris Ekrani
# =========================================================================
def programi_baslat(id_ent, boy_ent, cam_ent, rot_var, blur_var, durum_cb, root):
    k_id = id_ent.get().strip()
    k_boy = boy_ent.get().strip()
    k_cam = cam_ent.get().strip()
    durum_str = durum_cb.get()

    if not k_id or not k_boy:
        messagebox.showerror("Hata", "Lütfen Hasta ID ve Boy alanlarını doldurun!")
        return
    try:
        boy_f = float(k_boy)
        if boy_f < 50 or boy_f > 250:
            raise ValueError
    except ValueError:
        messagebox.showerror("Hata", "Geçerli bir boy girin (50 - 250 cm arası)!")
        return

    # Kamera kaynağını belirle
    if k_cam.isdigit():
        CFG.KAMERA_KAYNAK = int(k_cam)
    else:
        CFG.KAMERA_KAYNAK = k_cam if k_cam else 0

    CFG.KULLANICI_ID = k_id
    CFG.KULLANICI_BOYU_CM = boy_f
    CFG.KAMERA_DONDUR = bool(rot_var.get())
    CFG.YUZU_GIZLE = bool(blur_var.get())

    # Klinik Etiket Belirle (Ground Truth)
    if "Sağlıklı" in durum_str:
        CFG.KLINIK_ETIKET = 0
    elif "Parkinson" in durum_str:
        CFG.KLINIK_ETIKET = 1
    else:
        CFG.KLINIK_ETIKET = -1

    root.destroy()
    analiz_baslat()


def video_dosyasi_sec(cam_ent):
    dosya = filedialog.askopenfilename(
        title="Yürüyüş Video Dosyası Seç",
        filetypes=[("Video Dosyaları", "*.mp4 *.avi *.mov *.mkv"), ("Tüm Dosyalar", "*.*")]
    )
    if dosya:
        cam_ent.delete(0, tk.END)
        cam_ent.insert(0, dosya)


def gecmis_sil():
    if not os.path.exists(CFG.VERI_DIZINI):
        messagebox.showinfo("Bilgi", "Silinecek veri yok!")
        return
    if messagebox.askyesno("Uyarı", "Tüm seans verileri kalıcı olarak silinecek.\nEmin misiniz?"):
        try:
            shutil.rmtree(CFG.VERI_DIZINI)
            os.makedirs(CFG.VERI_DIZINI, exist_ok=True)
            messagebox.showinfo("Başarılı", "Tüm seanslar silindi!")
        except Exception as e:
            messagebox.showerror("Hata", f"Silme hatası: {e}")


def giris_ekrani():
    root = tk.Tk()
    root.title("NYTAS-Parkinson | Giriş & Yapılandırma")
    root.geometry("530x650")
    root.eval('tk::PlaceWindow . center')
    root.configure(bg="#1a1a2e")

    tk.Label(root, text="NYTAS — Parkinson Modülü",
             font=("Helvetica", 16, "bold"), bg="#1a1a2e", fg="#e94560").pack(pady=(16, 4))
    tk.Label(root, text="TÜBİTAK 1002 Hızlı Destek Programı",
             font=("Helvetica", 9), bg="#1a1a2e", fg="#888").pack()

    frm = tk.Frame(root, bg="#1a1a2e")
    frm.pack(pady=12, padx=20, fill="x")

    # Hasta ID
    tk.Label(frm, text="Hasta ID:", font=("Helvetica", 10, "bold"), bg="#1a1a2e", fg="#eee").grid(row=0, column=0, pady=6, sticky="w")
    id_ent = tk.Entry(frm, font=("Helvetica", 10), width=24)
    id_ent.insert(0, "hasta_001")
    id_ent.grid(row=0, column=1, pady=6, padx=10, sticky="w")

    # Boy
    tk.Label(frm, text="Boy (cm):", font=("Helvetica", 10, "bold"), bg="#1a1a2e", fg="#eee").grid(row=1, column=0, pady=6, sticky="w")
    boy_ent = tk.Entry(frm, font=("Helvetica", 10), width=24)
    boy_ent.insert(0, "175.0")
    boy_ent.grid(row=1, column=1, pady=6, padx=10, sticky="w")

    # Klinik Durum / Etiket (Ground Truth)
    tk.Label(frm, text="Klinik Durum:", font=("Helvetica", 10, "bold"), bg="#1a1a2e", fg="#38bdf8").grid(row=2, column=0, pady=6, sticky="w")
    durum_cb = ttk.Combobox(
        frm,
        values=[
            "Bilinmiyor / Rutin Tarama",
            "Sağlıklı Kontrol Grubu (Referans)",
            "Tanı Almış Parkinson Hastası"
        ],
        state="readonly",
        font=("Helvetica", 9),
        width=25
    )
    durum_cb.current(0)
    durum_cb.grid(row=2, column=1, pady=6, padx=10, sticky="w")

    # Kamera Kaynağı
    tk.Label(frm, text="Kamera Kaynağı:", font=("Helvetica", 10, "bold"), bg="#1a1a2e", fg="#eee").grid(row=3, column=0, pady=6, sticky="w")
    cam_ent = tk.Entry(frm, font=("Helvetica", 10), width=24)
    cam_ent.insert(0, "0")
    cam_ent.grid(row=3, column=1, pady=6, padx=10, sticky="w")

    btn_file = tk.Button(frm, text="Video Seç", font=("Helvetica", 8, "bold"), bg="#2c3e50", fg="white",
                         command=lambda: video_dosyasi_sec(cam_ent))
    btn_file.grid(row=3, column=2, pady=6, padx=2)

    cam_info = "(0: Dahili Web Cam, 1: Harici Cam, veya IP URL / Video Yolu)"
    tk.Label(frm, text=cam_info, font=("Helvetica", 8), bg="#1a1a2e", fg="#aaa").grid(row=4, column=0, columnspan=3, sticky="w", pady=(0, 6))

    # Checkboxlar
    rot_var = tk.IntVar(value=0)
    tk.Checkbutton(frm, text="Telefon Dikey Çekim Modu (90° Döndür)", variable=rot_var,
                   font=("Helvetica", 9), bg="#1a1a2e", fg="#eee",
                   selectcolor="#333", activebackground="#1a1a2e").grid(row=5, column=0, columnspan=3, pady=4, sticky="w")

    blur_var = tk.IntVar(value=1)
    tk.Checkbutton(frm, text="Yüzü Gizle / Bulanıklaştır (KVKK Uyumlu)", variable=blur_var,
                   font=("Helvetica", 9), bg="#1a1a2e", fg="#eee",
                   selectcolor="#333", activebackground="#1a1a2e").grid(row=6, column=0, columnspan=3, pady=4, sticky="w")

    uyari = (
        "ÖNEMLİ BİLGİLENDİRME:\n"
        "• Kameraya dik açıda ve tam boy görünecek şekilde yürüyün.\n"
        "• Kollarınızı doğal salınımında rahat bırakın.\n"
        "• Önerilen analiz süresi: En az 30 saniye.\n"
        "• Çıkış için video penceresinde 'Q' tuşuna basın."
    )
    tk.Label(root, text=uyari, fg="#e94560", bg="#1a1a2e",
             font=("Helvetica", 8), justify="left").pack(pady=6, padx=20, fill="x")

    btn_frm = tk.Frame(root, bg="#1a1a2e")
    btn_frm.pack(pady=10)

    tk.Button(btn_frm, text="▶ Analizi Başlat", font=("Helvetica", 11, "bold"),
              bg="#0f3460", fg="white", width=22, height=1,
              command=lambda: programi_baslat(id_ent, boy_ent, cam_ent, rot_var, blur_var, durum_cb, root)).grid(row=0, column=0, pady=5)
    tk.Button(btn_frm, text="🗑 Geçmiş Seansları Sil", font=("Helvetica", 9),
              bg="#c0392b", fg="white", width=22,
              command=gecmis_sil).grid(row=1, column=0, pady=5)

    root.mainloop()


# =========================================================================
# Ana Analiz Dongusu
# =========================================================================
def analiz_baslat():
    seans_no = _sonraki_seans_no(CFG.VERI_DIZINI)
    log_kl = os.path.join(CFG.VERI_DIZINI, f"seans_{seans_no:03d}")
    log = _log_kur(seans_no, log_kl)

    log.info("=" * 55)
    log.info("  NYTAS-PARKINSON | Yuruyus Analiz Modulu")
    log.info("=" * 55)
    log.info(f"  Hasta       : {CFG.KULLANICI_ID}")
    log.info(f"  Boy         : {CFG.KULLANICI_BOYU_CM} cm")
    log.info(f"  Kamera      : {CFG.KAMERA_KAYNAK}")
    log.info(f"  Dondurme    : {'Evet (90°)' if CFG.KAMERA_DONDUR else 'Hayir'}")
    log.info(f"  Seans       : #{seans_no:03d}")
    log.info(f"  Tarih       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Yuz Gizle   : {'Evet' if CFG.YUZU_GIZLE else 'Hayir'}")
    log.info("=" * 55)

    if not model_hazirla(log):
        messagebox.showerror("Model Hatası", f"MediaPipe modeli indirilemedi/bulunamadı:\n{MODEL_DOSYASI}")
        return

    ml_model = None
    try:
        ml_model = joblib.load(CFG.ML_MODEL_DOSYASI)
        log.info(f"ML model yuklendi: {type(ml_model).__name__}")
    except FileNotFoundError:
        log.warning("ML model bulunamadi -> sadece veri toplanacak")
    except Exception as e:
        log.error(f"ML model hatasi: {e}")

    try:
        cap = VideoStream(CFG.KAMERA_KAYNAK).start()
    except Exception as e:
        log.error(f"Kamera baslatilamadi: {e}")
        messagebox.showerror("Kamera Hatası", f"Kamera akışı başlatılamadı:\n{e}")
        return

    kamera_ok = False
    for _ in range(40):
        time.sleep(0.1)
        if cap.read() is not None:
            kamera_ok = True
            break
    if not kamera_ok:
        log.error("Kamera baglantisi kurulamadi!")
        cap.stop()
        messagebox.showerror(
            "Kamera Bağlantı Hatası",
            f"Kamera görüntüsü alınamadı!\n\n"
            f"Seçilen Kaynak: {CFG.KAMERA_KAYNAK}\n\n"
            "Olası Nedenler & Çözümler:\n"
            "1. Kameranız başka bir program (Zoom, Teams vb.) tarafından kullanılıyor olabilir.\n"
            "2. Dahili web kamera için kaynak kısmına '0' veya '1' yazmayı deneyin.\n"
            "3. IP kamera kullanıyorsanız URL adresinin doğruluğunu kontrol edin."
        )
        return
    log.info("Kamera hazir. Cikis: q tusu")

    analizor = ParkinsonAnalizor()
    temporal = ParkinsonTemporal(buf=CFG.BUFFER_BOYUTU, fps=CFG.KAMERA_FPS)
    veri_logger = VeriLogger(seans_no, CFG.VERI_DIZINI, CFG.VIDEO_KAYDET)

    tahmin_sonucu = "Veri Toplaniyor..."
    tahmin_guveni = 0.0
    uyari = ""
    fps_sayaci, fps_ref, anlik_fps = 0, time.time(), 0.0
    frame_ts_ms = 0

    with open(MODEL_DOSYASI, "rb") as f:
        model_buffer = f.read()

    base_opts = mp.tasks.BaseOptions(model_asset_buffer=model_buffer)
    pose_opts = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_pose_detection_confidence=CFG.MIN_DETECTION_CONF,
        min_pose_presence_confidence=CFG.MIN_TRACKING_CONF,
        num_poses=1,
    )

    seans_baslangic = time.time()

    with mp.tasks.vision.PoseLandmarker.create_from_options(pose_opts) as landmarker:
        while True:
            frame = cap.read()
            if frame is None:
                time.sleep(0.01)
                continue

            if CFG.KAMERA_DONDUR:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (CFG.FRAME_GENISLIK, CFG.FRAME_YUKSEKLIK))
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts_ms += 1

            try:
                results = landmarker.detect_for_video(mp_img, frame_ts_ms)
            except Exception as e:
                log.error(f"Pose hatasi: {e}")
                continue

            fv: Optional[dict] = None
            tm: Optional[dict] = None

            if results.pose_landmarks and results.pose_world_landmarks:
                lm = results.pose_landmarks[0]
                world_lm = results.pose_world_landmarks[0]

                # KVKK: Yuz gizleme
                if CFG.YUZU_GIZLE:
                    yuz_gizle(frame, lm, w, h)

            img = frame.copy()

            if results.pose_landmarks and results.pose_world_landmarks:
                ts = time.time()
                fv_raw, uyari = analizor.hesapla(lm, world_lm, h, w, ts)

                if fv_raw is not None:
                    fv = fv_raw
                    temporal.guncelle(fv, ts)
                    tm = temporal.metrikler()

                    # ML Tahmin + Hibrit Klinik Karar (Parkinson modeli yukluyse)
                    klinik_risk_sayisi = 0
                    if ml_model is not None and tm:
                        try:
                            # 8 odak biyobelirtec — tumu temporal metriklerden (tutarli)
                            pred_kol_asimetri   = tm.get("kol_asimetri_ort", 0.0)
                            pred_adim_uzunlugu  = tm.get("adim_uzunlugu_ort", 0.0)
                            pred_govde_egimi    = tm.get("govde_egimi_ort", 0.0)
                            pred_kadans         = tm.get("kadans_spm", 0.0)
                            pred_fog            = tm.get("fog_skoru", 0.0)
                            pred_hiz            = tm.get("yuruyus_hizi_cms", 0.0)
                            pred_tremor         = tm.get("tremor_hz_max", 0.0)
                            pred_kol_genlik     = tm.get("kol_genlik_min", 0.0)

                            df_pred = pd.DataFrame([[
                                pred_kol_asimetri, pred_adim_uzunlugu,
                                pred_govde_egimi, pred_kadans,
                                pred_fog, pred_hiz,
                                pred_tremor, pred_kol_genlik,
                            ]], columns=[
                                "kol_asimetri", "adim_uzunlugu_cm",
                                "govde_egimi", "kadans_spm",
                                "fog_skoru", "yuruyus_hizi_cms",
                                "tremor_hz", "kol_genlik_ort",
                            ])
                            kod = ml_model.predict(df_pred)[0]
                            if hasattr(ml_model, "predict_proba"):
                                tahmin_guveni = float(ml_model.predict_proba(df_pred)[0][kod]) * 100
                            else:
                                tahmin_guveni = 0.0
                            ml_sonuc = "NORMAL" if kod == 0 else "PARKINSON RISKI"

                            # --- Hibrit Klinik Kural Motoru ---
                            # 8 biyobelirtecin literatur esikleriyle karsilastirilmasi
                            klinik_risk_sayisi = 0
                            if pred_kol_asimetri > 12.0:    klinik_risk_sayisi += 1  # >%12
                            if pred_adim_uzunlugu < 50.0:   klinik_risk_sayisi += 1  # <50cm
                            if pred_govde_egimi > 7.0:      klinik_risk_sayisi += 1  # >7 derece
                            if pred_kadans > 0 and pred_kadans < 80.0:  klinik_risk_sayisi += 1  # <80 adim/dk
                            if pred_hiz > 0 and pred_hiz < 90.0:        klinik_risk_sayisi += 1  # <90 cm/s
                            if pred_kol_genlik > 0 and pred_kol_genlik < 14.0:  klinik_risk_sayisi += 1  # <14cm
                            if pred_tremor > 3.0:           klinik_risk_sayisi += 1  # >3 Hz
                            if pred_fog > 0.4:              klinik_risk_sayisi += 1  # >0.4

                            # Final karar: ML veya Klinik Kural (3+ parametre riskli)
                            if ml_sonuc == "PARKINSON RISKI" or klinik_risk_sayisi >= 3:
                                tahmin_sonucu = "PARKINSON RISKI"
                            else:
                                tahmin_sonucu = "NORMAL"

                        except Exception as e:
                            log.error(f"Tahmin hatasi: {e}")

                    # Klinik risk sayisini temporal metriklere ekle (CSV kaydı icin)
                    if tm:
                        tm["klinik_risk_sayisi"] = klinik_risk_sayisi

                    veri_logger.kaydet(img, fv, tm if tm else {},
                                       tahmin_sonucu, tahmin_guveni, ts)

                skeleton_ciz(img, lm, w, h)
            else:
                uyari = "Pose tespit edilemiyor"

            # FPS
            fps_sayaci += 1
            now = time.time()
            elapsed = now - fps_ref
            if elapsed >= 2.0:
                anlik_fps = fps_sayaci / elapsed
                fps_sayaci, fps_ref = 0, now

            gecen_sure_sn = time.time() - seans_baslangic
            img = ParkinsonHUD.ciz(img, fv, tm, tahmin_sonucu, tahmin_guveni, seans_no, uyari, gecen_sure_sn=gecen_sure_sn)
            cv2.putText(img, f"FPS:{anlik_fps:.0f}", (w - 70, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ParkinsonHUD.GRI, 1)
            cv2.imshow(f"NYTAS-Parkinson | Seans #{seans_no:03d}", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.stop()
    veri_logger.kapat()
    cv2.destroyAllWindows()
    log.info(f"Seans #{seans_no:03d} tamamlandi. Toplam kare: {veri_logger.frame_sayaci}")
    log.info(f"Veriler: {CFG.VERI_DIZINI}/parkinson_master.csv")

    # Otomatik Medikal Klinik Raporu Derle ve Aç
    try:
        from kaynak_kodlar.rapor_olusturucu import rapor_olustur
        log.info(f"Klinik hekim raporu derleniyor (Seans #{seans_no:03d})...")
        rapor_dosyasi = rapor_olustur(seans_no, CFG.VERI_DIZINI, otomatik_ac=True)
        if rapor_dosyasi:
            log.info(f"Klinik rapor hazirlandi ve acildi: {rapor_dosyasi}")
    except Exception as e:
        log.error(f"Klinik rapor olusturulamadi: {e}")


# =========================================================================
if __name__ == "__main__":
    if "--direct" in sys.argv:
        import argparse
        parser = argparse.ArgumentParser(description="NYTAS-Parkinson Yürüyüş Analizi")
        parser.add_argument("--direct", action="store_true")
        parser.add_argument("--id", default="hasta_001")
        parser.add_argument("--boy", type=float, default=175.0)
        parser.add_argument("--kamera", default="0")
        parser.add_argument("--dondur", action="store_true")
        parser.add_argument("--no-blur", action="store_true")
        parser.add_argument("--etiket", type=int, default=-1, choices=[-1, 0, 1], help="-1: Bilinmiyor, 0: Saglikli, 1: Parkinson")
        args = parser.parse_args()

        CFG.KULLANICI_ID = args.id
        CFG.KULLANICI_BOYU_CM = args.boy
        CFG.KAMERA_KAYNAK = int(args.kamera) if str(args.kamera).isdigit() else args.kamera
        CFG.KAMERA_DONDUR = args.dondur
        CFG.YUZU_GIZLE = not args.no_blur
        CFG.KLINIK_ETIKET = args.etiket
        analiz_baslat()
    else:
        giris_ekrani()
