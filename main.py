"""
=============================================================================
NYTAS-PARKINSON™ v1.2 — Profesyonel Klinik Destek & Yürüyüş Analiz Platformu
TÜBİTAK 1002 Hızlı Destek Programı
=============================================================================
"""

import os
import sys
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path

# Proje kök dizini
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

try:
    from ayarlar.settings import (
        MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, DOCS_DIR,
        POSE_LANDMARKER_PATH, PARKINSON_MODEL_PATH, DATASET_NORMAL_PATH, DATASET_PARKINSON_PATH
    )
except ImportError:
    try:
        from config.settings import (
            MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, DOCS_DIR,
            POSE_LANDMARKER_PATH, PARKINSON_MODEL_PATH, DATASET_NORMAL_PATH, DATASET_PARKINSON_PATH
        )
    except ImportError:
        MODELS_DIR = BASE_DIR / "modeller"
        RAW_DATA_DIR = BASE_DIR / "veriler" / "raw"
        PROCESSED_DATA_DIR = BASE_DIR / "veriler" / "processed" / "nytas_parkinson_veri"
        DOCS_DIR = BASE_DIR / "dokumanlar"
        POSE_LANDMARKER_PATH = MODELS_DIR / "pose_landmarker.task"
        PARKINSON_MODEL_PATH = MODELS_DIR / "parkinson_model.pkl"
        DATASET_NORMAL_PATH = RAW_DATA_DIR / "dataset_normal_parkinson.csv"
        DATASET_PARKINSON_PATH = RAW_DATA_DIR / "dataset_parkinson_parkinson.csv"


def python_executable():
    in_venv = (sys.prefix != sys.base_prefix)
    if not in_venv:
        venv_py = BASE_DIR / ".venv" / "Scripts" / "python.exe" if os.name == 'nt' else BASE_DIR / ".venv" / "bin" / "python"
        if venv_py.exists():
            return str(venv_py)
    return sys.executable


def klasor_ac(yol: Path):
    yol.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        os.startfile(str(yol))
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', str(yol)])
    else:
        subprocess.Popen(['xdg-open', str(yol)])


class NYTASGui(tk.Tk):
    # Modern Medical Tech Renk Paleti
    C_BG = "#0f172a"          # Çok Koyu Lacivert Slate
    C_CARD = "#1e293b"        # Kart Arka Planı
    C_CARD_LIGHT = "#334155"  # Açık Kart / Çerçeve
    C_ACCENT = "#0284c7"      # Parlak Mavi
    C_ACCENT_HOVER = "#0369a1"
    C_CYAN = "#38bdf8"        # Açık Cyan
    C_GREEN = "#10b981"       # Yeşil
    C_AMBER = "#f59e0b"       # Sarı / Turuncu
    C_RED = "#ef4444"         # Kırmızı
    C_TEXT = "#f8fafc"        # Ana Metin Beyaz
    C_MUTED = "#94a3b8"       # İkincil Metin

    def __init__(self):
        super().__init__()

        self.title("NYTAS-PARKINSON™ v1.3 — Yürüyüş Analiz & Yapay Zeka Platformu")

        # Ekran boyutuna göre dinamik geometri
        try:
            sw = self.winfo_screenwidth()
            sh = self.winfo_screenheight()
            init_w = min(1180, max(1000, int(sw * 0.90)))
            init_h = min(860, max(720, int(sh * 0.88)))
            self.geometry(f"{init_w}x{init_h}")
        except Exception:
            self.geometry("1060x800")

        self.minsize(920, 600)
        self.configure(bg=self.C_BG)

        # Windows'ta tam ekran / maksimize olarak başlat (tüm sayfa otomatik sığsın)
        if os.name == 'nt':
            try:
                self.state('zoomed')
            except Exception:
                pass

        self._stil_yapilandir()
        self._ust_banner_olustur()
        self._sekmeleri_olustur()
        self._alt_durum_cubugu()

        # İlk durum kontrolü
        self.after(200, self._durum_gostergelerini_guncelle)

    def _stil_yapilandir(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        # Notebook Stili
        style.configure("TNotebook", background=self.C_BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background="#1e293b",
            foreground="#94a3b8",
            padding=[18, 9],
            font=("Segoe UI", 10, "bold"),
            borderwidth=0
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#0284c7")],
            foreground=[("selected", "#ffffff")]
        )

        # Scrollbar
        style.configure("Vertical.TScrollbar", background="#1e293b", troughcolor="#0f172a", borderwidth=0)

    def _ust_banner_olustur(self):
        banner = tk.Frame(self, bg="#1e293b", padx=20, pady=10, highlightthickness=1, highlightbackground="#334155")
        banner.pack(fill="x", side="top", padx=12, pady=(6, 4))

        ust_satir = tk.Frame(banner, bg="#1e293b")
        ust_satir.pack(fill="x")

        # Sol Başlık
        sol_baslik = tk.Frame(ust_satir, bg="#1e293b")
        sol_baslik.pack(side="left")

        logo_lbl = tk.Label(
            sol_baslik,
            text="🧬 NYTAS-PARKINSON™",
            font=("Segoe UI", 16, "bold"),
            fg="#38bdf8",
            bg="#1e293b"
        )
        logo_lbl.pack(side="left")

        surum_lbl = tk.Label(
            sol_baslik,
            text="v1.3  |  TÜBİTAK 1002 Hızlı Destek Programı",
            font=("Segoe UI", 9, "italic"),
            fg="#94a3b8",
            bg="#1e293b"
        )
        surum_lbl.pack(side="left", padx=12, pady=4)

        # Sağ Durum Rozetleri
        self.rozet_frame = tk.Frame(ust_satir, bg="#1e293b")
        self.rozet_frame.pack(side="right")

        self.lbl_rozet_pose = tk.Label(self.rozet_frame, text="Pose: ...", font=("Segoe UI", 8, "bold"),
                                       fg="#ffffff", bg="#334155", padx=8, pady=3)
        self.lbl_rozet_pose.pack(side="left", padx=4)

        self.lbl_rozet_ml = tk.Label(self.rozet_frame, text="ML Model: ...", font=("Segoe UI", 8, "bold"),
                                     fg="#ffffff", bg="#334155", padx=8, pady=3)
        self.lbl_rozet_ml.pack(side="left", padx=4)

        # Açıklama Metni Kutusu
        aciklama_kutu = tk.Frame(banner, bg="#0f172a", padx=12, pady=6, highlightthickness=1, highlightbackground="#334155")
        aciklama_kutu.pack(fill="x", pady=(6, 0))

        aciklama_metni = (
            "Hoş geldiniz! NYTAS-PARKINSON, kamera tabanlı yapay zeka (BlazePose 33 Landmark) ile "
            "Parkinson hastalığına özgü 8 dijital biyobelirteci (kol salınım asimetrisi, adım uzunluğu, donma, tremor vb.) "
            "canlı olarak analiz eden ve eğitilmiş makine öğrenmesi modelleri ile klinik risk değerlendirmesi sunan sistemdir."
        )
        aciklama = tk.Label(
            aciklama_kutu,
            text=aciklama_metni,
            font=("Segoe UI", 9),
            fg="#cbd5e1",
            bg="#0f172a",
            wraplength=900,
            justify="left"
        )
        aciklama.pack(anchor="w")

    def _sekmeleri_olustur(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=6)

        # Sekme 1: Canlı Analiz & Başlatıcı
        self.tab_analiz = tk.Frame(self.notebook, bg=self.C_BG, padx=12, pady=10)
        self.notebook.add(self.tab_analiz, text="🎥  Canlı Yürüyüş Analizi")
        self._olustur_sekme_analiz()

        # Sekme 2: Makine Öğrenmesi & Model Eğitimi
        self.tab_egitim = tk.Frame(self.notebook, bg=self.C_BG, padx=12, pady=10)
        self.notebook.add(self.tab_egitim, text="🤖  Model Eğitimi (ML)")
        self._olustur_sekme_egitim()

        # Sekme 3: Sistem Durumu & Dosyalar
        self.tab_durum = tk.Frame(self.notebook, bg=self.C_BG, padx=12, pady=10)
        self.notebook.add(self.tab_durum, text="🔍  Sistem & Dosya Yönetimi")
        self._olustur_sekme_durum()

        # Sekme 4: 8 Biyobelirteç Bilgisi
        self.tab_bilgi = tk.Frame(self.notebook, bg=self.C_BG, padx=12, pady=10)
        self.notebook.add(self.tab_bilgi, text="🧬  8 Dijital Biyobelirteç")
        self._olustur_sekme_bilgi()

    def _olustur_sekme_analiz(self):
        # Dikey Kaydırılabilir Canvas Sistemi (Her çözünürlükte tüm sayfa görünür & kaydırılabilir)
        canvas = tk.Canvas(self.tab_analiz, bg=self.C_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab_analiz, orient="vertical", command=canvas.yview)

        scrollable_frame = tk.Frame(canvas, bg=self.C_BG)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)

        # Fare tekerleği desteği
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Ana Kart Düzeni
        kart = tk.Frame(scrollable_frame, bg=self.C_CARD, padx=20, pady=16, highlightthickness=1, highlightbackground=self.C_CARD_LIGHT)
        kart.pack(fill="both", expand=True, padx=2, pady=2)

        lbl_baslik = tk.Label(
            kart,
            text="🎥 Canlı Yürüyüş Analizi & Klinik Seans Modülü",
            font=("Segoe UI", 12, "bold"),
            fg=self.C_CYAN,
            bg=self.C_CARD
        )
        lbl_baslik.pack(anchor="w", pady=(0, 6))

        aciklama = (
            "Bu modül, MediaPipe Pose Landmarker yapay zeka motorunu ve önceden eğitilmiş makine öğrenmesi modelini kullanarak "
            "hastanın yürüyüşünü gerçek zamanlı analiz eder; 8 dijital biyobelirteci (kol salınımı, asimetri, adım uzunluğu, donma, tremor vb.) "
            "hesaplar ve seans kaydı olarak 'data/processed/nytas_parkinson_veri/' altına arşivler."
        )
        lbl_aciklama = tk.Label(
            kart,
            text=aciklama,
            font=("Segoe UI", 9),
            fg="#cbd5e1",
            bg=self.C_CARD,
            wraplength=900,
            justify="left"
        )
        lbl_aciklama.pack(anchor="w", pady=(0, 10))

        # Yönergeler Kutusu
        yonerge_kutu = tk.Frame(kart, bg="#0f172a", padx=14, pady=8, highlightthickness=1, highlightbackground="#334155")
        yonerge_kutu.pack(fill="x", pady=(0, 12))

        yonege_metin = (
            "📌 TEST & ÖLÇÜM YÖNERGELERİ:\n"
            "1. 'Canlı Analiz Başlat' butonuna bastığınızda hasta bilgisi ve kamera seçim penceresi açılır (webcam 0, USB cam 1 veya IP url).\n"
            "2. Hastanın baştan ayağa tam boy göründüğünden ve kollarını serbest bıraktığından emin olun (en az 30 saniye önerilir).\n"
            "3. Analizi tamamlayıp kaydetmek ve klinik raporu otomatik açmak için video ekranında 'Q' tuşuna basınız."
        )
        lbl_yonerge = tk.Label(
            yonerge_kutu,
            text=yonege_metin,
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg="#0f172a",
            justify="left"
        )
        lbl_yonerge.pack(anchor="w")

        # Büyük CTA Butonu
        btn_baslat = tk.Button(
            kart,
            text="▶  CANLI ANALİZ VE SEANS GİRİŞİNİ BAŞLAT",
            font=("Segoe UI", 11, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            activebackground="#0369a1",
            activeforeground="#ffffff",
            relief="flat",
            cursor="hand2",
            pady=10,
            command=self._analizi_baslat
        )
        btn_baslat.pack(fill="x", pady=(0, 6))

        # En Son Seans Raporu Butonu
        btn_rapor = tk.Button(
            kart,
            text="📄  EN SON SEANSIN KLİNİK RAPORUNU GÖRÜNTÜLE (HTML/PDF)",
            font=("Segoe UI", 10, "bold"),
            bg="#059669",
            fg="#ffffff",
            activebackground="#047857",
            activeforeground="#ffffff",
            relief="flat",
            cursor="hand2",
            pady=8,
            command=self._son_raporu_ac
        )
        btn_rapor.pack(fill="x", pady=(0, 10))

        # Ayırıcı çizgi
        sep = tk.Frame(kart, bg="#334155", height=1)
        sep.pack(fill="x", pady=(2, 8))

        lbl_gelismis = tk.Label(
            kart,
            text="⚡ GELİŞMİŞ ANALİZ ARAÇLARI (VİDEO ANALİZİ & İKİ FAZLI TEST)",
            font=("Segoe UI", 9, "bold"),
            fg=self.C_CYAN,
            bg=self.C_CARD
        )
        lbl_gelismis.pack(anchor="w", pady=(0, 6))

        # 2 Sütunlu Panel (Yan yana modern kartlar)
        ikili_panel = tk.Frame(kart, bg=self.C_CARD)
        ikili_panel.pack(fill="x", pady=(0, 4))
        ikili_panel.columnconfigure(0, weight=1)
        ikili_panel.columnconfigure(1, weight=1)

        # Sol Kolon: Video Analiz Kartı
        kart_video = tk.Frame(ikili_panel, bg="#0f172a", padx=12, pady=10, highlightthickness=1, highlightbackground="#334155")
        kart_video.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        btn_video = tk.Button(
            kart_video,
            text="🎞  VİDEO DOSYASINDAN ANALİZ & RAPORLAMA",
            font=("Segoe UI", 10, "bold"),
            bg="#b45309",
            fg="#ffffff",
            activebackground="#92400e",
            activeforeground="#ffffff",
            relief="flat",
            cursor="hand2",
            pady=8,
            command=self._video_analiz_baslat
        )
        btn_video.pack(fill="x", pady=(0, 6))

        lbl_video_aciklama = tk.Label(
            kart_video,
            text=(
                "Örnek hasta yürüyüş videosunu (.mp4/.avi/.mov/.mkv) seçerek tek tıkla "
                "BlazePose analizi yapıp medikal HTML klinik raporu üretin."
            ),
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg="#0f172a",
            wraplength=410,
            justify="left"
        )
        lbl_video_aciklama.pack(fill="x")

        # Sağ Kolon: İki Fazlı Klinik Seans Kartı
        kart_iki_faz = tk.Frame(ikili_panel, bg="#0f172a", padx=12, pady=10, highlightthickness=1, highlightbackground="#334155")
        kart_iki_faz.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        btn_iki_faz = tk.Button(
            kart_iki_faz,
            text="🔬  İKİ FAZLI OTOMATİK KLİNİK SEANS",
            font=("Segoe UI", 10, "bold"),
            bg="#7c3aed",
            fg="#ffffff",
            activebackground="#6d28d9",
            activeforeground="#ffffff",
            relief="flat",
            cursor="hand2",
            pady=8,
            command=self._iki_fazli_seans_baslat
        )
        btn_iki_faz.pack(fill="x", pady=(0, 6))

        lbl_iki_faz_aciklama = tk.Label(
            kart_iki_faz,
            text=(
                "Faz 1 (İstirahat Tremor, 45s) ➔ Faz 2 (Yürüyüş Analizi, 60s) yönergeli akış. "
                "Otomatik geçiş, SPACE ile faz atlama ve seans sonu otomatik rapor."
            ),
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg="#0f172a",
            wraplength=410,
            justify="left"
        )
        lbl_iki_faz_aciklama.pack(fill="x")

    def _analizi_baslat(self):
        py_cmd = python_executable()
        cmd = [py_cmd, str(BASE_DIR / "kaynak_kodlar" / "nytas_parkinson.py")]

        try:
            self._alt_durum_mesaj("Canlı Yürüyüş Analiz ve Yapılandırma Penceresi Açılıyor...")
            subprocess.Popen(cmd)
        except Exception as e:
            messagebox.showerror("Başlatma Hatası", f"Analiz modülü başlatılamadı:\n{e}")

    def _son_raporu_ac(self):
        """En son tamamlanan seansın HTML klinik raporunu tarayıcıda açar."""
        import webbrowser
        import glob
        veri_klasoru = str(PROCESSED_DATA_DIR)
        
        # En yüksek numaralı seans klasörünü bul
        seans_klasorleri = sorted(glob.glob(os.path.join(veri_klasoru, "seans_*")), reverse=True)
        if not seans_klasorleri:
            messagebox.showinfo("Bilgi", "Henüz kayıtlı bir seans bulunmamaktadır!\nÖnce bir yürüyüş testi yapınız.")
            return

        for s_kl in seans_klasorleri:
            raporlar = glob.glob(os.path.join(s_kl, "klinik_rapor_seans_*.html"))
            if raporlar:
                webbrowser.open(f"file:///{os.path.abspath(raporlar[0])}")
                self._alt_durum_mesaj(f"Klinik Rapor Açıldı: {os.path.basename(raporlar[0])}")
                return

        # Rapor dosyası bulunamadıysa, en son seansı üretmeyi dene
        try:
            from kaynak_kodlar.rapor_olusturucu import rapor_olustur
            en_son_ad = os.path.basename(seans_klasorleri[0])
            s_no = int(en_son_ad.replace("seans_", ""))
            p = rapor_olustur(s_no, veri_klasoru, otomatik_ac=True)
            if p:
                self._alt_durum_mesaj(f"Klinik Rapor Üretildi ve Açıldı: Seans #{s_no:03d}")
            else:
                messagebox.showwarning("Uyarı", f"Seans #{s_no} için işlenebilir veri bulunamadı.")
        except Exception as e:
            messagebox.showerror("Hata", f"Rapor oluşturulurken hata: {e}")

    def _video_analiz_baslat(self):
        """Seçilen video dosyasını analiz modülüyle işler ve HTML raporu üretir."""
        from tkinter import filedialog
        dosya = filedialog.askopenfilename(
            title="Analiz Edilecek Yürüyüş Videosu Seçin",
            filetypes=[
                ("Video Dosyaları", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        if not dosya:
            return

        # Hasta bilgisi isteme diyaloğu
        bilgi_win = tk.Toplevel(self)
        bilgi_win.title("Video Analizi — Hasta Bilgileri")
        bilgi_win.geometry("400x300")
        bilgi_win.configure(bg="#0f172a")
        bilgi_win.grab_set()
        bilgi_win.resizable(False, False)

        tk.Label(bilgi_win, text="🎞 Video Analizi Hasta Bilgileri",
                 font=("Segoe UI", 11, "bold"), fg="#38bdf8", bg="#0f172a").pack(pady=(16, 4))
        tk.Label(bilgi_win, text=f"📁 {Path(dosya).name}",
                 font=("Segoe UI", 8), fg="#94a3b8", bg="#0f172a").pack(pady=(0, 10))

        frm = tk.Frame(bilgi_win, bg="#1e293b", padx=16, pady=12)
        frm.pack(padx=16, fill="x")

        tk.Label(frm, text="Hasta ID:", font=("Segoe UI", 9), bg="#1e293b", fg="#e2e8f0").grid(
            row=0, column=0, sticky="w", pady=5)
        id_ent = tk.Entry(frm, font=("Segoe UI", 9), width=20,
                          bg="#0f172a", fg="#f1f5f9", insertbackground="white")
        id_ent.insert(0, "video_hasta_001")
        id_ent.grid(row=0, column=1, pady=5, padx=8)

        tk.Label(frm, text="Boy (cm):", font=("Segoe UI", 9), bg="#1e293b", fg="#e2e8f0").grid(
            row=1, column=0, sticky="w", pady=5)
        boy_ent = tk.Entry(frm, font=("Segoe UI", 9), width=20,
                           bg="#0f172a", fg="#f1f5f9", insertbackground="white")
        boy_ent.insert(0, "175.0")
        boy_ent.grid(row=1, column=1, pady=5, padx=8)

        def basla():
            k_id  = id_ent.get().strip() or "video_hasta_001"
            k_boy = boy_ent.get().strip()
            try:
                boy_f = float(k_boy)
                if boy_f < 50 or boy_f > 250:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Hata", "Geçerli bir boy girin (50–250 cm)!", parent=bilgi_win)
                return

            bilgi_win.destroy()
            py_cmd = python_executable()
            cmd = [
                py_cmd,
                str(BASE_DIR / "kaynak_kodlar" / "nytas_parkinson.py"),
                "--direct",
                "--id",     k_id,
                "--boy",    str(boy_f),
                "--kamera", dosya,
            ]
            try:
                self._alt_durum_mesaj(f"Video Analizi Başlatıldı: {Path(dosya).name}")
                subprocess.Popen(cmd)
            except Exception as exc:
                messagebox.showerror("Hata", f"Video analiz modülü başlatılamadı:\n{exc}")

        tk.Button(bilgi_win, text="▶  Video Analizini Başlat",
                  font=("Segoe UI", 10, "bold"), bg="#b45309", fg="white",
                  relief="flat", cursor="hand2", pady=8,
                  command=basla).pack(fill="x", padx=16, pady=(14, 4))
        tk.Button(bilgi_win, text="İptal",
                  font=("Segoe UI", 9), bg="#334155", fg="white",
                  relief="flat", cursor="hand2",
                  command=bilgi_win.destroy).pack(fill="x", padx=16, pady=(0, 10))

    def _iki_fazli_seans_baslat(self):
        """İki fazlı klinik seans modülünü ayrı process olarak başlatır."""
        py_cmd = python_executable()
        cmd = [py_cmd, str(BASE_DIR / "kaynak_kodlar" / "iki_fazli_seans.py"), "--gui"]
        try:
            self._alt_durum_mesaj("İki Fazlı Klinik Seans Yapılandırma Penceresi Açılıyor...")
            subprocess.Popen(cmd)
        except Exception as e:
            messagebox.showerror("Başlatma Hatası",
                                 f"İki Fazlı Seans modülü başlatılamadı:\n{e}\n\n"
                                 "Lütfen kaynak_kodlar/iki_fazli_seans.py dosyasının mevcut olduğundan emin olun.")



    def _olustur_sekme_egitim(self):
        ust_panel = tk.Frame(self.tab_egitim, bg=self.C_BG)
        ust_panel.pack(fill="x", pady=(0, 10))

        secenekler = [
            ("📊 Referans CSV ile Eğit",
             "veriler/raw/ altındaki hazır normal ve Parkinson verileriyle Random Forest / Gradient Boosting modelini eğitir.",
             "#059669", ["kaynak_kodlar/model_egitim_parkinson.py"]),
            ("📈 Canlı Seans Verileri ile Eğit",
             "Toplanan nytas_parkinson_veri/parkinson_master.csv gerçek hasta kayıtlarını kullanarak modeli günceller.",
             "#d97706", ["kaynak_kodlar/model_egitim_parkinson.py", "--master-csv"]),
            ("🧪 Sentetik Veri ile Test Et",
             "Literatür biyobelirteç aralıklarına göre rastgele sentetik veri üreterek algoritmayı simüle eder.",
             "#7c3aed", ["kaynak_kodlar/model_egitim_parkinson.py", "--sentetik"]),
        ]

        for title, desc, color, args in secenekler:
            card = tk.Frame(ust_panel, bg=self.C_CARD, padx=14, pady=10, highlightthickness=1, highlightbackground=self.C_CARD_LIGHT)
            card.pack(side="left", fill="both", expand=True, padx=4)

            btn = tk.Button(
                card, text=title, font=("Segoe UI", 10, "bold"), fg="#ffffff", bg=color,
                activebackground="#1e293b", activeforeground="#ffffff", relief="flat", cursor="hand2", pady=6,
                command=lambda a=args, t=title: self._model_egit_calistir(a, t)
            )
            btn.pack(fill="x", pady=(0, 6))

            lbl = tk.Label(card, text=desc, font=("Segoe UI", 8), fg=self.C_MUTED, bg=self.C_CARD, wraplength=260, justify="left")
            lbl.pack(fill="x")

        # Canlı Çıktı Terminali
        lbl_term = tk.Label(self.tab_egitim, text="📋 Model Eğitim & Doğrulama Terminal Çıktısı:",
                            font=("Segoe UI", 10, "bold"), fg=self.C_CYAN, bg=self.C_BG)
        lbl_term.pack(anchor="w", pady=(6, 4))

        term_frame = tk.Frame(self.tab_egitim, bg="#000000", highlightthickness=1, highlightbackground=self.C_CARD_LIGHT)
        term_frame.pack(fill="both", expand=True)

        sc = ttk.Scrollbar(term_frame)
        sc.pack(side="right", fill="y")

        self.txt_egitim_log = tk.Text(term_frame, font=("Consolas", 9), bg="#050811", fg="#38bdf8",
                                      yscrollcommand=sc.set, relief="flat", padx=10, pady=10)
        self.txt_egitim_log.pack(side="left", fill="both", expand=True)
        sc.config(command=self.txt_egitim_log.yview)
        self.txt_egitim_log.insert("1.0", "Model eğitimi başlatıldığında doğruluk (accuracy), AUC-ROC ve özellik önemlilikleri burada görünecektir.\n")

    def _model_egit_calistir(self, args_list, baslik):
        py_cmd = python_executable()
        full_cmd = [py_cmd] + [str(BASE_DIR / a) if not a.startswith("-") else a for a in args_list]

        def target():
            self._alt_durum_mesaj(f"Model Eğitiliyor: {baslik}...")
            self.txt_egitim_log.delete("1.0", tk.END)
            self.txt_egitim_log.insert(tk.END, f"=== {baslik.upper()} BAŞLATILDI ===\n\n")

            try:
                proc = subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", errors="replace")
                for line in proc.stdout:
                    self.txt_egitim_log.insert(tk.END, line)
                    self.txt_egitim_log.see(tk.END)
                proc.wait()

                if proc.returncode == 0:
                    self._alt_durum_mesaj(f"Başarılı: {baslik}")
                    messagebox.showinfo("Eğitim Tamamlandı", f"{baslik} başarıyla tamamlandı!\nModel 'modeller/parkinson_model.pkl' olarak kaydedildi.")
                else:
                    self._alt_durum_mesaj(f"Eğitim Hatası: {baslik}")
            except Exception as e:
                self.txt_egitim_log.insert(tk.END, f"\n[HATA]: {e}\n")
                self._alt_durum_mesaj(f"Hata oluştu: {e}")

            self.after(500, self._durum_gostergelerini_guncelle)

        threading.Thread(target=target, daemon=True).start()

    def _olustur_sekme_durum(self):
        ana_panel = tk.Frame(self.tab_durum, bg=self.C_BG)
        ana_panel.pack(fill="both", expand=True)

        # Durum Tablosu Kartı
        tablo_kart = tk.Frame(ana_panel, bg=self.C_CARD, padx=16, pady=14, highlightthickness=1, highlightbackground=self.C_CARD_LIGHT)
        tablo_kart.pack(fill="x", pady=(0, 12))

        lbl_t = tk.Label(tablo_kart, text="📂 Bileşenler ve Dosya Sağlığı", font=("Segoe UI", 11, "bold"), fg=self.C_CYAN, bg=self.C_CARD)
        lbl_t.pack(anchor="w", pady=(0, 10))

        self.durum_satirlari = {}
        bilesenler = [
            ("MediaPipe Pose Landmarker Task", POSE_LANDMARKER_PATH, "modeller/pose_landmarker.task", "MediaPipe 33 Landmark AI Modeli"),
            ("Eğitilmiş Parkinson ML Modeli", PARKINSON_MODEL_PATH, "modeller/parkinson_model.pkl", "Random Forest Sınıflandırıcı"),
            ("Normal Kontrol Veri Seti (CSV)", DATASET_NORMAL_PATH, "veriler/raw/dataset_normal_parkinson.csv", "Kontrol Grubu Verileri"),
            ("Parkinson Hasta Veri Seti (CSV)", DATASET_PARKINSON_PATH, "veriler/raw/dataset_parkinson_parkinson.csv", "Hasta Grubu Verileri"),
            ("Canlı Seans Veri Deposu", PROCESSED_DATA_DIR, "veriler/processed/nytas_parkinson_veri/", "Kaydedilen Seanslar & Master CSV"),
        ]

        for name, path, rel_str, desc in bilesenler:
            row = tk.Frame(tablo_kart, bg="#0f172a", padx=10, pady=6, highlightthickness=1, highlightbackground="#334155")
            row.pack(fill="x", pady=3)

            lbl_n = tk.Label(row, text=name, font=("Segoe UI", 9, "bold"), fg=self.C_TEXT, bg="#0f172a", width=28, anchor="w")
            lbl_n.pack(side="left")

            lbl_d = tk.Label(row, text=desc, font=("Segoe UI", 8), fg=self.C_MUTED, bg="#0f172a", width=28, anchor="w")
            lbl_d.pack(side="left")

            lbl_p = tk.Label(row, text=rel_str, font=("Segoe UI", 8), fg="#38bdf8", bg="#0f172a", width=34, anchor="w")
            lbl_p.pack(side="left")

            lbl_st = tk.Label(row, text="...", font=("Segoe UI", 9, "bold"), bg="#0f172a", width=14, anchor="e")
            lbl_st.pack(side="right")

            self.durum_satirlari[name] = (path, lbl_st)

        # Hızlı Klasör Açma Kartı
        klasor_kart = tk.Frame(ana_panel, bg=self.C_CARD, padx=16, pady=14, highlightthickness=1, highlightbackground=self.C_CARD_LIGHT)
        klasor_kart.pack(fill="x")

        lbl_k = tk.Label(klasor_kart, text="⚡ Hızlı Klasör & Doküman Erişimi", font=("Segoe UI", 11, "bold"), fg=self.C_CYAN, bg=self.C_CARD)
        lbl_k.pack(anchor="w", pady=(0, 10))

        btn_row = tk.Frame(klasor_kart, bg=self.C_CARD)
        btn_row.pack(fill="x")

        k_butonlar = [
            ("📁 Modeller Klasörü", MODELS_DIR),
            ("📁 Ham Veri Setleri", RAW_DATA_DIR),
            ("📁 Seans Kayıtları", PROCESSED_DATA_DIR),
            ("📁 Dokümanlar & Başvuru", DOCS_DIR),
        ]

        for txt, pth in k_butonlar:
            b = tk.Button(btn_row, text=txt, font=("Segoe UI", 9, "bold"), bg="#334155", fg="#ffffff",
                          activebackground="#475569", activeforeground="#ffffff", relief="flat", cursor="hand2", padx=12, pady=6,
                          command=lambda p=pth: klasor_ac(p))
            b.pack(side="left", padx=4)

        b_rapor = tk.Button(btn_row, text="🔄 Tüm Raporları Güncelle", font=("Segoe UI", 9, "bold"), bg="#0284c7", fg="#ffffff",
                            activebackground="#0369a1", activeforeground="#ffffff", relief="flat", cursor="hand2", padx=12, pady=6,
                            command=self._tum_raporlari_yenile)
        b_rapor.pack(side="left", padx=4)

    def _tum_raporlari_yenile(self):
        try:
            from kaynak_kodlar.rapor_olusturucu import tum_seanslari_raporla
            self._alt_durum_mesaj("Tüm seanslar için medikal klinik raporlar derleniyor...")
            raporlar = tum_seanslari_raporla(str(PROCESSED_DATA_DIR))
            messagebox.showinfo("Başarılı", f"Toplam {len(raporlar)} seans için klinik raporlar güncellendi ve hazırlandı!")
            self._alt_durum_mesaj(f"{len(raporlar)} seans raporu güncellendi.")
        except Exception as e:
            messagebox.showerror("Hata", f"Raporlar güncellenirken hata: {e}")

    def _olustur_sekme_bilgi(self):
        txt_frame = tk.Frame(self.tab_bilgi, bg=self.C_CARD, padx=12, pady=12, highlightthickness=1, highlightbackground=self.C_CARD_LIGHT)
        txt_frame.pack(fill="both", expand=True)

        sc = ttk.Scrollbar(txt_frame)
        sc.pack(side="right", fill="y")

        txt = tk.Text(txt_frame, font=("Segoe UI", 9), bg="#0f172a", fg="#f1f5f9",
                      yscrollcommand=sc.set, relief="flat", padx=14, pady=14, wrap="word")
        txt.pack(side="left", fill="both", expand=True)
        sc.config(command=txt.yview)

        bilgi = """
========================================================================================
NYTAS-PARKINSON™ | ÖLÇÜLEN 8 DİJİTAL BİYOBELİRTEÇ & KLİNİK LİTERATÜR ALTYAPISI
========================================================================================

1. KOL SALINIM GENLİĞİ (Arm Swing Amplitude):
   • Açıklama: Sağ ve sol kolun yürüyüş sırasındaki omuza göre bağıl tepe-çukur (Peak-to-Peak) dikey genliği (cm).
   • Klinik Önemi: Parkinson hastalığının en erken motor belirtilerinden biri kol salınım genliğinde belirgin azalmadır (hipokinezi).

2. KOL SALINIM ASİMETRİSİ (Arm Swing Asymmetry):
   • Açıklama: İki kol arasındaki salınım farkı yüzdesi (|Sol - Sağ| / Max * 100).
   • Klinik Önemi: Parkinson hastalığı sıklıkla asimetrik başlar. Bu biyobelirteç erken tanı için en hassas göstergedir.

3. ADIM UZUNLUĞU (Stride Length):
   • Açıklama: 3D Dünya koordinatlarında sol ve sağ ayak bilekleri arasındaki maksimum mesafe (cm).
   • Klinik Önemi: Parkinson hastalarında adımlar kısalır (küçük adımlarla yürüyüş).

4. KADANS VE ADIM DEĞİŞKENLİĞİ (Cadence & Gait Variability):
   • Açıklama: Dakikadaki adım sayısı (adım/dk) ve adımlar arası zaman aralığındaki varyasyon katsayısı.
   • Klinik Önemi: Düzensiz kadans ve ritim bozukluğu düşme riskinin habercisidir.

5. DONMA SKORU (Freezing of Gait - FOG):
   • Açıklama: Yürüyüşün aniden kesilmesi, ayakların yere yapışması hissi ve titreme frekansı (0.0: Yok, 0.5: Şüpheli, 1.0: Belirgin FOG).
   • Klinik Önemi: İleri evre Parkinson'da en ciddi motor blokajıdır.

6. GÖVDE ÖNE EĞİMİ (Trunk Flexion / Camptocormia):
   • Açıklama: Omuz merkezi ile kalça merkezi arasındaki omurga öne eğilme açısı (derece).
   • Klinik Önemi: Fleksiyon postürü (öne eğik duruş) karakteristik Parkinson bulgusudur.

7. YÜRÜYÜŞ HIZI (Gait Velocity):
   • Açıklama: Kalça merkezinin birim zamandaki ilerleme hızı (cm/sn).
   • Klinik Önemi: Bradikinezi (hareket yavaşlığı) değerlendirmesinde temel metriktir.

8. EL TREMOR FREKANSI (Hand Tremor Frequency via FFT):
   • Açıklama: Gövde sarsıntısından izole edilmiş el/bilek hareketine Hızlı Fourier Dönüşümü (FFT) uygulanarak hesaplanan baskın frekans (Hz).
   • Klinik Önemi: Parkinson istirahat tremoru tipik olarak 3 - 7 Hz bandında zirve yapar.
========================================================================================
"""
        txt.insert("1.0", bilgi)
        txt.config(state="disabled")

    def _durum_gostergelerini_guncelle(self):
        pose_ok = POSE_LANDMARKER_PATH.exists()
        ml_ok = PARKINSON_MODEL_PATH.exists()

        if hasattr(self, 'lbl_rozet_pose'):
            if pose_ok:
                self.lbl_rozet_pose.config(text="● Pose: HAZIR", bg="#065f46", fg="#a7f3d0")
            else:
                self.lbl_rozet_pose.config(text="● Pose: EKSİK", bg="#991b1b", fg="#fecaca")

        if hasattr(self, 'lbl_rozet_ml'):
            if ml_ok:
                self.lbl_rozet_ml.config(text="● ML: HAZIR", bg="#065f46", fg="#a7f3d0")
            else:
                self.lbl_rozet_ml.config(text="● ML: EĞİTİM GEREKLİ", bg="#92400e", fg="#fde68a")

        if hasattr(self, 'durum_satirlari'):
            for name, (path, lbl_st) in self.durum_satirlari.items():
                if path.exists():
                    lbl_st.config(text="✔ Mevcut", fg="#10b981")
                else:
                    lbl_st.config(text="✖ Eksik", fg="#ef4444")

    def _alt_durum_cubugu(self):
        self.statusbar = tk.Label(
            self,
            text="Sistem Hazır | NYTAS-PARKINSON v1.2",
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg="#1e293b",
            anchor="w",
            padx=14,
            pady=4
        )
        self.statusbar.pack(side="bottom", fill="x")

    def _alt_durum_mesaj(self, txt: str):
        self.statusbar.config(text=f"● {txt}")


if __name__ == "__main__":
    app = NYTASGui()
    app.mainloop()
