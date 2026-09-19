# 🚀 NYTAS-PARKINSON — Gelecek Yol Haritası & Prototip Uygulama Planı

> **TÜBİTAK 1002 Hızlı Destek Programı & Klinik Ürünleştirme Vizyonu**  
> **Hazırlanma Tarihi:** Eylül 2026  
> **Mevcut Sürüm:** v1.3 (3D Sagittal Gövde Eğimi, Z-Ekseni Hız, Klinik Eşikler, Hibrit Karar Motoru)

---

## 🎯 Vizyon ve Temel Amaç

NYTAS-PARKINSON projesinin temel amacı; karmaşık, pahalı ve giyilebilir sensörlere ihtiyaç duymadan, yalnızca standart bir RGB kamera ve yapay zeka ile **Parkinson Hastalığı'nın motor semptomlarını erken evrede tespit etmek**, seyrini takip etmek ve hekimlere objektif biyomekanik raporlar sunmaktır.

Bu doküman, sistemin laboratuvar aşamasından **klinik olarak doğrulanmış, hekim dostu bir prototip ürüne** dönüşmesi için izlenecek 4 aşamalı yol haritasını içerir.

---

## 🗺️ 4 Aşamalı Gelişim Yol Haritası

```mermaid
graph TD
    subgraph Faz 1 [Faz 1: Klinik Veri & Validasyon (1-2 Ay)]
        F1_1[Klinik Etik & Protokol Onayı]
        F1_2[Gerçek Hasta & Kontrol Video Kayıtları]
        F1_3[MDS-UPDRS Skoru ile Eşleştirme]
        F1_4[Modelin Gerçek Veriyle Eğitilmesi]
    end

    subgraph Faz 2 [Faz 2: Prototip Arayüz & Hekim Raporlama (2-3 Ay)]
        F2_1[Modern Masaüstü/Web Arayüzü]
        F2_2[Otomatik PDF/HTML Klinik Rapor]
        F2_3[ON/OFF İlaç Etkinliği Karşılaştırması]
        F2_4[Çoklu Hasta Veritabanı SQLite]
    end

    subgraph Faz 3 [Faz 3: İleri Biyobelirteçler & Fonksiyonel Testler (2-3 Ay)]
        F3_1[180° Dönüş Analizi - En Bloc Turn]
        F3_2[Çift Görev - Dual Task Yürüyüşü]
        F3_3[Otur-Kalk Testi - STS 5x Analizi]
        F3_4[Kamera Açı & Mesafe Otomatik Kalibrasyonu]
    end

    subgraph Faz 4 [Faz 4: Mobil & Uzaktan İzleme - Tele-Tıp (Gelecek Vizyonu)]
        F4_1[Mobil Uygulama - Android/iOS MediaPipe]
        F4_2[Evde Haftalık Koridor Yürüyüş Testi]
        F4_3[Bulut Hekim Takip Paneli]
        F4_4[KVKK / HIPAA Uyumlu Güvenli Altyapı]
    end

    Faz 1 --> Faz 2
    Faz 2 --> Faz 3
    Faz 3 --> Faz 4
```

---

## 📌 Faz Detayları ve Eylem Planı

### 🔹 FAZ 1: Klinik Veri Doğrulama ve Etiketleme (1 - 2 Ay)
Mevcut modelimiz sentetik olarak üretilmiş ve literatür normlarına kalibre edilmiş verilerle eğitilmiştir. Projenin akademik ve klinik geçerliliği için gerçek hastalarla test edilmesi şarttır.

1. **Klinik Protokol ve Video Arşivi:**
   * Nöroloji kliniği ile ortak çalışma: Hoehn & Yahr Evre 1, 2 ve 3 düzeyinde Parkinson hastaları ve yaş uyumlu sağlıklı kontrol grubu.
   * Standart 5-10 metrelik yürüyüş parkuru belirlenmesi.
2. **MDS-UPDRS-III Skor Korelasyonu:**
   * Modelin sadece ikili ("Normal" / "Risk") sınıflandırma yapması yerine, uzman nörolog tarafından verilen MDS-UPDRS-III alt parametreleriyle (Örn: Parametre 3.10 Yürüyüş, 3.11 Donma, 3.14 Postüral Stabilite) korelasyon skorları üretilmesi.
3. **Gerçek Veriyle Transfer Learning / Fine-Tuning:**
   * Toplanan gerçek hasta seanslarıyla modelin yeniden eğitilerek gerçek klinik hassasiyetinin doğrulanması.

---

### 🔹 FAZ 2: Prototip Uygulama & Otomatik Hekim Raporlama (2 - 3 Ay)
Kullanıcı deneyimini artıran, klinik ortamda hekimlerin ve fizyoterapistlerin 30 saniyede sonuç alabileceği bir prototip arayüz.

1. **Otomatik PDF / HTML Klinik Rapor Oluşturucu:**
   * Seans bittiğinde tek tıkla üretilen 2 sayfalık renkli hekim raporu.
   * **İçerik:**
     * Hasta demografik bilgileri (Yaş, Boy, Cinsiyet, İlaç Saati)
     * 8 Biyobelirtecin Klinik Referans Aralıklarıyla Karşılaştırma Grafiği (Radar / Spider Chart)
     * Risk puanlaması ve hibrit karar motoru özeti
     * Yürüyüş hız ve kol salınım dalga formları (Zaman serisi grafikleri)
2. **ON / OFF İlaç Dönemi Takibi:**
   * Parkinson hastaları ilaç (Levodopa) öncesinde ("OFF" dönemi) yavaş ve kısıtlı, ilaç sonrasında ("ON" dönemi) ise daha rahat yürür.
   * Sisteme "Seans Tipi: İlaç Öncesi / İlaç Sonrası" seçeneği eklenerek ilacın motor fonksiyonlara etkisi kıyaslanacak.
3. **Modern Arayüz & Yerel Veritabanı:**
   * Tkinter arayüzünün modern PyQt6 veya web tabanlı (FastAPI + React / WebView) teknolojiye taşınması.
   * Hastaların geçmiş seanslarını listeleyen, zamansal iyileşme/kötüleşme eğrisini çizen SQLite yerel veri tabanı.

---

### 🔹 FAZ 3: İleri Biyobelirteçler ve Fonksiyonel Klinik Testler (2 - 3 Ay)
Literatürde Parkinson'a özgü en kritik motor bozuklukları sisteme entegre etmek.

1. **180° Dönüş Analizi (Turning Analysis):**
   * Parkinson hastaları düz yürümekten ziyade dönüş yaparken en çok zorlanırlar (adım sayısı artar, tek parça halinde "en bloc" dönerler).
   * Hastanın yürüyüş sonunda geri dönerken attığı adım sayısı ve dönüş süresi otomatik hesaplanacak.
2. **Çift Görev Yürüyüşü (Dual-Task Walking):**
   * Hastaya yürürken zihinsel bir görev verilerek (örn. 100'den geriye 7'şer sayma) bilişsel yük altında yürümesi istenir. Parkinson hastalarında çift görevde kol salınımı ve hız dramatik olarak düşer.
3. **5 Kez Otur-Kalk Testi (Five Times Sit-to-Stand - 5XSTS):**
   * Yürüyüş moduna ek olarak, sandalyeden kalkıp oturma süresi ve gövde ivmelenmesinin MediaPipe ile ölçülmesi.

---

### 🔹 FAZ 4: Mobil ve Evde Uzaktan İzleme (Tele-Sağlık) (Gelecek Vizyonu)
Hastaneye gidemeyen ileri evre hastalar için ev ortamında takip.

1. **Mobil Uygulama (Flutter / Android - MediaPipe Mobile SDK):**
   * Hasta veya yakınının telefon kamerasını bir yere sabitleyerek koridorda yürümesi.
2. **Bulut Hekim Portalı:**
   * Hastanın haftalık test sonuçlarının doğrudan takip eden nöroloğun ekranına düşmesi.
3. **KVKK & HIPAA Uyumluluğu:**
   * Tüm video verilerinin cihazda işlenmesi, sunucuya sadece sayısal biyobelirteçlerin şifreli aktarılması.

---

## 🛠️ İlk Uygulanacak Adım (Öneri: Hekim Raporlama Modülü)

Bu yol haritasında en hızlı somut çıktı verecek ve projeyi doğrudan bir üst seviyeye taşıyacak çalışma **Faz 2 kapsamındaki Otomatik PDF/HTML Raporlama Modülü**dür.

```
+-------------------------------------------------------------+
| NYTAS-PARKINSON KLINIK ANALIZ RAPORU                        |
| Hasta ID: HASTA_001 | Boy: 175 cm | Tarih: 2026-09-19      |
+-------------------------------------------------------------+
| TAHMIN: PARKINSON RISKI (Guven: %87) | Klinik Risk: 4/8     |
+-------------------------------------------------------------+
| BIYOBELIRTEC       | DEGER       | NORMAL ARALIK | DURUM    |
|--------------------+-------------+---------------+----------|
| Kol Asimetrisi     | % 21.4      | < %12.0       | [RISKLI] |
| Yuruyus Hizi       | 74.2 cm/s   | > 90.0 cm/s   | [RISKLI] |
| Govde One Egimi    | 11.8 derece | < 7.0 derece  | [RISKLI] |
| Adim Uzunlugu      | 42.1 cm     | > 50.0 cm     | [RISKLI] |
| Kadans             | 104 adim/dk | 90 - 135      | [NORMAL] |
| Tremor (3.5-7Hz)   | Yok         | Yok           | [NORMAL] |
| Freezing (FOG)     | Yok         | Yok           | [NORMAL] |
| Kol Genlik Ort.    | 18.5 cm     | > 14.0 cm     | [NORMAL] |
+-------------------------------------------------------------+
```

Bu modül eklendiğinde canlı analiz bittiğinde `veriler/processed/` klasörüne otomatik olarak profesyonel bir hekim raporu üretilecektir.
