# RealTime-Face-Sentiment-Analysis 🚀

Bu proje, **DeepFace** ve **OpenCV** kütüphanelerini kullanarak gerçek zamanlı çoklu yüz algılama, duygu analizi, yaş ve cinsiyet tahmini gerçekleştiren yapay zeka tabanlı bir asistan sistemidir.

## ✨ Öne Çıkan Özellikler

* **👥 Çoklu Yüz Algılama (Multi-Face):** Aynı karedeki birden fazla kişiyi eş zamanlı olarak analiz eder.
* **📊 Gelişmiş Duygu Analizi:** 7 farklı duygu durumunu (Mutlu, Üzgün, Kızgın, Şaşkın vb.) gerçek zamanlı takip eder.
* **⚖️ Stabilizasyon Algoritması:** Tahminlerdeki anlık dalgalanmaları önlemek için "Majority Voting" ve "Moving Average" (Hareketli Ortalama) teknikleri kullanılmıştır.
* **🎵 Akıllı Aksiyon Tetikleyici (Spotify):** Kullanıcının "Üzgün" olduğu saptandığında otomatik olarak neşeli bir Spotify çalma listesi başlatır.
* **🛠️ Mühendislik Kalibrasyonları:** Düşük ışık koşulları için Gaussian Blur önişlemesi ve özel cinsiyet hassasiyeti (Gender Threshold) ayarları entegre edilmiştir.

## 🛠️ Kullanılan Teknolojiler

* **Dil:** Python
* **Kütüphaneler:** OpenCV, DeepFace, TensorFlow, NumPy
* **Model:** VGG-Face (Kişi tanıma için)

## 🚀 Kurulum ve Çalıştırma

1.  **Depoyu klonlayın:**
    ```bash
    git clone [https://github.com/RukiyeGizemGokmen/RealTime-Face-Sentiment-Analysis.git](https://github.com/RukiyeGizemGokmen/RealTime-Face-Sentiment-Analysis.git)
    cd RealTime-Face-Sentiment-Analysis
    ```

2.  **Sanal ortam oluşturun ve aktif edin:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows için: venv\Scripts\activate
    ```

3.  **Gerekli kütüphaneleri kurun:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Uygulamayı başlatın:**
    ```bash
    python duygu_analizi.py
    ```

## 📂 Klasör Yapısı

* `duygu_analizi.py`: Ana uygulama kodu.
* `GIZEM/`: Kişi tanıma için referans fotoğrafların bulunduğu klasör (Yerel kullanım içindir).
* `requirements.txt`: Proje bağımlılıkları.
* `analiz_raporu.txt`: Oturum sırasında tutulan analiz logları.

---
**Geliştiren:** [Rukiye Gizem Gökmen](https://github.com/RukiyeGizemGokmen)  
*Bilgisayar Mühendisliği 3. Sınıf Öğrencisi*
