import cv2
from deepface import DeepFace
import os
import numpy as np
from collections import Counter
from datetime import datetime
import time
import webbrowser 

# Gereksiz uyarıları ve TensorFlow loglarını kapat
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 1. VERİ TABANI VE LOG HAZIRLIĞI ---
db_path = os.path.join(os.getcwd(), "GIZEM")
log_file = "analiz_raporu.txt"

cap = cv2.VideoCapture(0)

duygular_tr = {
    "angry": "Kizgin", "disgust": "Igrenmis", "fear": "Korkmus",
    "happy": "Mutlu", "sad": "Uzgun", "surprise": "Saskin", "neutral": "Normal"
}

# --- STABİLİZASYON VE ANALİZ AYARLARI ---
frame_count = 0
analysis_frequency = 5 
last_results = []
histories = {"age": [], "emotion": [], "gender": [], "name": []}

# Mühendislik Kalibrasyonları
yas_dusurme_miktari = 7 
# KRİTİK AYAR: Bu değeri %20'ye çektim. 
# Model %20 bile "Woman" dese sistem onu 'Kadin' olarak etiketleyecek.
KADIN_HASSASIYETI = 20.0 
UZGUN_SURE_ESIGI = 12 

# --- SPOTIFY KONTROL ---
spotify_acildi = False 
SPOTIFY_URL = "https://open.spotify.com/playlist/37i9dQZF1DXdPec7WLTqzq" # Happy Hits

prev_frame_time = 0

print("--- Cinsiyet Hassasiyeti Maksimuma Çıkarıldı ---")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        frame = cv2.flip(frame, 1)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time

        # --- ANALİZ AŞAMASI ---
        if frame_count % analysis_frequency == 0:
            try:
                # Görüntü yumuşatma (Yüzdeki sert gölgeleri azaltarak cinsiyet doğruluğunu artırır)
                processed_frame = cv2.GaussianBlur(frame, (5, 5), 0)
                
                analysis = DeepFace.analyze(processed_frame, actions=['emotion', 'age', 'gender'], 
                                           enforce_detection=False, detector_backend='opencv')
                
                for face in analysis:
                    region = face['region']
                    face_img = processed_frame[region['y']:region['y']+region['h'], 
                                               region['x']:region['x']+region['w']]
                    kisi_ismi = "Bilinmiyor"
                    if os.path.exists(db_path) and face_img.size > 0:
                        try:
                            recognition = DeepFace.find(img_path=face_img, db_path=db_path, 
                                                        model_name='VGG-Face', enforce_detection=False, silent=True)
                            if recognition and not recognition[0].empty:
                                identity_path = recognition[0]['identity'][0]
                                kisi_ismi = os.path.basename(os.path.dirname(identity_path))
                        except: pass
                    face['identity_name'] = kisi_ismi
                last_results = analysis
            except: pass

        # --- GÖRSELLEŞTİRME VE FİLTRELEME ---
        if last_results:
            for res in last_results:
                region = res['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                if w < 100: continue 

                # 1. Yaş
                raw_age = int(res.get('age', 0)) - yas_dusurme_miktari
                histories["age"].append(raw_age)
                if len(histories["age"]) > 20: histories["age"].pop(0)
                stable_age = max(0, int(np.mean(histories["age"])))

                # 2. Duygu
                emo_en = res.get('dominant_emotion', 'neutral')
                histories["emotion"].append(emo_en)
                if len(histories["emotion"]) > 20: histories["emotion"].pop(0)
                stable_emo_en = Counter(histories["emotion"]).most_common(1)[0][0]
                stable_emo_tr = duygular_tr.get(stable_emo_en, stable_emo_en)

                # 3. Cinsiyet (Geliştirilmiş Oylama Sistemi)
                g_scores = res.get('gender', {})
                woman_prob = g_scores.get('Woman', 0)
                
                # Kadın eşiğini çok düşürdük, artık daha kolay 'Kadin' diyecek
                current_gender = "Kadin" if woman_prob > KADIN_HASSASIYETI else "Erkek"
                histories["gender"].append(current_gender)
                if len(histories["gender"]) > 25: histories["gender"].pop(0) # Belleği artırdık
                stable_gender = Counter(histories["gender"]).most_common(1)[0][0]

                # 4. Kimlik
                name = res.get('identity_name', "Bilinmiyor")

                # --- UI TASARIMI ---
                color = (0, 0, 255) if stable_emo_en == 'sad' else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y-35), (x+w, y), color, -1)
                bilgi = f"{name} | {stable_gender} | {stable_emo_tr} | {stable_age}"
                cv2.putText(frame, bilgi, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

                # --- SPOTIFY TETİKLEME ---
                if stable_emo_en == 'sad':
                    cv2.putText(frame, "Modun mu dusuk? Muzik aciliyor...", (20, 450), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    if histories["emotion"].count('sad') >= UZGUN_SURE_ESIGI and not spotify_acildi:
                        webbrowser.open(SPOTIFY_URL)
                        spotify_acildi = True 

                if stable_emo_en in ['happy', 'neutral']:
                    spotify_acildi = False

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow('Gizem AI - Gender Fixed Interface', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()