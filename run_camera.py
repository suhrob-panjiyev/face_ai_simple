import time
import sqlite3
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


#SETTINGS 
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

CAM_INDEX = 0
USE_DSHOW = True              # Windows uchun
THRESHOLD = 0.70              # 0.70 dan past bo'lsa UNKNOWN
COOLDOWN_SEC = 60             # 1 minutda 1 marta yozadi

DB_PATH = "attendance.sqlite"



def load_labels(path: str):
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)  # "0 Suhrob" -> ["0","Suhrob"]
            labels.append(parts[1] if len(parts) > 1 else parts[0])
    return labels


def init_db(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            confidence REAL NOT NULL,
            ts INTEGER NOT NULL
        )
    """)
    conn.commit()
    return conn


def log_attendance(conn, name: str, confidence: float):
    ts = int(time.time())
    conn.execute("INSERT INTO attendance(name, confidence, ts) VALUES (?, ?, ?)",
                 (name, float(confidence), ts))
    conn.commit()


def preprocess_tm(frame_bgr: np.ndarray) -> np.ndarray:
    # Teachable Machine: 224x224 RGB, (x/127.5)-1
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)

    x = np.asarray(img).astype(np.float32)
    x = (x / 127.5) - 1.0
    x = np.expand_dims(x, axis=0)
    return x


def main():
    labels = load_labels(LABELS_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Face detector (Haar) — fon ta'sirini kamaytiradi
    haar_path = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(haar_path)

    conn = init_db(DB_PATH)
    last_logged = {}  # name -> last_time

    backend = cv2.CAP_DSHOW if USE_DSHOW else 0
    cap = cv2.VideoCapture(CAM_INDEX, backend)

    if not cap.isOpened():
        raise RuntimeError("Kamera ochilmadi. CAM_INDEX=0/1/2 sinab ko'ring yoki USE_DSHOW=True qiling.")

    print("✅ Kamera ishga tushdi. 'q' bosib chiqasiz.")
    print(f"✅ DB: {DB_PATH}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame olinmadi.")
            break

        # 1) Yuzni topamiz
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        # 2) Eng katta yuzni olamiz (agar topilsa)
        face_img = frame
        box = None
        if len(faces) > 0:
            x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            box = (x, y, w, h)
            face_img = frame[y:y+h, x:x+w]

        # 3) Modelga beramiz
        data = preprocess_tm(face_img)
        pred = model.predict(data, verbose=0)[0]
        idx = int(np.argmax(pred))
        conf = float(pred[idx])

        name = labels[idx] if idx < len(labels) else str(idx)

        # 4) UNKNOWN filtri
        display_name = name
        if conf < THRESHOLD:
            display_name = "UNKNOWN"

        # 5) Davomatga yozish (cooldown bilan)
        status = ""
        now = time.time()
        if display_name != "UNKNOWN":
            last_t = last_logged.get(display_name, 0)
            if now - last_t >= COOLDOWN_SEC:
                log_attendance(conn, display_name, conf)
                last_logged[display_name] = now
                status = "LOGGED"
                print(f"✅ LOGGED: {display_name} conf={conf:.2f}")
            else:
                status = "SEEN"

        # 6) Ekranga chiqarish
        if box:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        text = f"{display_name} conf={conf:.2f} {status}"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Attendance AI (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()


if __name__ == "__main__":
    main()
