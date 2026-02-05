import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(page_title="Face ID Demo", page_icon="🧠", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5", compile=False)

def load_labels():
    labels = []
    with open("labels.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)  # "0 Suhrob"
            labels.append(parts[1] if len(parts) > 1 else parts[0])
    return labels

def preprocess_tm(pil_img: Image.Image):
    # Teachable Machine standart preprocess
    img = ImageOps.fit(pil_img.convert("RGB"), (224, 224), Image.Resampling.LANCZOS)
    x = np.asarray(img).astype(np.float32)
    x = (x / 127.5) - 1.0
    x = np.expand_dims(x, axis=0)
    return x

st.title("🧠 Face ID Demo (Teachable Machine)")
st.caption("Rasm yuklang → model kimligini taxmin qiladi. (Deploy uchun qulay demo)")

threshold = st.slider("Threshold (UNKNOWN)", 0.0, 1.0, 0.80, 0.01)

uploaded = st.file_uploader("Rasm yuklang (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Boshlash uchun rasm yuklang.")
    st.stop()

img = Image.open(uploaded)
st.image(img, caption="Yuklangan rasm", use_container_width=True)

model = load_model()
labels = load_labels()

x = preprocess_tm(img)
pred = model.predict(x, verbose=0)[0]
idx = int(np.argmax(pred))
conf = float(pred[idx])
name = labels[idx] if idx < len(labels) else str(idx)

final_name = name if conf >= threshold else "UNKNOWN"

st.subheader("Natija")
st.write(f"**Label:** {final_name}")
st.write(f"**Confidence:** {conf:.2f}")

st.divider()
st.caption("Debug: barcha classlar")
for i, lab in enumerate(labels):
    st.write(f"- {lab}: {pred[i]:.2f}")
