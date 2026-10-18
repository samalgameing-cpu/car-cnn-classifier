
import os
import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image

st.set_page_config(
    page_title="CarVision AI",
    page_icon="🚘",
    layout="wide"
)

# ---------- Colourful CSS ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101629 0%, #182848 55%, #243b55 100%);
    color: #f8fafc;
}
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}
.hero {
    padding: 30px;
    border-radius: 24px;
    text-align: center;
    background: linear-gradient(110deg, #6366f1, #8b5cf6, #ec4899);
    color: white;
    margin-bottom: 26px;
    box-shadow: 0 10px 35px #00000035;
}
.hero h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 5px;
}
.hero p {
    color: #f3e8ff;
    font-size: 17px;
}
.panel {
    background: #202b45;
    padding: 24px;
    border: 1px solid #3c4b6b;
    border-radius: 20px;
    margin-bottom: 20px;
}
div.stButton > button {
    background: linear-gradient(90deg, #6366f1, #d946ef);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: bold;
    width: 100%;
}
div.stButton > button:hover {
    border: 1px solid #ffffff;
    color: white;
}
[data-testid="stMetric"] {
    background: #202b45;
    border: 1px solid #3c4b6b;
    padding: 18px;
    border-radius: 16px;
}
h1, h2, h3, p, label {
    color: #f8fafc;
}
[data-testid="stFileUploader"] {
    background: #202b45;
    border: 1px dashed #818cf8;
    padding: 15px;
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🚘 CarVision AI</h1>
    <p>Smart car image classification powered by CNN</p>
    <p>Upload an image • Predict the car • Explore probabilities</p>
</div>
""", unsafe_allow_html=True)

MODEL_PATH = "car_cnn_model.keras"
FILE_ID = "1yLQ3ebS5iWpV7j5zzIHoYQaRslHqCixx"

EXPECTED_CLASSES = [
    "Audi", "Hyundai Creta", "Mahindra Scorpio",
    "Rolls Royce", "Swift", "Tata Safari", "Toyota Innova"
]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        result = gdown.download(
            id=FILE_ID, output=MODEL_PATH, quiet=False
        )
        if not result or not os.path.exists(MODEL_PATH):
            raise RuntimeError("Google Drive model download failed.")
    return tf.keras.models.load_model(MODEL_PATH)

def load_classes():
    with open("class_names.txt", "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    if classes != EXPECTED_CLASSES:
        raise RuntimeError(
            "Please check the 7 car names and their order in class_names.txt."
        )
    return classes

try:
    model = load_model()
    class_names = load_classes()
    if model.output_shape[-1] != len(class_names):
        raise RuntimeError("Model outputs and class names do not match.")
except Exception as error:
    st.error("Unable to load the model or class names.")
    st.code(str(error))
    st.stop()

st.markdown(
    '<div class="panel"><h3>📸 Upload Your Car Image</h3>'
    '<p>Choose a JPG, JPEG or PNG image to begin.</p></div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.image(image, caption="Your uploaded image",
                 use_container_width=True)

    with right:
        st.markdown("### 🔍 Ready to identify?")
        st.write("Our CNN model will analyse the image and estimate its car category.")

        if st.button("✨ Predict Car", type="primary"):
            with st.spinner("Analysing image..."):
                resized = image.resize((128, 128))
                array = np.array(resized, dtype=np.float32)
                array = np.expand_dims(array, axis=0)
                predictions = model.predict(array, verbose=0)[0]

            index = int(np.argmax(predictions))
            confidence = float(predictions[index]) * 100

            st.markdown("---")
            st.markdown("### 🏆 Prediction Result")
            st.success(f"🚗 Predicted Car: {class_names[index]}")
            st.metric("Prediction Confidence", f"{confidence:.2f}%")

            st.markdown("### 📊 All Class Probabilities")
            for name, score in zip(class_names, predictions):
                st.write(f"**{name}** — {score * 100:.2f}%")
                st.progress(float(score))

st.markdown("---")
st.caption("CarVision AI | CNN-based image classification project")
