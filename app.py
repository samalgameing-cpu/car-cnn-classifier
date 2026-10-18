
import os
import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image

st.set_page_config(page_title="Car CNN Classifier", page_icon="🚗")
st.title("🚗 Car Image Classification")

MODEL_PATH = "car_cnn_model.keras"
FILE_ID = "1yLQ3ebS5iWpV7j5zzIHoYQaRslHqCixx"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    downloaded = gdown.download(
        id=FILE_ID,
        output=MODEL_PATH,
        quiet=False
    )

    if not downloaded or not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model download failed. Check Google Drive sharing.")

    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open("class_names.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

try:
    model = load_model()
    class_names = load_classes()

    if model.output_shape[-1] != len(class_names):
        st.error(
            f"Model has {model.output_shape[-1]} outputs, "
            f"but class_names.txt has {len(class_names)} classes."
        )
        st.stop()

    st.success(f"Model loaded: {len(class_names)} car categories")

except Exception as e:
    st.error("Model load nahi ho paya.")
    st.code(str(e))
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a car image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict Car", type="primary"):
        image = image.resize((128, 128))
        array = np.array(image, dtype=np.float32)
        array = np.expand_dims(array, axis=0)

        with st.spinner("Predicting..."):
            predictions = model.predict(array, verbose=0)[0]

        index = int(np.argmax(predictions))
        st.success(f"Predicted car: {class_names[index]}")
        st.metric("Confidence", f"{predictions[index] * 100:.2f}%")

        st.subheader("All class probabilities")
        for name, score in zip(class_names, predictions):
            st.write(f"{name}: {score * 100:.2f}%")
            st.progress(float(score))
