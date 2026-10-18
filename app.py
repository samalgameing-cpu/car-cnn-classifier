
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Car CNN Classifier", page_icon="🚗")
st.title("🚗 Car Image Classification")
st.write("Upload an image to classify it using your CNN model.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("car_cnn_model.keras")

try:
    model = load_model()
    with open("class_names.txt", "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

uploaded = st.file_uploader(
    "Upload a car image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        image = image.resize((128, 128))
        arr = np.array(image, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)

        predictions = model.predict(arr, verbose=0)[0]
        index = int(np.argmax(predictions))

        st.success(f"Predicted class: {class_names[index]}")
        st.write(f"Confidence: {predictions[index] * 100:.2f}%")
