
import os
import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image

st.set_page_config(
    page_title="Car CNN Classifier",
    page_icon="🚗"
)

st.title("🚗 Car Image Classification")
st.write("Upload a car image and let the CNN model predict its class.")

MODEL_PATH = "car_cnn_model.keras"
FILE_ID = "1yLQ3ebS5iWpV7j5zzIHoYQaRslHqCixx"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        result = gdown.download(
            url,
            MODEL_PATH,
            quiet=False,
            fuzzy=True
        )

        if not result or not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Model download failed. Check Google Drive sharing."
            )

        if os.path.getsize(MODEL_PATH) < 1000000:
            raise RuntimeError(
                "Downloaded model file looks incomplete."
            )

    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open("class_names.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

try:
    with st.spinner("Loading CNN model... Please wait."):
        model = load_model()
        class_names = load_classes()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.info(
        "Check Google Drive sharing and ensure class_names.txt "
        "is present in the GitHub repository."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a car image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(
        image,
        caption="Uploaded car image",
        use_container_width=True
    )

    if st.button("Predict Car"):
        resized_image = image.resize((128, 128))
        image_array = np.array(
            resized_image,
            dtype=np.float32
        )
        image_array = np.expand_dims(image_array, axis=0)

        with st.spinner("Predicting..."):
            predictions = model.predict(image_array, verbose=0)[0]

        if len(predictions) != len(class_names):
            st.error(
                "Model classes do not match class_names.txt."
            )
        else:
            best_index = int(np.argmax(predictions))
            confidence = float(predictions[best_index]) * 100

            st.success(
                f"Predicted class: {class_names[best_index]}"
            )
            st.metric("Confidence", f"{confidence:.2f}%")

            st.subheader("Class Probabilities")
            for name, probability in zip(class_names, predictions):
                st.write(f"{name}: {probability * 100:.2f}%")
                st.progress(float(probability))
