
import os
import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image

# Page settings
st.set_page_config(
    page_title="Car CNN Classifier",
    page_icon="🚗",
    layout="centered"
)

# App heading
st.title("🚗 Car Image Classification")
st.write(
    "Upload a car image and let our trained CNN model "
    "predict its class."
)

MODEL_PATH = "car_cnn_model.keras"
FILE_ID = "1yLQ3ebS5iWpV7j5zzIHoYQaRslHqCixx"


# Download and load the trained model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading CNN model... Please wait."):
            result = gdown.download(
                id=FILE_ID,
                output=MODEL_PATH,
                quiet=False
            )

        if (
            not result
            or not os.path.exists(MODEL_PATH)
            or os.path.getsize(MODEL_PATH) < 1000000
        ):
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)

            raise RuntimeError(
                "Model download failed. Check that the Google "
                "Drive file is shared as Anyone with the link."
            )

    return tf.keras.models.load_model(MODEL_PATH)


# Read class names
@st.cache_data
def load_classes():
    with open("class_names.txt", "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


# Initialize model and class names
try:
    model = load_model()
    class_names = load_classes()

except Exception as error:
    st.error("Unable to load the CNN model.")
    st.code(str(error))
    st.info(
        "Check Google Drive sharing, class_names.txt, "
        "and the Streamlit app logs."
    )
    st.stop()


# Upload an image
uploaded_file = st.file_uploader(
    "📷 Choose a car image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Your uploaded image",
            use_container_width=True
        )

        if st.button("🔍 Predict Car", type="primary"):
            with st.spinner("Predicting car class..."):
                resized_image = image.resize((128, 128))

                image_array = np.array(
                    resized_image,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

            if len(predictions) != len(class_names):
                st.error(
                    "Model output classes do not match "
                    "the names in class_names.txt."
                )
            else:
                best_index = int(np.argmax(predictions))
                confidence = float(
                    predictions[best_index] * 100
                )

                st.success(
                    f"Predicted class: {class_names[best_index]}"
                )

                st.metric(
                    "Prediction confidence",
                    f"{confidence:.2f}%"
                )

                st.subheader("📊 Class Probabilities")

                for name, probability in zip(
                    class_names,
                    predictions
                ):
                    st.write(
                        f"{name}: {probability * 100:.2f}%"
                    )
                    st.progress(float(probability))

    except Exception as error:
        st.error("Image prediction failed.")
        st.code(str(error))
