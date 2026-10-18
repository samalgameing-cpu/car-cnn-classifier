
import os
import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image

st.set_page_config(
    page_title="Car CNN Classifier",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Image Classification")
st.write("Upload a car image to predict its category.")

MODEL_PATH = "car_cnn_model.keras"
FILE_ID = "1yLQ3ebS5iWpV7j5zzIHoYQaRslHqCixx"

EXPECTED_CLASSES = [
    "Audi",
    "Hyundai Creta",
    "Mahindra Scorpio",
    "Rolls Royce",
    "Swift",
    "Tata Safari",
    "Toyota Innova"
]


@st.cache_resource
def load_model():
    # Download model only if it is not already present.
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
            or os.path.getsize(MODEL_PATH) < 1_000_000
        ):
            raise RuntimeError(
                "Model download failed. Check Google Drive sharing "
                "and confirm that the file ID is correct."
            )

    return tf.keras.models.load_model(MODEL_PATH)


def load_classes():
    # Read the class names directly from the current repository file.
    with open("class_names.txt", "r", encoding="utf-8") as file:
        classes = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if len(classes) != 7:
        raise RuntimeError(
            f"class_names.txt contains {len(classes)} classes, not 7. "
            "Update class_names.txt in GitHub."
        )

    if classes != EXPECTED_CLASSES:
        raise RuntimeError(
            "The class names in class_names.txt do not match the "
            "expected order. Please check the file in GitHub."
        )

    return classes


try:
    model = load_model()
    class_names = load_classes()

    if model.output_shape[-1] != len(class_names):
        raise RuntimeError(
            f"Model has {model.output_shape[-1]} outputs, but "
            f"class_names.txt has {len(class_names)} classes."
        )

except Exception as error:
    st.error("Unable to load the correct model or class names.")
    st.code(str(error))
    st.info(
        "Check the Google Drive model file and the class_names.txt "
        "file in your GitHub repository."
    )
    st.stop()


st.success("CNN model loaded successfully!")
st.write("Available car categories:", ", ".join(class_names))

uploaded_file = st.file_uploader(
    "📷 Upload a car image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded car image",
            use_container_width=True
        )

        if st.button("🔍 Predict Car", type="primary"):
            with st.spinner("Predicting car category..."):
                resized_image = image.resize((128, 128))

                image_array = np.array(
                    resized_image,
                    dtype=np.float32
                )

                image_array = np.expand_dims(image_array, axis=0)

                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

            if len(predictions) != len(class_names):
                st.error("Model output and class names do not match.")
            else:
                best_index = int(np.argmax(predictions))
                best_class = class_names[best_index]
                confidence = float(predictions[best_index]) * 100

                st.subheader("Prediction Result")
                st.success(f"Predicted car: {best_class}")
                st.metric(
                    "Prediction confidence",
                    f"{confidence:.2f}%"
                )

                st.subheader("Class Probabilities")

                for name, probability in zip(
                    class_names,
                    predictions
                ):
                    st.write(f"{name}: {probability * 100:.2f}%")
                    st.progress(float(probability))

    except Exception as error:
        st.error("Image prediction failed.")
        st.code(str(error))
