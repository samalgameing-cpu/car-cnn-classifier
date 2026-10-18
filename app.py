
import os
import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image

st.set_page_config(
    page_title="CarVision AI | Smart Classifier",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = "car_cnn_model.keras"
FILE_ID = "1yLQ3ebS5iWpV7j5zzIHoYQaRslHqCixx"

CLASSES = [
    "Audi",
    "Hyundai Creta",
    "Mahindra Scorpio",
    "Rolls Royce",
    "Swift",
    "Tata Safari",
    "Toyota Innova"
]

# ---------------- PREMIUM FULL-PAGE DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #05091d;
    --panel: #0c1735;
    --purple: #a855f7;
    --pink: #ec4899;
    --cyan: #22d3ee;
    --green: #34d399;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #442079 0%, transparent 34%),
        radial-gradient(ellipse at 90% 15%, #063d69 0%, transparent 32%),
        linear-gradient(145deg, #05091d, #0a1030 55%, #120829);
    color: #f8fafc;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    max-width: 100%;
    padding: 1rem 2rem 1.5rem;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.8rem;
}

.brand {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    padding: 10px 4px 22px;
    border-bottom: 1px solid #29375c;
    margin-bottom: 18px;
}

.logo {
    font-size: clamp(24px, 3vw, 35px);
    font-weight: 900;
    letter-spacing: -1.5px;
    color: #fff;
}

.logo span {
    background: linear-gradient(90deg, #22d3ee, #a855f7, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.pill {
    display: inline-block;
    padding: 9px 15px;
    border: 1px solid #6753d9;
    border-radius: 40px;
    color: #d9d6ff;
    background: #15163b;
    font-size: 12px;
    font-weight: 700;
    margin-left: 5px;
}

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 25px;
    padding: clamp(24px, 4vw, 55px);
    margin-bottom: 18px;
    background:
        radial-gradient(ellipse at 80% 20%, #5625a8 0%, transparent 35%),
        radial-gradient(ellipse at 90% 100%, #075985 0%, transparent 38%),
        linear-gradient(110deg, #11153d, #181044 55%, #071d40);
    border: 1px solid #4935a3;
    box-shadow: 0 12px 45px #0005, inset 0 0 40px #7c3aed12;
}

.hero:after {
    content: "🚘";
    position: absolute;
    right: 5%;
    top: 2%;
    font-size: clamp(100px, 19vw, 250px);
    opacity: .15;
    filter: drop-shadow(0 0 35px #22d3ee);
    pointer-events: none;
}

.eyebrow {
    color: #67e8f9;
    font-weight: 800;
    font-size: 12px;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero h1 {
    font-size: clamp(32px, 5vw, 66px);
    line-height: 1.08;
    letter-spacing: -2.5px;
    margin: 12px 0;
    font-weight: 900;
    color: white;
}

.gradient-text {
    background: linear-gradient(90deg, #22d3ee, #818cf8, #f0abfc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    max-width: 670px;
    color: #c6d2f3;
    font-size: clamp(14px, 1.5vw, 18px);
    line-height: 1.7;
}

.feature {
    display: inline-block;
    margin: 12px 8px 0 0;
    padding: 10px 14px;
    border-radius: 12px;
    background: #ffffff0c;
    border: 1px solid #ffffff22;
    color: #e0e7ff;
    font-size: 12px;
    font-weight: 700;
}

.section-title {
    font-size: clamp(21px, 2.5vw, 30px);
    font-weight: 900;
    color: white;
    margin: 5px 0;
}

.section-sub {
    color: #9baed5;
    font-size: 13px;
    margin-bottom: 14px;
}

.panel {
    height: 100%;
    min-height: 300px;
    padding: clamp(18px, 2.2vw, 30px);
    border-radius: 22px;
    background: linear-gradient(145deg, #101d40, #0b1330);
    border: 1px solid #344a79;
    box-shadow: 0 10px 30px #0003, inset 0 0 24px #6366f10b;
}

.upload-panel {
    border-color: #8b5cf6;
    background:
        radial-gradient(ellipse at 0% 0%, #6622a344, transparent 55%),
        linear-gradient(145deg, #121a42, #0a1534);
}

.result-panel {
    border-color: #0e7490;
    background:
        radial-gradient(ellipse at 100% 0%, #0891b222, transparent 55%),
        linear-gradient(145deg, #0c1d3b, #08152f);
}

.stFileUploader {
    padding: 12px;
    border: 2px dashed #818cf8;
    border-radius: 17px;
    background: #6366f10c;
}

.stFileUploader label, .stFileUploader small {
    color: #e0e7ff !important;
}

div.stButton > button {
    width: 100%;
    min-height: 52px;
    border: 0;
    border-radius: 14px;
    color: white;
    font-size: 16px;
    font-weight: 800;
    background: linear-gradient(95deg, #6366f1, #a855f7, #ec4899);
    box-shadow: 0 6px 25px #8b5cf655;
    transition: transform .2s ease, box-shadow .2s ease;
}

div.stButton > button:hover {
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 10px 32px #c026d355;
    border: 1px solid #f5d0fe;
}

[data-testid="stMetric"] {
    padding: 17px;
    border-radius: 16px;
    background: linear-gradient(135deg, #13254a, #1c1645);
    border: 1px solid #5b4bb7;
}

[data-testid="stMetricLabel"] {
    color: #b8c6eb !important;
}

[data-testid="stMetricValue"] {
    color: #5eead4 !important;
    font-weight: 900;
}

.car-card {
    min-height: 120px;
    height: 100%;
    border-radius: 17px;
    padding: 18px 10px;
    text-align: center;
    background: linear-gradient(145deg, #192b52, #10142f);
    border: 1px solid #364c7b;
    transition: transform .2s ease, border-color .2s ease;
}

.car-card:hover {
    transform: translateY(-4px);
    border-color: #c084fc;
}

.car-emoji {
    font-size: 30px;
    margin-bottom: 8px;
}

.car-name {
    font-weight: 800;
    color: #f1f5ff;
    font-size: 13px;
}

.car-number {
    color: #67e8f9;
    font-size: 10px;
    margin-top: 7px;
    letter-spacing: 2px;
}

.prob-row {
    padding: 10px 12px;
    margin: 7px 0;
    border-radius: 12px;
    background: #ffffff08;
    border: 1px solid #ffffff0c;
}

.footer {
    margin-top: 20px;
    padding: 20px 10px;
    text-align: center;
    border-top: 1px solid #29375c;
    color: #93a4ca;
    font-size: 12px;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #22d3ee, #8b5cf6, #ec4899);
}

@media (max-width: 700px) {
    .block-container {
        padding: 0.6rem 0.7rem;
    }
    .hero {
        padding: 24px 18px;
    }
    .pill {
        margin: 4px 2px;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="brand">
    <div class="logo">🚘 CarVision <span>AI</span></div>
    <div>
        <span class="pill">🧠 CNN POWERED</span>
        <span class="pill">✓ 7 CAR CATEGORIES</span>
    </div>
</div>

<div class="hero">
    <div class="eyebrow">NEXT-GEN AUTOMOTIVE INTELLIGENCE</div>
    <h1>See the Car.<br><span class="gradient-text">Know the Car.</span></h1>
    <p>Upload any supported car image and let our Convolutional Neural Network
    predict its category, estimate confidence and reveal the probability
    distribution in seconds.</p>
    <span class="feature">⚡ AI Classification</span>
    <span class="feature">🎯 Confidence Score</span>
    <span class="feature">📊 7-Class Analysis</span>
</div>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading your CNN model..."):
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
                "Model download failed. Check Google Drive permissions."
            )
    return tf.keras.models.load_model(MODEL_PATH)

def load_classes():
    with open("class_names.txt", "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if names != CLASSES:
        raise RuntimeError(
            "class_names.txt must contain the 7 car names in the "
            "expected training order."
        )
    return names

try:
    model = load_model()
    class_names = load_classes()
    if model.output_shape[-1] != len(class_names):
        raise RuntimeError(
            f"Model has {model.output_shape[-1]} outputs but "
            f"{len(class_names)} class names were found."
        )
except Exception as e:
    st.error("⚠️ Model setup needs attention")
    st.code(str(e))
    st.stop()

# ---------------- MAIN DASHBOARD ----------------
upload_col, result_col = st.columns([1, 1], gap="medium")

with upload_col:
    st.markdown("""
    <div class="panel upload-panel">
        <div class="eyebrow">01 / IMAGE INPUT</div>
        <div class="section-title">📸 Upload Your Car</div>
        <div class="section-sub">Give your image a moment in the spotlight.</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose JPG, JPEG or PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    image = None
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Your uploaded image",
                use_container_width=True
            )
        except Exception:
            st.error("This image could not be opened.")

    if image is not None:
        predict_clicked = st.button(
            "✨  ANALYSE MY CAR",
            type="primary",
            use_container_width=True
        )
    else:
        st.markdown("""
        <div class="panel" style="text-align:center; min-height:125px;
        display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:38px;">🖼️</div>
            <div style="font-size:16px; font-weight:800; color:#e0e7ff;">
                Your next prediction starts here
            </div>
            <div style="font-size:12px; color:#9baed5; margin-top:8px;">
                Upload a clear car photo to activate the AI
            </div>
        </div>
        """, unsafe_allow_html=True)
        predict_clicked = False

with result_col:
    st.markdown("""
    <div class="panel result-panel">
        <div class="eyebrow">02 / AI INTELLIGENCE</div>
        <div class="section-title">🎯 Prediction Studio</div>
        <div class="section-sub">Your classification results appear here.</div>
    </div>
    """, unsafe_allow_html=True)

    if image is None:
        st.markdown("""
        <div class="panel" style="text-align:center; min-height:260px;
        display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:65px;">🔮</div>
            <div style="font-size:21px; font-weight:900; color:#e0e7ff;">
                Waiting for your image
            </div>
            <div style="font-size:13px; color:#9baed5; margin-top:10px;">
                Upload an image and press Analyse My Car.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif predict_clicked:
        try:
            with st.spinner("🧠 CNN is analysing the image..."):
                resized = image.resize((128, 128))
                array = np.array(resized, dtype=np.float32)
                array = np.expand_dims(array, axis=0)
                predictions = model.predict(array, verbose=0)[0]

            index = int(np.argmax(predictions))
            confidence = float(predictions[index]) * 100

            st.markdown(f"""
            <div class="panel" style="border-color:#22d3ee;">
                <div class="eyebrow">✓ ANALYSIS COMPLETE</div>
                <div style="font-size:14px; color:#a5b4fc; margin-top:15px;">
                    YOUR PREDICTED CAR
                </div>
                <div style="font-size:clamp(27px,3vw,43px); font-weight:900;
                color:white; margin:8px 0;">
                    🚘 {class_names[index]}
                </div>
                <div style="color:#5eead4; font-weight:800;">
                    Highest model probability
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.metric("🎯 Prediction Confidence", f"{confidence:.2f}%")
            st.markdown("### 📊 Class Probability Breakdown")

            sorted_indices = np.argsort(predictions)[::-1]
            for i in sorted_indices:
                name = class_names[int(i)]
                score = float(predictions[int(i)])
                st.markdown(
                    f'<div class="prob-row"><b>{name}</b>'
                    f'<span style="float:right;color:#67e8f9;">'
                    f'{score*100:.2f}%</span></div>',
                    unsafe_allow_html=True
                )
                st.progress(score)

        except Exception as e:
            st.error("Prediction failed.")
            st.code(str(e))

    else:
        st.markdown("""
        <div class="panel" style="min-height:260px; text-align:center;
        display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:48px;">⚡</div>
            <div style="font-size:18px; color:#e0e7ff; font-weight:800;">
                Image ready for analysis
            </div>
            <div style="font-size:13px; color:#9baed5; margin-top:8px;">
                Click the glowing Analyse My Car button.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- CATEGORY GALLERY ----------------
st.markdown("---")
st.markdown("""
<div class="section-title">🚗 Meet Our 7 Car Categories</div>
<div class="section-sub">Seven categories recognised by your trained CNN model.</div>
""", unsafe_allow_html=True)

gallery = st.columns(7, gap="small")
for i, (col, name) in enumerate(zip(gallery, CLASSES), start=1):
    with col:
        st.markdown(f"""
        <div class="car-card">
            <div class="car-emoji">🚘</div>
            <div class="car-name">{name}</div>
            <div class="car-number">CLASS {i:02d}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <div style="font-size:19px; font-weight:900; color:white;">
        🚘 CarVision <span style="color:#c084fc;">AI</span>
    </div>
    <div style="margin-top:9px;">
        SMARTER VISION &nbsp; • &nbsp; BETTER INSIGHTS &nbsp; • &nbsp; CNN CLASSIFICATION
    </div>
    <div style="margin-top:8px; color:#65779e;">
        Built with Python, TensorFlow and Streamlit
    </div>
</div>
""", unsafe_allow_html=True)
