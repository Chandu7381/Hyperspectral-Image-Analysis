import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="🌾 Crop Health & Nutrient Detector",
    page_icon="🌿",
    layout="wide",
)

# --- Custom CSS (Green, Modern Theme) ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #e9f5db 0%, #cfe1b9 100%);
        font-family: 'Poppins', sans-serif;
        color: #1b4332;
    }
    .main {
        background-color: rgba(255,255,255,0.95);
        padding: 2.5rem;
        border-radius: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }
    h1 {
        color: #2d6a4f;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    h2, h3 {
        color: #40916c;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2d6a4f, #40916c);
        color: white;
        border-radius: 12px;
        height: 3em;
        font-weight: 600;
        width: 100%;
        transition: 0.3s;
        border: none;
    }
    .stButton>button:hover {
        transform: scale(1.03);
        background: linear-gradient(90deg, #1b4332, #2d6a4f);
    }
    .stFileUploader>div>div>button {
        background-color: #74c69d !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    .uploadedImage {
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    .prediction-card {
        background-color: #e6ffed;
        border-radius: 20px;
        padding: 1.8rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        text-align: center;
        margin-top: 2rem;
        transition: transform 0.2s ease-in-out;
    }
    .prediction-card:hover {
        transform: scale(1.02);
    }
    /* --- Footer Styling --- */
    .footer {
        text-align: center;
        color: #ffffff;
        padding: 2rem 0;
        margin-top: 3rem;
        background: linear-gradient(90deg, #2d6a4f, #40916c);
        border-top-left-radius: 20px;
        border-top-right-radius: 20px;
        box-shadow: 0 -4px 15px rgba(0,0,0,0.15);
    }
    .footer b {
        color: #d8f3dc;
    }
    .footer p {
        margin: 0.3rem;
        font-size: 0.95rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.title("🌾 Hyperspectral Image Analysis for Nutrient Stress")
st.markdown("""
Welcome to the **Smart Agriculture Analyzer** —  
Detect **nutrient deficiency**, **water stress**, and **leaf diseases** 🌿  
powered by **AI + Hyperspectral Imaging Technology**.
""")

# --- Load ML Model ---
MODEL_PATH = os.path.join("models", "best_model.keras")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

with st.spinner("🧠 Loading Machine Learning Model..."):
    model = load_model()

# --- Load Class Labels ---
train_dir = "data/train"
class_names = sorted(os.listdir(train_dir))
st.sidebar.header("🌱 Class Categories")
st.sidebar.success(f"{len(class_names)} classes detected")
for name in class_names:
    st.sidebar.write(f"- {name}")

# --- Upload Section ---
st.markdown("### 📸 Upload Your Crop/Leaf Image")
uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="🌿 Uploaded Image", use_container_width=True, output_format="auto")

    # Preprocess image
    img_arr = np.array(img)
    img_resized = cv2.resize(img_arr, (128, 128))
    input_arr = np.expand_dims(img_resized.astype("float32") / 255.0, axis=0)

    # Predict
    preds = model.predict(input_arr)
    pred_idx = np.argmax(preds)
    confidence = float(np.max(preds)) * 100
    label = class_names[pred_idx]

    # --- Prediction Card ---
    st.markdown("---")
    st.markdown(f"""
    <div class="prediction-card">
        <h2>🧠 Prediction Result</h2>
        <h3>🌱 {label}</h3>
        <p style='font-size:18px;'>Confidence: <b>{confidence:.2f}%</b></p>
    </div>
    """, unsafe_allow_html=True)

    if confidence < 50:
        st.warning("⚠️ The model is uncertain — please try another image.")
    else:
        st.info("✅ Confidence looks good — reliable prediction.")

    # Probability breakdown
    if st.checkbox("📊 Show Probability Distribution"):
        probs = {class_names[i]: float(preds[0][i]) for i in range(min(len(class_names), len(preds[0])))}
        st.json(probs)

else:
    st.info("📥 Upload a clear, high-quality image of the crop or leaf to begin analysis.")

# --- Footer (Beautiful Green Gradient Bar) ---
st.markdown("""
<div class='footer'>
    <h4>🌾 Hyperspectral Image Analysis for Nutrient Stress</h4>
    <p><b>Developed by:</b> Adhikari Ashis Kumar Das</p>
    <p><b>Institution:</b> Centurion University of Technology and Management (CUTM)</p>
    <p>© 2025 All Rights Reserved | Academic Research Project</p>
</div>
""", unsafe_allow_html=True)
