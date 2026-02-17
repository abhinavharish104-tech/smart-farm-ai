import streamlit as st
import joblib
import numpy as np
from PIL import Image
import pickle

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_irrigation_model():
    return joblib.load("best_models/irrigation_model.pkl")

@st.cache_resource
def load_weights():
    with open("best_models/model_weights.pkl", "rb") as f:
        return pickle.load(f)

irrigation_model = load_irrigation_model()
weights = load_weights()

# ---------------- SIMPLE NN FORWARD PASS ----------------
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=1, keepdims=True)

def dense(x, w, b):
    return x @ w + b

def simple_predict(img):
    x = img.flatten().reshape(1, -1)
    w, b = weights[-1]  # last dense layer
    logits = dense(x, w, b)
    return softmax(logits)

# ---------------- UI ----------------
st.title("🌱 Smart Irrigation & Crop Disease Dashboard")

tab1, tab2 = st.tabs(["Irrigation Prediction", "Disease Detection"])

# =========================================================
# IRRIGATION TAB
# =========================================================
with tab1:

    st.header("Irrigation Recommendation")

    soil = st.slider("Soil Moisture (%)", 0, 100, 20)
    temp = st.slider("Temperature (°C)", 10, 45, 30)
    humidity = st.slider("Humidity (%)", 0, 100, 60)
    rainfall = st.selectbox("Rainfall (mm)", [0, 5, 10])
    crop = st.selectbox("Crop Type", ["Tomato", "Potato", "Pepper"])

    crop_map = {
        "Tomato":[1,0,0],
        "Potato":[0,1,0],
        "Pepper":[0,0,1]
    }

    sample = np.array([[soil, temp, humidity, rainfall] + crop_map[crop]])

    if st.button("Predict Irrigation", key="irrigate"):

        # ---------- AGRONOMIC SAFETY ----------
        if soil >= 85:
            st.error("🚫 Irrigation blocked: Soil already saturated")
            st.info("Risk: root oxygen deficiency & fungal infection")
            stress = 70 + (soil - 85) * 1.5
            status = "Over-Irrigation / Root Hypoxia Risk"

        elif rainfall >= 10:
            st.warning("Recent rainfall sufficient — irrigation skipped")
            stress = 10
            status = "Optimal Moisture"

        else:
            # ---------- ML PREDICTION ----------
            irrigation_need = irrigation_model.predict(sample)[0]
            st.success(f"Recommended irrigation: {irrigation_need:.2f} liters per plant")

            # ---------- BIOLOGICAL STRESS MODEL ----------
            if soil < 30:
                stress = 80 + (30 - soil) * 0.6
                status = "Severe Drought Stress"

            elif soil < 60:
                stress = 40 - (soil - 30) * 0.8
                status = "Mild Water Stress"

            else:
                stress = 10
                status = "Optimal Moisture"

        # ---------- FINAL OUTPUT ----------
        st.metric("Plant Stress Index", f"{int(stress)}%")
        st.write(f"Plant condition: {status}")

# =========================================================
# DISEASE TAB
# =========================================================
with tab2:

    st.header("Disease Detection")

    uploaded_file = st.file_uploader(
        "Upload Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((64, 64))
        img = np.array(img) / 255.0

        preds = simple_predict(img)
        pred_class = int(np.argmax(preds))

        st.success(f"Disease detected: Class {pred_class}")
        st.write(f"Confidence: {np.max(preds)*100:.2f}%")
