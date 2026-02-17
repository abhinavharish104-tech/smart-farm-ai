import streamlit as st
import joblib
import numpy as np
from PIL import Image
import pickle

# ================= LOAD MODELS =================
@st.cache_resource
def load_irrigation_model():
    return joblib.load("best_models/irrigation_model.pkl")

@st.cache_resource
def load_weights():
    with open("best_models/model_weights.pkl", "rb") as f:
        return pickle.load(f)

irrigation_model = load_irrigation_model()
weights = load_weights()

# ================= SIMPLE NN =================
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=1, keepdims=True)

def dense(x, w, b):
    return x @ w + b

def simple_predict(img):
    x = img.flatten().reshape(1, -1)
    w, b = weights[-1]
    logits = dense(x, w, b)
    return softmax(logits)

# ================= UI =================
st.title("🌱 Smart Irrigation & Crop Disease Dashboard")

tab1, tab2 = st.tabs(["Irrigation Prediction", "Disease Detection"])

# ---------- IRRIGATION ----------
with tab1:

    soil = st.slider("Soil Moisture (%)", 0, 100, 20)
    temp = st.slider("Temperature (°C)", 10, 45, 30)
    humidity = st.slider("Humidity (%)", 0, 100, 60)
    rainfall = st.selectbox("Rainfall (mm)", [0, 5, 10])
    crop = st.selectbox("Crop Type", ["Tomato", "Potato", "Pepper"])

    crop_map = {"Tomato":[1,0,0], "Potato":[0,1,0], "Pepper":[0,0,1]}
    sample = np.array([[soil, temp, humidity, rainfall] + crop_map[crop]])

    if st.button("Predict Irrigation"):

        if soil >= 85:
            st.error("Irrigation blocked: Soil saturated")
            stress = 80
            status = "Root hypoxia risk"

        elif rainfall >= 10:
            st.warning("Rainfall sufficient")
            stress = 10
            status = "Optimal moisture"

        else:
            irrigation_need = irrigation_model.predict(sample)[0]
            st.success(f"Recommended irrigation: {irrigation_need:.2f} L/plant")

            if soil < 30:
                stress = 80
                status = "Severe drought stress"
            elif soil < 60:
                stress = 40
                status = "Mild stress"
            else:
                stress = 10
                status = "Healthy"

        st.metric("Plant Stress Index", f"{stress}%")
        st.write(status)

# ---------- DISEASE ----------
with tab2:

    uploaded = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB").resize((64,64))
        img = np.array(img)/255.0

        preds = simple_predict(img)
        pred = int(np.argmax(preds))

        st.success(f"Disease Class: {pred}")
        st.write(f"Confidence: {np.max(preds)*100:.2f}%")
