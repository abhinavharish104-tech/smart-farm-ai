import streamlit as st
import joblib
import numpy as np
from PIL import Image
import pickle

# ---------------- LOAD MODELS ----------------
irrigation_model = joblib.load("best_models/irrigation_model.pkl")

with open("best_models/model_weights.pkl", "rb") as f:
    weights = pickle.load(f)


# ---------------- SIMPLE CNN FORWARD PASS ----------------
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=1, keepdims=True)

def dense(x, w, b):
    return x @ w + b

def simple_predict(img):
    x = img.flatten().reshape(1, -1)

    # last dense layer weights
    w, b = weights[-1]
    logits = dense(x, w, b)
    return softmax(logits)


# ---------------- UI ----------------
st.title("🌱 Smart Irrigation & Crop Disease Dashboard")

tab1, tab2 = st.tabs(["Irrigation Prediction", "Disease Detection"])

# --- Irrigation Prediction ---
with tab1:
    st.header("Irrigation Recommendation")

    soil = st.slider("Soil Moisture (%)", 0, 100, 20)
    temp = st.slider("Temperature (°C)", 10, 45, 30)
    humidity = st.slider("Humidity (%)", 0, 100, 60)
    rainfall = st.selectbox("Rainfall (mm)", [0, 5, 10])
    crop = st.selectbox("Crop Type", ["Tomato", "Potato", "Pepper"])

    crop_map = {"Tomato":[1,0,0], "Potato":[0,1,0], "Pepper":[0,0,1]}
    sample = np.array([[soil, temp, humidity, rainfall] + crop_map[crop]])

    if st.button("Predict Irrigation"):

        # -------- AGRONOMY SAFETY RULES --------
        if soil >= 85:
            st.error("🚫 Irrigation blocked: Soil already saturated")
            st.info("Risk: root oxygen deficiency & fungal infection")
            st.metric("Plant Water Stress Index", f"{100-soil}%")
            st.stop()

        if rainfall >= 10:
            st.warning("Recent rainfall sufficient — irrigation skipped")
            st.stop()

        # -------- ML PREDICTION --------
        irrigation_need = irrigation_model.predict(sample)[0]

        # -------- INTERPRETABLE OUTPUT --------
        st.success(f"Recommended irrigation: {irrigation_need:.2f} liters per plant")

        stress = 100 - soil
        st.metric("Plant Water Stress Index", f"{stress}%")

        if stress < 20:
            st.write("Plant condition: Comfortable")
        elif stress < 50:
            st.write("Plant condition: Mild water stress")
        else:
            st.write("Plant condition: Severe water stress")

# DISEASE
with tab2:
    st.header("Disease Detection")

    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).resize((64,64))
        img = np.array(img)/255.0

        preds = simple_predict(img)
        pred_class = int(np.argmax(preds))

        st.success(f"Disease detected: Class {pred_class}")


