import streamlit as st
import joblib
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ================= LOAD MODELS =================

@st.cache_resource
def load_irrigation_model():
    return joblib.load("best_models/irrigation_model.pkl")

@st.cache_resource
def load_disease_model():
    return load_model("best_models/best_model_compat.h5")

irrigation_model = load_irrigation_model()
disease_model = load_disease_model()

# ================= UI =================

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

    if st.button("Predict Irrigation"):

        # Agronomic safety rules
        if soil >= 85:
            st.error("🚫 Irrigation blocked: Soil saturated")
            stress = 80
            status = "Root hypoxia risk"

        elif rainfall >= 10:
            st.warning("Recent rainfall sufficient")
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
                status = "Mild water stress"
            else:
                stress = 10
                status = "Healthy"

        st.metric("Plant Stress Index", f"{stress}%")
        st.write(status)

# =========================================================
# DISEASE TAB
# =========================================================
with tab2:

    st.header("Leaf Disease Detection")

    uploaded = st.file_uploader("Upload Leaf Image", type=["jpg","jpeg","png"])

    if uploaded is not None:

        img = Image.open(uploaded).convert("RGB").resize((64,64))
        img = np.array(img) / 255.0
        img = img.reshape(1, 64, 64, 3)

        preds = disease_model.predict(img)
        pred_class = int(np.argmax(preds))
        confidence = float(np.max(preds) * 100)

        st.success(f"Disease Class: {pred_class}")
        st.write(f"Confidence: {confidence:.2f}%")

