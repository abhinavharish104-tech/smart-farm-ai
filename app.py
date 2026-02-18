import streamlit as st
import joblib
import numpy as np
from PIL import Image
import onnxruntime as ort

# ================= LOAD MODELS =================

@st.cache_resource
def load_irrigation_model():
    return joblib.load("best_models/irrigation_model.pkl")

@st.cache_resource
def load_disease_model():
    return ort.InferenceSession("best_models/plant_disease.onnx")

irrigation_model = load_irrigation_model()
disease_session = load_disease_model()

# ================= DISEASE PREDICTION =================

CLASS_NAMES = ["Tomato", "Potato", "Pepper"]  # change if needed

def predict_disease(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224,224))
    img = np.array(img) / 255.0
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)

    input_name = disease_session.get_inputs()[0].name
    preds = disease_session.run(None, {input_name: img})[0]

    pred_class = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return pred_class, confidence

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

# =========================================================
# DISEASE TAB
# =========================================================

with tab2:

    st.header("Disease Detection")

    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Leaf", use_column_width=True)

        pred_class, confidence = predict_disease(uploaded_file)

        st.success(f"Disease Class: {CLASS_NAMES[pred_class]}")
        st.write(f"Confidence: {confidence*100:.2f}%")
