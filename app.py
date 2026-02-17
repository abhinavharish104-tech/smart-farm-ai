import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load models
irrigation_model = joblib.load("models/irrigation_model.pkl")
disease_model = load_model("models/best_model.h5")

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
        irrigation_need = irrigation_model.predict(sample)[0]
        st.success(f"Recommended irrigation: {irrigation_need:.2f} liters per plant")

# --- Disease Detection ---
with tab2:
    st.header("Disease Detection")
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png"])
    if uploaded_file:
        img = image.load_img(uploaded_file, target_size=(224,224))
        x = image.img_to_array(img)/255.0
        x = np.expand_dims(x, axis=0)
        preds = disease_model.predict(x)
        pred_class = np.argmax(preds, axis=1)[0]
        st.success(f"Disease detected: Class {pred_class}")