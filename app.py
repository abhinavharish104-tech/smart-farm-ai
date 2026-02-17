import streamlit as st
import joblib
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# ---------------- LOAD MODELS ----------------
irrigation_model = joblib.load("models/irrigation_model.pkl")

# Load TFLite model
interpreter = tflite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- UI ----------------
st.title("🌱 Smart Irrigation & Crop Disease Dashboard")

tab1, tab2 = st.tabs(["Irrigation Prediction", "Disease Detection"])

# =====================================================
# IRRIGATION
# =====================================================
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

# =====================================================
# DISEASE DETECTION
# =====================================================
with tab2:
    st.header("Disease Detection")

    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

    if uploaded_file is not None:

        # Preprocess image
        img = Image.open(uploaded_file).resize((224,224))
        img = np.array(img)/255.0
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)

        # TFLite inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])

        pred_class = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))

        st.success(f"Disease detected: Class {pred_class}  |  Confidence: {confidence:.2f}")
