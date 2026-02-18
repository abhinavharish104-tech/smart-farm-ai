import streamlit as st
import joblib
import numpy as np

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("best_models/irrigation_model.pkl")

model = load_model()

st.set_page_config(page_title="Smart Farm AI", layout="wide")

st.title("🌱 Smart Farm Virtual IoT Dashboard")

st.markdown("Real-time crop health & irrigation decision support system")

# =====================================================
# INPUT PANEL (Virtual Sensors)
# =====================================================
st.sidebar.header("📡 Virtual Field Sensors")

soil = st.sidebar.slider("Soil Moisture (%)", 0, 100, 25)
temp = st.sidebar.slider("Temperature (°C)", 10, 45, 30)
humidity = st.sidebar.slider("Air Humidity (%)", 0, 100, 60)
rainfall = st.sidebar.selectbox("Recent Rainfall (mm)", [0, 5, 10])
crop = st.sidebar.selectbox("Crop Type", ["Tomato", "Potato", "Pepper"])

crop_map = {
    "Tomato":[1,0,0],
    "Potato":[0,1,0],
    "Pepper":[0,0,1]
}

sample = np.array([[soil, temp, humidity, rainfall] + crop_map[crop]])

# =====================================================
# DECISION ENGINE
# =====================================================
st.header("📊 AI Field Analysis")

col1, col2, col3 = st.columns(3)

# Stress calculation
if soil >= 85:
    stress = 85 + (soil - 85) * 1.2
    status = "Root Oxygen Deficiency Risk"
    irrigation = 0
    advice = "Stop irrigation immediately"

elif rainfall >= 10:
    stress = 10
    status = "Optimal Moisture"
    irrigation = 0
    advice = "Rainfall sufficient"

else:
    irrigation = model.predict(sample)[0]

    if soil < 30:
        stress = 80
        status = "Severe Drought Stress"
        advice = "Urgent irrigation required"

    elif soil < 60:
        stress = 40
        status = "Mild Water Stress"
        advice = "Moderate irrigation recommended"

    else:
        stress = 10
        status = "Healthy"
        advice = "No action needed"

# =====================================================
# DISPLAY METRICS
# =====================================================
col1.metric("💧 Water Needed (L/plant)", f"{irrigation:.2f}")
col2.metric("🌿 Plant Stress Index", f"{int(stress)}%")
col3.metric("📌 Crop Status", status)

st.subheader("🧠 AI Recommendation")
st.info(advice)

# =====================================================
# VISUAL INDICATOR
# =====================================================
st.subheader("Field Condition")

if stress > 70:
    st.error("🚨 Critical Condition")
elif stress > 40:
    st.warning("⚠️ Monitor Crop")
else:
    st.success("✅ Healthy Field")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("Virtual IoT powered Smart Farming Decision System")
