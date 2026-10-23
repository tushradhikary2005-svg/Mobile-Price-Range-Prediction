

import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱")

@st.cache_resource
def load_models():
    return joblib.load("mobile_price_models_compressed.pkl")

models = load_models()
range_model = models["price_range_model"]
price_model = models["price_model"]

st.title("📱 Mobile Price Prediction")
st.write("Enter mobile specifications below.")


phone_weight = st.number_input("Phone Weight (g)", 100.0, 400.0, 200.0)
rear_camera_mp = st.number_input("Rear Main Camera (MP)", 2.0, 250.0, 50.0)
rear_camera_count = st.number_input("Rear Camera Count", 1, 10, 3)
front_camera_mp = st.number_input("Front Camera (MP)", 2.0, 100.0, 16.0)
ram_gb = st.selectbox("RAM (GB)", [3, 4, 8, 16, 32])
storage_gb = st.selectbox("Storage (GB)", [64, 128, 256, 512, 1024])
battery_mah = st.number_input("Battery (mAh)", 1000, 10000, 5000)
display_size = st.number_input("Display Size (inches)", 3.0, 10.0, 6.5)
support_5g = st.selectbox("5G Support", ["Yes", "No"])
dual_sim = st.selectbox("Dual SIM", ["Yes", "No"])
charger_wattage = st.selectbox("Charger Wattage (W)", [10, 18, 20, 25, 32, 42, 45, 65, 70, 80, 100, 120])

if st.button("Predict Mobile Price"):
    input_data = pd.DataFrame([{
        "Phone_Weight_g": phone_weight,
        "Rear_Main_Camera_MP": rear_camera_mp,
        "Rear_Camera_Count": rear_camera_count,
        "Front_Camera_MP": front_camera_mp,
        "RAM_GB": ram_gb,
        "Storage_GB": storage_gb,
        "Battery_mAh": battery_mah,
        "Display_Size_inches": display_size,
        "5G_Support": support_5g,
        "Dual_SIM": dual_sim,
        "Charger_Wattage_W": charger_wattage
    }])

    input_data = input_data[models["feature_columns"]]

    predicted_range = range_model.predict(input_data)[0]
    predicted_price = price_model.predict(input_data)[0]

    st.success(f"Predicted Price Range: {predicted_range}")
    st.metric("Estimated Mobile Price", f"₹{predicted_price:,.0f}")
