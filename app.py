import streamlit as st
import pickle
import pandas as pd

st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱"
)

st.title("📱 Mobile Price Prediction")
st.write("Predict a mobile phone's price range and estimated price.")

@st.cache_resource
def load_models():
    with open("mobile_price_models_compressed.pkl", "rb") as f:
        return pickle.load(f)

models = load_models()

st.success("Models loaded successfully!")
