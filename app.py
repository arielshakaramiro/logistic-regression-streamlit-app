import streamlit as st
import numpy as np
import pickle
from PIL import Image
import os

st.set_page_config(page_title="Logistic Regression Classifier", layout="centered")

# Styling
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔮 Logistic Regression Classifier")
st.markdown("Masukkan nilai input untuk memprediksi kelas (0/1) menggunakan model Logistic Regression.")

x_input = st.slider("📊 Masukkan nilai X", 0.0, 10.0, 5.0, 0.1)

# Load model
model_path = "models/logistic_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

if st.button("Prediksi"):
    x_np = np.array([[x_input]])
    y_pred = model.predict(x_np)[0]
    prob = model.predict_proba(x_np)[0][1]
    st.success(f"✅ Prediksi: {y_pred} (Probabilitas kelas 1: {prob:.2f})")

if os.path.exists("outputs/hasil_logistic_regression.png"):
    st.image("outputs/hasil_logistic_regression.png", caption="📈 Visualisasi Hasil Klasifikasi", use_column_width=True)
