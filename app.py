import streamlit as st
import numpy as np
import pickle
from PIL import Image
import os

# Load model
model_path = "models/logistic_model.pkl"
with open(model_path, "rb") as file:
    loaded_model = pickle.load(file)

# Tampilan Streamlit
st.title("Prediksi Klasifikasi dengan Logistic Regression")

# Input dari pengguna
x_input = st.number_input("Masukkan nilai X:", min_value=0.0, max_value=10.0, step=0.1)

# Tombol untuk prediksi
if st.button("Prediksi"):
    x_np = np.array([[x_input]])
    y_pred = loaded_model.predict(x_np)[0]
    prob = loaded_model.predict_proba(x_np)[0][1]

    st.success(f"Hasil Prediksi: {y_pred} (Probabilitas kelas 1: {prob:.2f})")

# Menampilkan grafik hasil training (jika tersedia)
image_path = "outputs/hasil_logistic_regression.png"
if os.path.exists(image_path):
    st.subheader("Visualisasi Hasil Model")
    image = Image.open(image_path)
    st.image(image, caption="Grafik Prediksi vs Data Asli")
