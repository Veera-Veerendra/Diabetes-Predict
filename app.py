import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load the trained model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🔍 Diabetes Prediction App")
st.write("Enter the values below to check if you might have diabetes.")

# User Input Form
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

# Prediction Button
if st.button("Predict"):
    # Convert inputs to DataFrame
    user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

    # Make Prediction
    prediction = model.predict(user_input)[0]
    prediction_proba = model.predict_proba(user_input)[0]

    # Display Result
    if prediction == 1:
        st.error(f"⚠️ High Risk! You may have diabetes.\nPrediction Confidence: {prediction_proba[1]*100:.2f}%")
    else:
        st.success(f"✅ Low Risk! You are unlikely to have diabetes.\nPrediction Confidence: {prediction_proba[0]*100:.2f}%")
