import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("diabetes_xgb_model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")
st.title("🩺 Diabetes Risk Prediction")
st.markdown("Enter patient data to estimate diabetes risk. Out-of-range inputs will be flagged.")

# Input fields
glucose = st.number_input("Blood Glucose Level", min_value=0.0, value=100.0)
hba1c = st.number_input("HbA1c Level", min_value=0.0, value=5.5)
bmi = st.number_input("BMI", min_value=0.0, value=24.0)
age = st.number_input("Age", min_value=1, value=30)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
smoking = st.selectbox("Smoking History", ["never", "former", "current", "unknown"])

# Warnings for out-of-range inputs (based on EDA thresholds)
if glucose < 11.5 or glucose > 247.5:
    st.warning("⚠️ Glucose value is outside expected range [11.5 - 247.5].")
if hba1c < 2.7 or hba1c > 8.3:
    st.warning("⚠️ HbA1c value is outside expected range [2.7 - 8.3].")
if bmi < 14.7 or bmi > 38.5:
    st.warning("⚠️ BMI value is outside expected range [14.7 - 38.5].")

# Preprocess
def preprocess_inputs():
    input_dict = {
        "HbA1c_level": [hba1c],
        "blood_glucose_level": [glucose],
        "bmi": [bmi],
        "cleaned_age": [age],
        "hypertension": [1 if hypertension == "Yes" else 0],
        "heart_disease": [1 if heart_disease == "Yes" else 0],
        "gender_Male": [1 if gender == "Male" else 0],
        "gender_Other": [1 if gender == "Other" else 0],
        "smoking_history_former": [1 if smoking == "former" else 0],
        "smoking_history_never": [1 if smoking == "never" else 0],
        "smoking_history_unknown": [1 if smoking == "unknown" else 0]
    }

    df = pd.DataFrame(input_dict)

    # Scale numeric features
    numeric_cols = ["HbA1c_level", "blood_glucose_level", "bmi", "cleaned_age"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df

if st.button("Predict Diabetes Risk"):
    input_df = preprocess_inputs()
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ High Risk: Likely diabetic. Risk Score: {probability:.2f}%")
    else:
        st.success(f"✅ Low Risk: Unlikely to be diabetic. Risk Score: {probability:.2f}%")