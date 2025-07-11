import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📉 Customer Churn Prediction App")
st.markdown("Fill out the customer information to predict churn:")

# Input features
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["No", "Yes"])
dependents = st.selectbox("Has Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (in months)", 0, 72, 24)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1500.0)

# Convert inputs into DataFrame
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [1 if senior == "Yes" else 0],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'InternetService': [internet_service],
    'Contract': [contract],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# Encode same as during training
cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'InternetService', 'Contract']
input_data = pd.get_dummies(input_data, columns=cat_cols)

# Align with training features
expected_cols = scaler.feature_names_in_
for col in expected_cols:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[expected_cols]

# Scale
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict Churn"):
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]
    if pred == 1:
        st.error(f"⚠️ The customer is likely to churn. (Probability: {prob:.2f})")
    else:
        st.success(f"✅ The customer is likely to stay. (Probability: {prob:.2f})")
