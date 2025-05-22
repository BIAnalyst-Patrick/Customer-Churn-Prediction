import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import io
import os

# --- Page Configuration ---
st.set_page_config(page_title="Customer Exit Predictor", layout="centered")

import pickle

import pickle
import requests
import io
import streamlit as st

# --- Load Model and Scaler from Google Drive ---
@st.cache_resource
def load_model_and_scaler():
    # Load model from Google Drive (model.pkl)
    model_file_id = '14sUI6NhttR32jUrOtuR8FRXwDzXwBNxs'  # model.pkl file ID
    url_model = f"https://drive.google.com/uc?export=download&id={model_file_id}"  # Correct direct download link
    response_model = requests.get(url_model)
    if response_model.status_code != 200:
        st.error(f'❌ Failed to download model.pkl from Google Drive!')
        return None, None
    model = pickle.load(io.BytesIO(response_model.content))  # Load the actual model

    # Load Random Forest model from Google Drive (rf.pkl)
    rf_file_id = '18Xw3sj2pGHOSNI65BAgdgboqlrDZW0X0'  # rf.pkl file ID
    url_rf = f"https://drive.google.com/uc?export=download&id={rf_file_id}"  # Correct direct download link
    response_rf = requests.get(url_rf)
    if response_rf.status_code != 200:
        st.error(f'❌ Failed to download rf.pkl from Google Drive!')
        return None, None
    rf_model = pickle.load(io.BytesIO(response_rf.content))  # Load the actual model

    return model, rf_model

# Load the model and Random Forest model
model, rf_model = load_model_and_scaler()

# Stop if model or Random Forest model not loaded
if not model or not rf_model:
    st.stop()  # Stop if either model or rf_model is not loaded



# --- Custom CSS ---
st.markdown("""
    <style>
        body {
            background-color: #e6f2ff;
        }
        .main {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .predict-btn {
            background-color: #0099ff;
            color: white;
            font-size: 18px;
            padding: 10px 24px;
            border: none;
            border-radius: 8px;
        }
        .predict-btn:hover {
            background-color: #007acc;
        }
        .result {
            font-size: 20px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header Image ---
if os.path.exists("header1.png"):
    st.image("header1.png", use_container_width=True)

# --- App Title ---
st.markdown('<div class="title">💼 Customer Exit Prediction</div>', unsafe_allow_html=True)
st.markdown("Predict whether a bank customer is likely to leave or stay.")

# --- Input Form ---
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", 300, 900, 600)
        age = st.slider("Age", 18, 100, 35)
        tenure = st.slider("Tenure", 0, 10, 3)
        balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
        geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])

    with col2:
        salary = st.number_input("Estimated Salary", 0.0, 200000.0, 70000.0)
        products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_card = st.selectbox("Has Credit Card", [0, 1])
        is_active = st.selectbox("Is Active Member", [0, 1])
        gender = st.selectbox("Gender", ['Male', 'Female'])

    submitted = st.form_submit_button("Predict", use_container_width=True)

# --- Encode Categorical ---
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0
gender_male = 1 if gender == "Male" else 0

# --- Prediction ---
if submitted:
    feature_names = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'Geography_Germany', 'Geography_Spain', 'Gender_Male'
    ]

    features = [credit_score, age, tenure, balance, products, has_card, is_active,
                salary, geo_germany, geo_spain, gender_male]

    input_df = pd.DataFrame([features], columns=feature_names)

    scale_vars = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]

    # Scale required features
    input_data_scaled = input_df.copy()
    input_data_scaled[scale_vars] = scaler.transform(input_data_scaled[scale_vars])

    # Prediction
    prediction = model.predict(input_data_scaled)[0]
    prob = model.predict_proba(input_data_scaled)[0][1]

    st.markdown("### 🔍 Prediction Result:")
    if prediction == 1:
        st.markdown(f"<div class='result' style='color:red;'>⚠️ The customer is <strong>likely to exit</strong>. (Probability: {prob:.2%})</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result' style='color:green;'>✅ The customer is <strong>likely to stay</strong>. (Probability: {prob:.2%})</div>", unsafe_allow_html=True)
