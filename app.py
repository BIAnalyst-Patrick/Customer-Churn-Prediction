import streamlit as st
import joblib
import pandas as pd
import numpy as np
import gdown
import os

# --- Page Configuration ---
st.set_page_config(page_title="Customer Exit Predictor", layout="centered")

# --- Load Model and Scaler ---
@st.cache_resource
def load_model_and_scaler():
    # Updated file_id for model
    file_id = '18Xw3sj2pGHOSNI65BAgdgboqlrDZW0X0'  # Updated file ID
    model_url = f"https://drive.google.com/uc?id={file_id}"
    
    # Download model using gdown
    model_output = 'model.pkl'  # Name of the downloaded model file
    try:
        gdown.download(model_url, model_output, quiet=False)
    except Exception as e:
        st.error(f'❌ Failed to download the model: {e}')
        return None, None
    
    # Load the model using joblib
    try:
        model = joblib.load(model_output)
    except Exception as e:
        st.error(f'❌ Failed to load the model: {e}')
        return None, None

    # Load scaler from local file
    if not os.path.exists('scaler.pkl'):
        st.error('❌ scaler.pkl not found locally!')
        return None, None
    scaler = joblib.load('scaler.pkl')

    return model, scaler

model, scaler = load_model_and_scaler()

if not model or not scaler:
    st.stop()  # Stop if model or scaler not loaded

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


    # Prediction result display
    st.markdown("### 🔍 Prediction Result:")

    # Simple and direct probability display with styling and bold text
    st.markdown(f"""
    <div style="font-size: 24px; font-weight: bold; color: #2e8b57;">
        The probability of the customer staying is <span style="color: #9ef01a;">{prob*100:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)