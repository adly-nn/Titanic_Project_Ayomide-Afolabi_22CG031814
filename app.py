# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        # Check if model is in current folder or 'model' subfolder
        try:
            return joblib.load('titanic_model.pkl')
        except FileNotFoundError:
            return joblib.load('model/titanic_model.pkl')
    except FileNotFoundError:
        return None

model = load_model()

# --- 3. UI LAYOUT ---
st.title("🚢 Titanic Survival Prediction System")
st.markdown("Enter passenger details to predict if they would have survived the disaster.")
st.divider()

if model is None:
    st.error("Error: Could not find 'titanic_model.pkl'. Please run model_development.py first.")
    st.stop()

# Input Columns
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        format_func=lambda x: f"{x}st Class" if x==1 else (f"{x}nd Class" if x==2 else f"{x}rd Class"),
        help="1st = Upper, 2nd = Middle, 3rd = Lower"
    )
    
    sex = st.radio("Gender", options=["Male", "Female"])
    
    age = st.slider("Age", 0, 100, 30)

with col2:
    embarked = st.selectbox(
        "Port of Embarkation",
        options=["Southampton", "Cherbourg", "Queenstown"]
    )
    
    fare = st.number_input("Fare ($)", min_value=0.0, value=32.0, step=1.0)

st.divider()

# --- 4. PREDICTION LOGIC ---
if st.button("Predict Survival", type="primary"):
    
    # 1. Encode Inputs (Must match training encoding!)
    sex_encoded = 0 if sex == "Male" else 1
    
    embarked_map = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}
    embarked_encoded = embarked_map[embarked]
    
    # 2. Create DataFrame
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'Fare': [fare],
        'Embarked': [embarked_encoded]
    })
    
    # 3. Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # Probability of survival
    
    # 4. Display Result
    if prediction == 1:
        st.success(f"**Result: SURVIVED** (Confidence: {probability:.1%})")
        st.balloons()
    else:
        st.error(f"**Result: DID NOT SURVIVE** (Confidence: {1-probability:.1%})")