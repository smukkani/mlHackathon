import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("modelDT.pkl")

st.title('Customer Interest Prediction for Insurance')

# User Input Features
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=18, max_value=100, value=30)
dl = st.selectbox('Driving License', [0, 1])
region = st.number_input('Region Code', min_value=0.0, max_value=100.0, value=28.0)
isInsured = st.selectbox('Previously Insured', [0, 1])
vehicleAge = st.selectbox('Vehicle Age', ['< 1 Year', '1-2 Year', '> 2 Years'])
isDamaged = st.selectbox('Vehicle Damage', ['Yes', 'No'])
premium = st.number_input('Annual Premium', min_value=0.0, value=30000.0)
salesChannel = st.number_input('Policy Sales Channel', min_value=0.0, max_value=200.0, value=26.0)
days = st.number_input('Vintage (Days with Company)', min_value=0, max_value=500, value=200)

# Convert categorical inputs to match trained model encoding
input_data = {
    "Gender": [1 if gender == 'Male' else 0],
    "Age": [age],
    "Driving_License": [dl],
    "Region_Code": [region],
    "Previously_Insured": [isInsured],
    "Vehicle_Age": [2 if vehicleAge == '> 2 Years' else (1 if vehicleAge == '1-2 Year' else 0)],
    "Vehicle_Damage": [1 if isDamaged == 'Yes' else 0],
    "Annual_Premium": [premium],
    "Policy_Sales_Channel": [salesChannel],
    "Vintage": [days]
}

# Convert input to DataFrame
X_input = pd.DataFrame(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(X_input)[0]
    result = "Will be Interested" if prediction == 1 else "Will Not be Interested"
    st.write(f"Predicted Result: **{result}**")