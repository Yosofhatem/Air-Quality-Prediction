import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import joblib  # For loading the scaler

# Load the pre-trained scaler and model
try:
    scaler = joblib.load('scaler.pkl')  # Load the pre-fitted scaler using joblib
except FileNotFoundError:
    st.error("Scaler file not found. Please ensure 'scaler.pkl' is present.")
    st.stop()

try:
    model = load_model('model.keras')  # Load the trained Keras model
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Define feature columns
columns = ['T', 'RH', 'NO2(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'AH']

st.markdown("<h1 style='text-align: center;'>Air Quality Prediction App ☢️</h1>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: center;'>Enter the following details to get a prediction of the air quality.</h5>", unsafe_allow_html=True)


# Create input fields for each feature
inputs = {}
for col in columns:
    inputs[col] = st.number_input(f"Enter value for {col}", value=0.0000, format="%.4f")

# Convert the inputs to a DataFrame
df_test = pd.DataFrame([inputs])

# Preprocess the data (scaling)
df_scaled = scaler.transform(df_test)  # Ensure scaler is fitted before calling transform()

# Define the limits for CO(GT) prediction in mg/m³
MIN_CO = 0.0  # Minimum limit
MAX_CO = 1150.0  # Maximum limit

# Define CO concentration levels and corresponding health risk messages
CO_CATEGORIES = {
    "Normal": (0.0, 11.5),
    "Cautionary": (11.5, 40.0),
    "Moderate": (40.0, 115.0),
    "Hazardous": (115.0, 345.0),
    "Dangerous": (345.0, 1150.0),
    "Life-Threatening": (1150.0, float('inf'))
}

# Function to classify the CO level based on concentration
def classify_co_level(co_value):
    for category, (low, high) in CO_CATEGORIES.items():
        if low <= co_value < high:
            return category
    return "Unknown"

# Assuming `model` is your trained model and `df_scaled` is the scaled data for prediction
if st.button('Make Prediction'):
    try:
        # Make the prediction
        predictions = model.predict(df_scaled)  # This will predict for the entire df_scaled

        # Extract the first prediction value (assuming it's a single output model)
        predicted_value = predictions[0][0]

        # Apply the limits using numpy.clip to ensure the predicted value stays within the defined range
        limited_prediction = np.clip(predicted_value, MIN_CO, MAX_CO)

        # Classify the CO level based on the predicted value
        co_category = classify_co_level(limited_prediction)

        # Display the prediction and the corresponding health message
        st.write(f"Prediction For CO(GT): {limited_prediction:.6f} mg/m³")
        st.write(f"CO Level Category: {co_category}")

        # Health Risk message based on the category
        if co_category == "Normal":
            st.success("The CO level is within safe limits. No immediate action required. 🛡️")
        elif co_category == "Cautionary":
            st.warning("CO levels are slightly elevated. Monitor the air quality. ⚠️")
        elif co_category == "Moderate":
            st.warning("Moderate CO levels detected. Prolonged exposure could be harmful. 🚨")
        elif co_category == "Hazardous":
            st.error("Hazardous CO levels detected! Immediate action is required. 💀")
        elif co_category == "Dangerous":
            st.error("Dangerous CO levels detected! Evacuate and seek medical attention. ⚠️☠️🚨")
        elif co_category == "Life-Threatening":
            st.error("Life-threatening CO levels detected! Immediate evacuation and intervention required. 😨")

    except Exception as e:
        st.error(f"Error making prediction: {e}")