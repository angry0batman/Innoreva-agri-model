import streamlit as st
import joblib
import numpy as np
import pandas as pd


model = joblib.load('soil_model.pkl')

st.title('Soil Analysis and Crop Recommendation System')


st.header('Enter Soil Parameters:')
N = st.number_input('Nitrogen Content (N)', min_value=0, max_value=100, step=1)
P = st.number_input('Phosphorus Content (P)', min_value=0, max_value=100, step=1)
K = st.number_input('Potassium Content (K)', min_value=0, max_value=100, step=1)
pH = st.number_input('Soil pH', min_value=4.5, max_value=8.5, step=0.1)
moisture = st.number_input('Soil Moisture (%)', min_value=10.0, max_value=100.0, step=0.1)
temperature = st.number_input('Soil Temperature (°C)', min_value=15.0, max_value=35.0, step=0.1)


if st.button('Predict'):
 
    input_features = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'pH': [pH],
        'Moisture': [moisture],
        'Temperature': [temperature]
    })

 
    predictions = model.predict(input_features)

    recommended_crop = predictions[0][0]
    recommended_fertilizer_quantity = predictions[0][1]
    predicted_yield = predictions[0][2]

   
    st.write('### Recommended Crop:', recommended_crop)
    st.write('### Recommended Fertilizer & Quantity:', recommended_fertilizer_quantity)
    st.write(f'### Predicted Yield: {predicted_yield:.2f} tons/ha')
