import streamlit as st
import pandas as pd
import joblib
import numpy as np


model = joblib.load('soil_model.pkl')
crop_encoder = joblib.load('crop_encoder.pkl')
fertilizer_encoder = joblib.load('fertilizer_encoder.pkl')


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

 
    crop_prediction = crop_encoder.inverse_transform([predictions[0][:len(crop_encoder.categories_[0])]])
    fertilizer_prediction = fertilizer_encoder.inverse_transform([predictions[0][len(crop_encoder.categories_[0]):-1]])
    yield_prediction = predictions[0][-1]

  
    st.write('### Recommended Crop:', crop_prediction[0][0])
    st.write('### Recommended Fertilizer & Quantity:', fertilizer_prediction[0][0])
    st.write(f'### Predicted Yield: {yield_prediction:.2f} tons/ha')
