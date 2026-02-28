import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------------
# Load saved objects
# -----------------------------------
with open("personality_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# -----------------------------------
# App config
# -----------------------------------
st.set_page_config(
    page_title="Personality Type Predictor",
    layout="centered"
)

st.title("Personality Type Prediction")
st.write("Provide the inputs below to predict the personality type.")

# -----------------------------------
# Feature list (MUST match training order)
# -----------------------------------
features = [
    'social_energy',
    'alone_time_preference',
    'talkativeness',
    'deep_reflection',
    'group_comfort',
    'party_liking',
    'listening_skill',
    'empathy',
    'organization',
    'leadership',
    'risk_taking',
    'public_speaking_comfort',
    'curiosity',
    'routine_preference',
    'excitement_seeking',
    'friendliness',
    'planning',
    'spontaneity',
    'adventurousness',
    'reading_habit',
    'sports_interest',
    'online_social_usage',
    'travel_desire',
    'gadget_usage',
    'work_style_collaborative',
    'decision_speed'
]

# -----------------------------------
# Input UI
# -----------------------------------
input_data = {}

for feature in features:
    input_data[feature] = st.slider(
        label=feature.replace("_", " ").title(),
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.5
    )

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("Predict Personality Type"):
    input_df = pd.DataFrame([input_data])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction_encoded = model.predict(input_scaled)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    st.success(f"Predicted Personality Type: **{prediction_label}**")