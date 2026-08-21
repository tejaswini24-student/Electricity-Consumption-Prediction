import streamlit as st
import pandas as pd
import joblib


# ==========================================
# LOAD TRAINED XGBOOST MODEL
# ==========================================

model = joblib.load("xgboost_electricity_model.pkl")


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Electricity Consumption Prediction",
    page_icon="⚡",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title("⚡ Electricity Consumption Prediction")

st.write(
    "Enter the following details to predict electricity consumption."
)


# ==========================================
# INPUT DETAILS
# ==========================================

st.subheader("Enter Input Details")

col1, col2 = st.columns(2)


with col1:

    temperature = st.number_input(
        "Temperature (°C)",
        value=25.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

    occupancy = st.number_input(
        "Occupancy (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    hour = st.number_input(
        "Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=1
    )


with col2:

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=1
    )

    day_of_week = st.number_input(
        "Day of Week (0 = Monday, 6 = Sunday)",
        min_value=0,
        max_value=6,
        value=3
    )

    is_weekend = st.selectbox(
        "Is Weekend?",
        ["No", "Yes"]
    )

    is_peak_hour = st.selectbox(
        "Is Peak Hour?",
        ["No", "Yes"]
    )


# ==========================================
# CONVERT YES/NO TO 0/1
# ==========================================

weekend_value = 1 if is_weekend == "Yes" else 0

peak_value = 1 if is_peak_hour == "Yes" else 0


# ==========================================
# PREDICTION
# ==========================================

st.divider()

if st.button(
    "⚡ Predict Electricity Consumption",
    use_container_width=True
):

    # IMPORTANT:
    # These are the EXACT 9 features expected
    # by the trained XGBoost model.

    input_data = pd.DataFrame({

        "temperature_c": [temperature],

        "humidity_percent": [humidity],

        "occupancy_percent": [occupancy],

        "hour": [hour],

        "day": [day],

        "month": [month],

        "day_of_week": [day_of_week],

        "is_weekend": [weekend_value],

        "is_peak_hour": [peak_value]
    })


    # Make prediction directly
    prediction = model.predict(input_data)[0]


    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.success("Prediction completed successfully!")

    st.metric(
        label="Predicted Electricity Consumption",
        value=f"{prediction:.2f} kWh"
    )

    