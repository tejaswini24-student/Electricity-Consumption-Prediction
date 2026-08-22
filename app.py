import streamlit as st
import pandas as pd
import joblib


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("xgboost_electricity_model.pkl")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Electricity Consumption Prediction",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚡ Electricity")
    st.title("Consumption Prediction")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🤖 XGBoost Prediction",
            "📊 Analytics",
            "🔮 Forecast",
            "🧠 Model Insights",
            "🚨 Smart Alerts",
            "📋 Prediction History"
        ]
    )

    st.divider()

    st.caption("⚡ Electricity Consumption Prediction")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🏠 Dashboard")

    st.write(
        "Welcome to the Electricity Consumption Prediction System."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🤖 Model",
            "XGBoost"
        )

    with col2:
        st.metric(
            "⚡ Prediction",
            "Available"
        )

    with col3:
        st.metric(
            "📊 Analytics",
            "Available"
        )

    with col4:
        st.metric(
            "🚨 Smart Alerts",
            "Active"
        )

    st.divider()

    st.subheader("⚡ Electricity Consumption Prediction")

    st.info(
        "Use the XGBoost Prediction section from the sidebar "
        "to predict electricity consumption."
    )


# ============================================================
# XGBOOST PREDICTION
# ============================================================

elif page == "🤖 XGBoost Prediction":

    st.title("🤖 XGBoost Prediction")

    st.write(
        "Enter the required details to predict electricity consumption."
    )

    st.divider()

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

    weekend_value = 1 if is_weekend == "Yes" else 0

    peak_value = 1 if is_peak_hour == "Yes" else 0

    st.divider()

    if st.button(
        "⚡ Predict Electricity Consumption",
        use_container_width=True
    ):

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

        prediction = model.predict(input_data)[0]

        st.success("Prediction completed successfully!")

        st.metric(
            label="Predicted Electricity Consumption",
            value=f"{prediction:.2f} kWh"
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Analytics")

    st.write(
        "Analyze electricity consumption patterns."
    )

    st.info(
        "Analytics dashboard will be added here."
    )


# ============================================================
# FORECAST
# ============================================================

elif page == "🔮 Forecast":

    st.title("🔮 Forecast")

    st.write(
        "View future electricity consumption forecasts."
    )

    st.info(
        "Forecasting functionality will be added here."
    )


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "🧠 Model Insights":

    st.title("🧠 Model Insights")

    st.write(
        "Understand how the XGBoost model works."
    )

    st.info(
        "Model performance and feature importance will be "
        "displayed here."
    )


# ============================================================
# SMART ALERTS
# ============================================================

elif page == "🚨 Smart Alerts":

    st.title("🚨 Smart Alerts")

    st.write(
        "Monitor electricity consumption and identify unusual "
        "or high consumption levels."
    )

    st.info(
        "Smart alert functionality will be added here."
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "📋 Prediction History":

    st.title("📋 Prediction History")

    st.write(
        "View your previous electricity consumption predictions."
    )

    st.info(
        "Prediction history will be added here."
    )
