import streamlit as st
import pandas as pd
import joblib


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

model = joblib.load("xgboost_electricity_model.pkl")

preprocessor = joblib.load("xgboost_preprocessor.pkl")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_consumption(
    temperature,
    humidity,
    occupancy,
    hour,
    day,
    month,
    day_of_week,
    is_weekend,
    season,
    is_peak_hour
):

    weekend_value = 1 if is_weekend == "Yes" else 0
    peak_value = 1 if is_peak_hour == "Yes" else 0

    input_data = pd.DataFrame({

        "temperature_c": [temperature],

        "humidity_percent": [humidity],

        "occupancy_percent": [occupancy],

        "hour": [hour],

        "day": [day],

        "month": [month],

        "day_of_week": [day_of_week],

        "is_weekend": [weekend_value],

        "season": [season],

        "is_peak_hour": [peak_value]
    })

    processed_input = preprocessor.transform(input_data)

    prediction = model.predict(processed_input)[0]

    return float(prediction)


# ============================================================
# PREDICTION HISTORY
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


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
            "🔬 Energy Impact Simulator",
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

        season = st.selectbox(
            "🌤️ Season",
            ["Winter", "Spring", "Summer", "Autumn"]
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

            "season": [season],

            "is_peak_hour": [peak_value]
        })

        processed_input = preprocessor.transform(input_data)

        prediction = model.predict(processed_input)[0]

        # Save prediction to history
        st.session_state.prediction_history.append({
           "Temperature": temperature,
            "Humidity": humidity,
            "Occupancy": occupancy,
            "Hour": hour,
            "Day": day,
            "Month": month,
            "Day of Week": day_of_week,
            "Weekend": is_weekend,
            "Peak Hour": is_peak_hour,
            "Season": season,
            "Prediction (kWh)": round(float(prediction), 2)
        })

        st.success("Prediction completed successfully!")

        st.metric(
        label="Predicted Electricity Consumption",
        value=f"{prediction:.2f} kWh"
    )


# ============================================================
# ENERGY IMPACT SIMULATOR
# ============================================================

elif page == "🔬 Energy Impact Simulator":

    st.title("🔬 Energy Impact Simulator")

    st.write(
        "Explore how changing electricity-use conditions "
        "could affect predicted consumption."
    )

    st.info(
        "💡 Change one or more conditions and compare the "
        "new scenario with the current scenario."
    )

    st.divider()

    # --------------------------------------------------------
    # CURRENT SCENARIO
    # --------------------------------------------------------

    st.subheader("📌 Current Scenario")

    col1, col2 = st.columns(2)

    with col1:

        current_temperature = st.number_input(
            "🌡️ Current Temperature (°C)",
            value=25.0,
            key="sim_current_temperature"
        )

        current_humidity = st.number_input(
            "💧 Current Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            key="sim_current_humidity"
        )

        current_occupancy = st.number_input(
            "🏠 Current Occupancy (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            key="sim_current_occupancy"
        )

        current_hour = st.number_input(
            "🕐 Current Hour",
            min_value=0,
            max_value=23,
            value=12,
            key="sim_current_hour"
        )

        current_day = st.number_input(
            "📅 Current Day",
            min_value=1,
            max_value=31,
            value=1,
            key="sim_current_day"
        )

    with col2:

        current_month = st.number_input(
            "📆 Current Month",
            min_value=1,
            max_value=12,
            value=1,
            key="sim_current_month"
        )

        current_day_of_week = st.number_input(
            "📅 Current Day of Week (0 = Monday)",
            min_value=0,
            max_value=6,
            value=3,
            key="sim_current_day_of_week"
        )

        current_weekend = st.selectbox(
            "Weekend?",
            ["No", "Yes"],
            key="sim_current_weekend"
        )

        current_peak = st.selectbox(
            "Peak Hour?",
            ["No", "Yes"],
            key="sim_current_peak"
        )

        current_season = st.selectbox(
            "🌤️ Current Season",
            ["Winter", "Spring", "Summer", "Autumn"],
            key="sim_current_season"
        )

    st.divider()

    # --------------------------------------------------------
    # SIMULATED SCENARIO
    # --------------------------------------------------------

    st.subheader("🔬 Simulated Scenario")

    st.write(
        "Now change the conditions to test a different "
        "energy-use scenario."
    )

    col1, col2 = st.columns(2)

    with col1:

        scenario_temperature = st.number_input(
            "🌡️ Scenario Temperature (°C)",
            value=25.0,
            key="sim_scenario_temperature"
        )

        scenario_humidity = st.number_input(
            "💧 Scenario Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            key="sim_scenario_humidity"
        )

        scenario_occupancy = st.number_input(
            "🏠 Scenario Occupancy (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            key="sim_scenario_occupancy"
        )

        scenario_hour = st.number_input(
            "🕐 Scenario Hour",
            min_value=0,
            max_value=23,
            value=18,
            key="sim_scenario_hour"
        )

        scenario_day = st.number_input(
            "📅 Scenario Day",
            min_value=1,
            max_value=31,
            value=1,
            key="sim_scenario_day"
        )

    with col2:

        scenario_month = st.number_input(
            "📆 Scenario Month",
            min_value=1,
            max_value=12,
            value=1,
            key="sim_scenario_month"
        )

        scenario_day_of_week = st.number_input(
            "📅 Scenario Day of Week (0 = Monday)",
            min_value=0,
            max_value=6,
            value=3,
            key="sim_scenario_day_of_week"
        )

        scenario_weekend = st.selectbox(
            "Scenario Weekend?",
            ["No", "Yes"],
            key="sim_scenario_weekend"
        )

        scenario_peak = st.selectbox(
            "Scenario Peak Hour?",
            ["No", "Yes"],
            key="sim_scenario_peak"
        )

        scenario_season = st.selectbox(
            "🌤️ Scenario Season",
            ["Winter", "Spring", "Summer", "Autumn"],
            key="sim_scenario_season"
        )

    st.divider()

    # --------------------------------------------------------
    # SIMULATE
    # --------------------------------------------------------

    if st.button(
        "🔬 Simulate Energy Impact",
        use_container_width=True
    ):

        current_prediction = predict_consumption(
            current_temperature,
            current_humidity,
            current_occupancy,
            current_hour,
            current_day,
            current_month,
            current_day_of_week,
            current_weekend,
            current_season,
            current_peak
        )

        scenario_prediction = predict_consumption(
            scenario_temperature,
            scenario_humidity,
            scenario_occupancy,
            scenario_hour,
            scenario_day,
            scenario_month,
            scenario_day_of_week,
            scenario_weekend,
            scenario_season,
            scenario_peak
        )

        difference = scenario_prediction - current_prediction

        if current_prediction != 0:
            percentage_change = (
                difference / current_prediction
            ) * 100
        else:
            percentage_change = 0

        st.success(
            "Simulation completed successfully!"
        )

        st.divider()

        # ----------------------------------------------------
        # COMPARISON
        # ----------------------------------------------------

        st.subheader("📊 Scenario Comparison")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Current Prediction",
                f"{current_prediction:.2f} kWh"
            )

        with col2:

            st.metric(
                "Simulated Prediction",
                f"{scenario_prediction:.2f} kWh"
            )

        with col3:

            st.metric(
                "Impact",
                f"{difference:+.2f} kWh",
                delta=f"{percentage_change:+.1f}%"
            )

        st.divider()

        # ----------------------------------------------------
        # IMPACT ANALYSIS
        # ----------------------------------------------------

        st.subheader("💡 Energy Impact Analysis")

        if difference > 0:

            st.warning(
                f"⚠️ The simulated scenario increases predicted "
                f"electricity consumption by "
                f"**{difference:.2f} kWh** "
                f"({percentage_change:.1f}%)."
            )

            st.write(
                "The changed conditions result in a higher "
                "predicted electricity demand."
            )

        elif difference < 0:

            reduction = abs(difference)

            st.success(
                f"🌱 The simulated scenario reduces predicted "
                f"electricity consumption by "
                f"**{reduction:.2f} kWh** "
                f"({abs(percentage_change):.1f}%)."
            )

            st.write(
                "The simulated conditions show a lower "
                "predicted electricity demand."
            )

        else:

            st.info(
                "ℹ️ The simulated scenario produces the same "
                "predicted electricity consumption."
            )

        st.divider()

        # ----------------------------------------------------
        # VISUAL IMPACT ANALYSIS
        # ----------------------------------------------------

        st.subheader("📈 Energy Impact Visualization")

        # Create comparison data
        comparison_df = pd.DataFrame({
            "Scenario": [
                "Current",
                "Simulated"
            ],
            "Consumption (kWh)": [
                current_prediction,
                scenario_prediction
            ]
        })

        # Display chart
        st.bar_chart(
            comparison_df.set_index("Scenario"),
            height=350
        )

        st.divider()

        # ----------------------------------------------------
        # IMPACT INDICATOR
        # ----------------------------------------------------

        if difference > 0:

            st.error(
                f"⚠️ Energy Impact: +{difference:.2f} kWh "
                f"({percentage_change:+.1f}%)"
            )

            st.write(
                f"The simulated conditions are expected to consume "
                f"**{difference:.2f} kWh more electricity** than the "
                f"current scenario."
            )

        elif difference < 0:

            st.success(
                f"🌱 Potential Energy Saving: "
                f"{abs(difference):.2f} kWh "
                f"({abs(percentage_change):.1f}%)"
            )

            st.write(
                f"The simulated conditions could reduce predicted "
                f"consumption by **{abs(difference):.2f} kWh**."
            )

        else:

            st.info(
                "ℹ️ No predicted change in electricity consumption."
            )

        st.divider()

        # ----------------------------------------------------
        # WHAT CHANGED?
        # ----------------------------------------------------

        st.subheader("🧠 What Changed?")

        st.write(
            "The simulated scenario is compared with the current "
            "scenario using the trained XGBoost model."
        )

        if difference > 0:

            st.warning(
                "The simulated conditions create higher predicted "
                "electricity demand."
            )

        elif difference < 0:

            st.success(
                "The simulated conditions create lower predicted "
                "electricity demand."
            )

        else:

            st.info(
                "Both scenarios produce approximately the same prediction."
            )

        st.divider()

        st.caption(
            "ℹ️ The simulator uses the trained XGBoost model "
            "to compare predictions under different input conditions. "
            "The result is a model-based scenario estimate, not a "
            "guarantee of actual energy savings."
        )
        
# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Analytics")

    st.write(
        "Simple analysis of your electricity consumption predictions."
    )

    st.divider()

    # --------------------------------------------------------
    # CHECK HISTORY
    # --------------------------------------------------------

    if len(st.session_state.prediction_history) == 0:

        st.info(
            "No prediction data available yet."
        )

        st.write(
            "Go to 🤖 XGBoost Prediction and make a prediction "
            "to see your analytics here."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        # ----------------------------------------------------
        # CONSUMPTION OVERVIEW
        # ----------------------------------------------------

        st.subheader("⚡ Consumption Overview")

        total_predictions = len(history_df)

        average_consumption = history_df[
            "Prediction (kWh)"
        ].mean()

        highest_consumption = history_df[
            "Prediction (kWh)"
        ].max()

        lowest_consumption = history_df[
            "Prediction (kWh)"
        ].min()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Predictions",
                total_predictions
            )

        with col2:
            st.metric(
                "Average",
                f"{average_consumption:.2f} kWh"
            )

        with col3:
            st.metric(
                "Highest",
                f"{highest_consumption:.2f} kWh"
            )

        with col4:
            st.metric(
                "Lowest",
                f"{lowest_consumption:.2f} kWh"
            )

        st.divider()

        # ----------------------------------------------------
        # LATEST PREDICTION
        # ----------------------------------------------------

        st.subheader("🔍 Latest Prediction")

        latest = history_df.iloc[-1]

        latest_consumption = latest["Prediction (kWh)"]

        if latest_consumption < 5:

            status = "🟢 Low Consumption"

            explanation = (
                "The predicted electricity consumption is relatively low."
            )

        elif latest_consumption < 8:

            status = "🟡 Moderate Consumption"

            explanation = (
                "The predicted electricity consumption is in a moderate range."
            )

        else:

            status = "🔴 High Consumption"

            explanation = (
                "The predicted electricity consumption is relatively high."
            )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Consumption",
                f"{latest_consumption:.2f} kWh"
            )

        with col2:

            st.metric(
                "Status",
                status
            )

        st.write(explanation)

        st.divider()

        # ----------------------------------------------------
        # LATEST INPUT CONDITIONS
        # ----------------------------------------------------

        st.subheader("📝 Latest Input Conditions")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🌡️ Temperature",
                f"{latest['Temperature']} °C"
            )

        with col2:
            st.metric(
                "💧 Humidity",
                f"{latest['Humidity']} %"
            )

        with col3:
            st.metric(
                "🏠 Occupancy",
                f"{latest['Occupancy']} %"
            )

        with col4:
            st.metric(
                "🕐 Hour",
                int(latest["Hour"])
            )

        col1, col2 = st.columns(2)

        with col1:
            st.write(
                f"**Weekend:** {latest['Weekend']}"
            )

        with col2:
            st.write(
                f"**Peak Hour:** {latest['Peak Hour']}"
            )

        st.divider()

        # ----------------------------------------------------
        # RECENT PREDICTIONS
        # ----------------------------------------------------

        st.subheader("📋 Recent Predictions")

        display_df = history_df[
            [
                "Temperature",
                "Humidity",
                "Occupancy",
                "Hour",
                "Weekend",
                "Peak Hour",
                "Season",
                "Prediction (kWh)"
            ]
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# FORECAST
# ============================================================

elif page == "🔮 Forecast":

    st.title("🔮 Forecast")

    st.write(
        "Estimate electricity consumption for the upcoming hours."
    )

    st.divider()

    # --------------------------------------------------------
    # FORECAST INPUTS
    # --------------------------------------------------------

    st.subheader("⚙️ Forecast Settings")

    col1, col2 = st.columns(2)

    with col1:

        temperature = st.number_input(
            "🌡️ Temperature (°C)",
            value=25.0
        )

        humidity = st.number_input(
            "💧 Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

        occupancy = st.number_input(
            "🏠 Occupancy (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0
        )

        starting_hour = st.slider(
            "🕐 Starting Hour",
            min_value=0,
            max_value=23,
            value=12
        )

    with col2:

        forecast_hours = st.slider(
            "⏱️ Hours to Forecast",
            min_value=1,
            max_value=12,
            value=6
        )

        day = st.number_input(
            "📅 Day of Month",
            min_value=1,
            max_value=31,
            value=15
        )

        month = st.number_input(
            "📆 Month",
            min_value=1,
            max_value=12,
            value=6
        )

        season = st.selectbox(
            "🌤️ Season",
            ["Winter", "Spring", "Summer", "Autumn"]
        )

    col1, col2 = st.columns(2)

    with col1:

        is_weekend = st.selectbox(
            "Weekend?",
            ["No", "Yes"]
        )

    with col2:

        is_peak_hour = st.selectbox(
            "Peak Hour?",
            ["No", "Yes"]
        )

    weekend_value = 1 if is_weekend == "Yes" else 0

    peak_value = 1 if is_peak_hour == "Yes" else 0

    st.divider()

    # --------------------------------------------------------
    # GENERATE FORECAST
    # --------------------------------------------------------

    if st.button(
        "🔮 Generate Forecast",
        use_container_width=True
    ):

        forecast_results = []

        for i in range(forecast_hours):

            future_hour = (starting_hour + i) % 24

            # Keep the same day-of-week value
            # used by the current prediction system
            future_day_of_week = 0

            # ------------------------------------------------
            # CREATE INPUT DATA
            # ------------------------------------------------

            input_data = pd.DataFrame({

                "temperature_c": [temperature],

                "humidity_percent": [humidity],

                "occupancy_percent": [occupancy],

                "hour": [future_hour],

                "day": [day],

                "month": [month],

                "day_of_week": [future_day_of_week],

                "is_weekend": [weekend_value],

                "season": [season],

                "is_peak_hour": [peak_value]

            })

            # ------------------------------------------------
            # PREPROCESS INPUT
            # ------------------------------------------------

            processed_input = preprocessor.transform(
                input_data
            )

            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            prediction = model.predict(
                processed_input
            )[0]

            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            forecast_results.append({

                "Hour": f"{future_hour:02d}:00",

                "Predicted Consumption (kWh)": round(
                    float(prediction),
                    2
                )

            })

        # ----------------------------------------------------
        # CREATE FORECAST DATAFRAME
        # ----------------------------------------------------

        forecast_df = pd.DataFrame(
            forecast_results
        )

        # ----------------------------------------------------
        # FORECAST SUMMARY
        # ----------------------------------------------------

        st.subheader("📊 Forecast Summary")

        average_forecast = forecast_df[
            "Predicted Consumption (kWh)"
        ].mean()

        highest_forecast = forecast_df[
            "Predicted Consumption (kWh)"
        ].max()

        lowest_forecast = forecast_df[
            "Predicted Consumption (kWh)"
        ].min()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "📊 Average",
                f"{average_forecast:.2f} kWh"
            )

        with col2:

            st.metric(
                "⬆️ Highest",
                f"{highest_forecast:.2f} kWh"
            )

        with col3:

            st.metric(
                "⬇️ Lowest",
                f"{lowest_forecast:.2f} kWh"
            )

        st.divider()

        # ----------------------------------------------------
        # FUTURE CONSUMPTION
        # ----------------------------------------------------

        st.subheader("📋 Future Consumption")

        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # SIMPLE ANALYSIS
        # ----------------------------------------------------

        st.subheader("💡 Forecast Analysis")

        if highest_forecast > 8:

            st.warning(
                "⚠️ Some upcoming hours show relatively "
                "high predicted electricity consumption."
            )

        elif average_forecast < 5:

            st.success(
                "🟢 The forecast shows relatively low "
                "electricity consumption."
            )

        else:

            st.info(
                "🟡 The forecast shows moderate "
                "electricity consumption."
            )


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "🧠 Model Insights":

    st.title("🧠 Model Insights")

    st.write(
        "Understand how the XGBoost model works and "
        "which features influence electricity consumption."
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL OVERVIEW
    # --------------------------------------------------------

    st.subheader("🤖 Model Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model",
            "XGBoost"
        )

    with col2:
        st.metric(
            "Task",
            "Regression"
        )

    with col3:
        st.metric(
            "Prediction",
            "Electricity Consumption"
        )

    st.divider()

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("📊 Feature Importance")

    try:

        importance = model.feature_importances_

        feature_names = [
            "Temperature",
            "Humidity",
            "Occupancy",
            "Hour",
            "Day",
            "Month",
            "Day of Week",
            "Weekend",
            "Peak Hour"
        ]

        # Make sure lengths match
        if len(importance) == len(feature_names):

            importance_df = pd.DataFrame({

                "Feature": feature_names,

                "Importance": importance

            }).sort_values(
                "Importance",
                ascending=False
            )

            st.bar_chart(
                importance_df.set_index("Feature")
            )

            st.dataframe(
                importance_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "Feature importance is available, but the "
                "feature names do not exactly match the model."
            )

    except Exception as e:

        st.warning(
            f"Unable to display feature importance: {e}"
        )

    st.divider()

    # --------------------------------------------------------
    # HOW XGBOOST WORKS
    # --------------------------------------------------------

    st.subheader("💡 How XGBoost Predicts")

    st.write(
        """
        XGBoost is a machine learning algorithm based on
        decision trees. It builds multiple trees sequentially,
        where each new tree attempts to improve the errors made
        by previous trees.

        The model uses input conditions such as temperature,
        humidity, occupancy, time and calendar information to
        estimate electricity consumption.
        """
    )


# ============================================================
# SMART ALERTS
# ============================================================

elif page == "🚨 Smart Alerts":

    st.title("🚨 Smart Alerts")

    st.write(
        "Monitor electricity consumption and identify "
        "high or unusual predicted consumption."
    )

    st.divider()

    if len(st.session_state.prediction_history) == 0:

        st.info(
            "No prediction data available yet."
        )

        st.write(
            "Go to 🤖 XGBoost Prediction and make a prediction "
            "to activate smart alerts."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        latest = history_df.iloc[-1]

        latest_consumption = float(
            latest["Prediction (kWh)"]
        )

        # ----------------------------------------------------
        # CURRENT STATUS
        # ----------------------------------------------------

        st.subheader("⚡ Current Consumption Status")

        if latest_consumption >= 8:

            st.error(
                "🔴 HIGH CONSUMPTION"
            )

            st.write(
                f"The latest prediction is "
                f"**{latest_consumption:.2f} kWh**, "
                "which is above the high-consumption threshold."
            )

        elif latest_consumption >= 5:

            st.warning(
                "🟡 MODERATE CONSUMPTION"
            )

            st.write(
                f"The latest prediction is "
                f"**{latest_consumption:.2f} kWh**. "
                "Consumption is within a moderate range."
            )

        else:

            st.success(
                "🟢 LOW CONSUMPTION"
            )

            st.write(
                f"The latest prediction is "
                f"**{latest_consumption:.2f} kWh**. "
                "Consumption is relatively low."
            )

        st.divider()

        # ----------------------------------------------------
        # ALERT SUMMARY
        # ----------------------------------------------------

        st.subheader("🚨 Alert Summary")

        high_count = len(
            history_df[
                history_df["Prediction (kWh)"] >= 8
            ]
        )

        moderate_count = len(
            history_df[
                (history_df["Prediction (kWh)"] >= 5)
                &
                (history_df["Prediction (kWh)"] < 8)
            ]
        )

        low_count = len(
            history_df[
                history_df["Prediction (kWh)"] < 5
            ]
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🔴 High",
                high_count
            )

        with col2:

            st.metric(
                "🟡 Moderate",
                moderate_count
            )

        with col3:

            st.metric(
                "🟢 Low",
                low_count
            )

        st.divider()

        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        st.subheader("💡 Recommendation")

        if latest_consumption >= 8:

            st.warning(
                "Consider reducing unnecessary electricity usage "
                "during this period. Check high-power appliances "
                "and reduce their usage if possible."
            )

        elif latest_consumption >= 5:

            st.info(
                "Electricity consumption is moderate. "
                "Continue monitoring usage during peak hours."
            )

        else:

            st.success(
                "Electricity consumption is currently low. "
                "Your predicted usage is within a lower range."
            )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "📋 Prediction History":

    st.title("📋 Prediction History")

    st.write(
        "View your previous electricity consumption predictions."
    )

    st.divider()

    if len(st.session_state.prediction_history) == 0:

        st.info(
            "No prediction history available yet."
        )

        st.write(
            "Go to 🤖 XGBoost Prediction and make a prediction "
            "to see it here."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.subheader("⚡ Consumption Overview")

        total_predictions = len(history_df)

        average_consumption = history_df[
            "Prediction (kWh)"
        ].mean()

        highest_consumption = history_df[
            "Prediction (kWh)"
        ].max()

        lowest_consumption = history_df[
            "Prediction (kWh)"
        ].min()

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Predictions",
                total_predictions
            )

        with col2:

            st.metric(
                "Average",
                f"{average_consumption:.2f} kWh"
            )

        with col3:

            st.metric(
                "Highest",
                f"{highest_consumption:.2f} kWh"
            )

        with col4:

            st.metric(
                "Lowest",
                f"{lowest_consumption:.2f} kWh"
            )

        st.divider()

        # ----------------------------------------------------
        # PREDICTION RECORDS
        # ----------------------------------------------------

        st.subheader("📋 Recent Predictions")

        display_df = history_df[
            [
                "Temperature",
                "Humidity",
                "Occupancy",
                "Hour",
                "Weekend",
                "Peak Hour",
                "Prediction (kWh)"
            ]
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # ----------------------------------------------------
        # CLEAR HISTORY
        # ----------------------------------------------------

        if st.button(
            "🗑️ Clear Prediction History",
            use_container_width=True
        ):

            st.session_state.prediction_history = []

            st.success(
                "Prediction history cleared successfully."
            )

            st.rerun()
