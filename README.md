# ⚡ Electricity Consumption Prediction

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://electricity-consumption-prediction-ceuc92hvxnvdmnfjzmfmwk.streamlit.app/)

A machine learning web application that predicts electricity consumption based on environmental, occupancy, and time-related conditions.

The project uses **XGBoost** for prediction and **Streamlit** to provide an interactive web interface.

## 🌐 Live Demo

🚀 **[Open Electricity Consumption Prediction App](https://electricity-consumption-prediction-ceuc92hvxnvdmnfjzmfmwk.streamlit.app/)**

Try the live application to:

- 🤖 Predict electricity consumption
- 🔬 Simulate energy impact under different conditions
- 📊 Analyze prediction results
- 🔮 Generate future forecasts
- 🧠 Explore model insights
- 🚨 Monitor smart alerts
- 📋 View prediction history

---

## 📌 Project Overview

Electricity consumption can vary depending on factors such as temperature, humidity, occupancy, time of day, and seasonal conditions.

This project uses machine learning to estimate electricity consumption from these input conditions.

The application allows users to:

- Enter electricity consumption conditions
- Predict electricity consumption in kWh
- Compare current and simulated energy scenarios
- Analyze previous predictions
- View consumption statistics
- Generate future consumption forecasts
- Understand important model features
- Receive smart consumption alerts

---

## 🚀 Features

### 🤖 Electricity Consumption Prediction

Enter:

- Temperature
- Humidity
- Occupancy
- Hour
- Day
- Month
- Day of Week
- Weekend status
- Peak hour status
- Season

The trained XGBoost model generates the predicted electricity consumption in **kWh**.

---

### 🔬 Energy Impact Simulator

The Energy Impact Simulator allows users to compare two different electricity-use scenarios.

Users can enter:

- Current conditions
- Simulated conditions

The application then uses the trained XGBoost model to calculate:

- Current predicted consumption
- Simulated predicted consumption
- Difference in consumption
- Percentage change

The simulator also provides:

- 📊 Scenario comparison
- 💡 Energy impact analysis
- 📈 Current vs simulated consumption visualization
- ⚡ Impact summary
- 🧠 Explanation of what changed

The result is a **model-based scenario estimate**, not a guarantee of actual energy savings.

---

### 📊 Analytics

The Analytics section provides a simple overview of prediction results, including:

- Total predictions
- Average consumption
- Highest consumption
- Lowest consumption
- Latest prediction
- Latest input conditions
- Recent prediction records

---

### 🔮 Forecast

The Forecast section estimates electricity consumption for upcoming hours based on the selected conditions.

Users can choose:

- Starting hour
- Number of hours to forecast
- Temperature
- Humidity
- Occupancy
- Day
- Month
- Season
- Weekend status
- Peak hour status

The application provides:

- 📊 Average forecast consumption
- ⬆️ Highest predicted consumption
- ⬇️ Lowest predicted consumption
- 📋 Future consumption table
- 💡 Forecast analysis

---

### 🧠 Model Insights

The Model Insights section provides information about the XGBoost model and displays feature importance to help understand which input variables influence predictions.

It includes:

- 🤖 Model overview
- 📊 Feature importance
- 💡 Explanation of how XGBoost predicts electricity consumption

---

### 🚨 Smart Alerts

The Smart Alerts section monitors predicted consumption and categorizes it as:

- 🟢 Low Consumption
- 🟡 Moderate Consumption
- 🔴 High Consumption

It also provides:

- Alert summary
- Current consumption status
- Simple recommendations based on the predicted consumption level

---

### 📋 Prediction History

The Prediction History section allows users to view previous predictions made during the current application session.

It provides:

- Prediction count
- Average consumption
- Highest consumption
- Lowest consumption
- Recent prediction records
- Option to clear prediction history

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| XGBoost | Machine learning model |
| Scikit-learn | Data preprocessing |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Joblib | Model serialization |
| Matplotlib | Energy impact visualization |
| Streamlit | Web application |
| Git & GitHub | Version control |

---

## 🧠 Machine Learning Model

The project uses **XGBoost Regression** to predict electricity consumption.

### Input Features

The model uses:

```text
Temperature
Humidity
Occupancy
Hour
Day
Month
Day of Week
Weekend
Peak Hour
Season
