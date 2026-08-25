# ⚡ Electricity Consumption Prediction

An AI-powered web application that predicts electricity consumption using a trained **XGBoost Regression model**.

The application provides an interactive dashboard where users can enter environmental, occupancy, time, and calendar-related information to estimate electricity consumption.

🌐 **Live Demo:**  
https://electricity-consumption-prediction-ceuc92hvxnvdmnfjzmfmwk.streamlit.app/

---

## 📌 Project Overview

Electricity consumption can vary depending on several factors such as temperature, humidity, occupancy, time of day, and seasonal conditions.

This project uses **Machine Learning with XGBoost Regression** to estimate electricity consumption based on these input conditions.

The model is integrated into a **Streamlit web application** with an interactive dashboard for prediction, analysis, forecasting, and model insights.

---

## ✨ Features

### 🏠 Dashboard
- Clean and interactive electricity monitoring dashboard
- Overview of the prediction system
- Quick access to all major features

### 🤖 XGBoost Prediction
- Predict electricity consumption in kWh
- Accepts multiple environmental and time-based inputs
- Displays prediction results with consumption status
- Visualizes the predicted consumption

### 🔬 Energy Impact Simulator
- Compare two different energy-use scenarios
- Calculate the difference between scenarios
- Display percentage change
- Visualize current vs simulated consumption

### 📊 Analytics
- View prediction statistics
- Average consumption
- Highest consumption
- Lowest consumption
- Prediction trend visualization
- Recent prediction information

### 🔮 Forecast
- Generate upcoming electricity-consumption estimates
- Forecast multiple future hours
- View average, highest, and lowest predicted consumption
- Interactive forecast visualization

### 🧠 Model Insights
- XGBoost model information
- Regression task overview
- Feature importance visualization
- Explanation of how XGBoost works

### 🚨 Smart Alerts
- Detect low, moderate, and high predicted consumption
- Display current consumption status
- Count different consumption levels
- Provide simple monitoring recommendations

### 📋 Prediction History
- Store predictions during the current application session
- View previous prediction inputs
- Review prediction results
- Clear prediction history when required

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Joblib**
- **Plotly**
- **Matplotlib**

---

## 🤖 Machine Learning Model

The project uses **XGBoost Regression** for predicting electricity consumption.

### Input Features

The model uses the following features:

| Feature | Description |
|---|---|
| Temperature | Temperature in °C |
| Humidity | Humidity percentage |
| Occupancy | Occupancy percentage |
| Hour | Hour of the day |
| Day | Day of the month |
| Month | Month |
| Day of Week | Day represented from 0–6 |
| Weekend | Whether the day is a weekend |
| Season | Seasonal information |
| Peak Hour | Whether the time is a peak hour |

### Prediction Output

The model predicts:

**Electricity Consumption in kWh**

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Feature Transformation
   ↓
XGBoost Regression
   ↓
Model Training
   ↓
Model Saving
   ↓
Streamlit Application
   ↓
User Input
   ↓
Electricity Consumption Prediction
