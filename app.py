import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(
    page_title="Electricity AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return joblib.load("xgboost_electricity_model.pkl")

@st.cache_resource
def load_preprocessor():
    return joblib.load("xgboost_preprocessor.pkl")

try:
    model = load_model()
    preprocessor = load_preprocessor()
except Exception as e:
    st.error("Unable to load the XGBoost model or preprocessor.")
    st.info("Make sure both .pkl files are in the same folder as app.py.")
    with st.expander("Technical details"):
        st.code(str(e))
    st.stop()

# -------------------- SESSION STATE --------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# -------------------- PROFESSIONAL UI --------------------
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(14,165,233,.10), transparent 28%),
        radial-gradient(circle at 90% 10%, rgba(59,130,246,.10), transparent 25%),
        linear-gradient(135deg,#07111f 0%,#0b1220 52%,#101d36 100%);
    color:#f8fafc;
}
.block-container {
    max-width:1500px;
    padding-top:1.5rem;
    padding-bottom:2rem;
}
section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#020617,#081426);
    border-right:1px solid rgba(148,163,184,.14);
}
section[data-testid="stSidebar"] * {
    color:#e2e8f0;
}
h1,h2,h3,h4 {
    color:#f8fafc !important;
}
.small-muted {
    color:#94a3b8;
    font-size:15px;
}
.hero {
    padding:28px 30px;
    border:1px solid rgba(56,189,248,.18);
    border-radius:24px;
    background:linear-gradient(135deg,rgba(14,165,233,.12),rgba(30,41,59,.40));
    box-shadow:0 18px 50px rgba(0,0,0,.18);
    margin-bottom:22px;
}
.hero-title {
    font-size:38px;
    font-weight:800;
    letter-spacing:-1px;
    color:#f8fafc;
}
.hero-accent {
    color:#38bdf8;
}
.hero-text {
    color:#94a3b8;
    font-size:16px;
    margin-top:8px;
}
.card {
    background:rgba(15,23,42,.70);
    border:1px solid rgba(148,163,184,.12);
    border-radius:18px;
    padding:20px;
    min-height:130px;
    box-shadow:0 10px 35px rgba(0,0,0,.16);
}
.card-icon {
    font-size:27px;
    margin-bottom:10px;
}
.card-label {
    color:#94a3b8;
    font-size:13px;
}
.card-value {
    color:#f8fafc;
    font-size:25px;
    font-weight:750;
    margin-top:4px;
}
.result-card {
    background:linear-gradient(135deg,rgba(14,165,233,.16),rgba(37,99,235,.10));
    border:1px solid rgba(56,189,248,.30);
    border-radius:24px;
    padding:30px;
    text-align:center;
    margin:20px 0;
}
.result-label {
    color:#94a3b8;
    font-size:15px;
}
.result-number {
    color:#38bdf8;
    font-size:48px;
    font-weight:850;
    margin:6px 0;
}
.result-status {
    color:#cbd5e1;
    font-size:18px;
}
.section-card {
    background:rgba(15,23,42,.62);
    border:1px solid rgba(148,163,184,.10);
    border-radius:18px;
    padding:22px;
    margin-bottom:18px;
}
.stButton > button {
    width:100%;
    min-height:46px;
    border-radius:12px;
    font-weight:700;
}
[data-testid="stMetric"] {
    background:rgba(15,23,42,.68);
    border:1px solid rgba(148,163,184,.11);
    border-radius:15px;
    padding:15px;
}
[data-testid="stDataFrame"] {
    border-radius:14px;
    overflow:hidden;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HELPERS --------------------
FEATURE_NAMES = [
    "Temperature", "Humidity", "Occupancy", "Hour", "Day",
    "Month", "Day of Week", "Weekend", "Season", "Peak Hour"
]

def predict_consumption(
    temperature, humidity, occupancy, hour, day, month,
    day_of_week, is_weekend, season, is_peak_hour
):
    input_data = pd.DataFrame({
        "temperature_c": [temperature],
        "humidity_percent": [humidity],
        "occupancy_percent": [occupancy],
        "hour": [hour],
        "day": [day],
        "month": [month],
        "day_of_week": [day_of_week],
        "is_weekend": [1 if is_weekend == "Yes" else 0],
        "season": [season],
        "is_peak_hour": [1 if is_peak_hour == "Yes" else 0],
    })
    processed = preprocessor.transform(input_data)
    return float(model.predict(processed)[0])

def status_for(value):
    if value < 5:
        return "🟢 Low Consumption"
    if value < 8:
        return "🟡 Moderate Consumption"
    return "🔴 High Consumption"

def add_history(values, prediction):
    st.session_state.prediction_history.append({
        "Temperature": values[0],
        "Humidity": values[1],
        "Occupancy": values[2],
        "Hour": values[3],
        "Day": values[4],
        "Month": values[5],
        "Day of Week": values[6],
        "Weekend": values[7],
        "Peak Hour": values[9],
        "Season": values[8],
        "Prediction (kWh)": round(prediction, 2),
    })

def section_title(title, subtitle=None):
    st.subheader(title)
    if subtitle:
        st.markdown(f'<div class="small-muted">{subtitle}</div>', unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:12px 0 18px;">'
        '<div style="font-size:46px;">⚡</div>'
        '<div style="font-size:24px;font-weight:800;color:#38bdf8;">Electricity AI</div>'
        '<div style="color:#94a3b8;font-size:13px;margin-top:5px;">Consumption Prediction</div>'
        '</div>',
        unsafe_allow_html=True
    )
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
            "📋 Prediction History",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("AI-powered electricity monitoring")

# -------------------- DASHBOARD --------------------
if page == "🏠 Dashboard":
    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">⚡ Electricity <span class="hero-accent">AI</span></div>'
        '<div class="hero-text">Smart electricity consumption prediction powered by XGBoost regression.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    history = st.session_state.prediction_history
    latest = history[-1]["Prediction (kWh)"] if history else None

    cols = st.columns(4)
    cards = [
        ("🤖", "Prediction Model", "XGBoost"),
        ("⚡", "Latest Prediction", f"{latest:.2f} kWh" if latest is not None else "Ready"),
        ("📊", "Predictions Made", str(len(history))),
        ("🚨", "Monitoring", "Active"),
    ]
    for col, (icon, label, value) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="card"><div class="card-icon">{icon}</div>'
                f'<div class="card-label">{label}</div>'
                f'<div class="card-value">{value}</div></div>',
                unsafe_allow_html=True
            )

    st.write("")
    left, right = st.columns([1.35, 1])

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_title("🚀 Quick Start", "Make a prediction in three simple steps.")
        st.markdown(
            "1. Open **XGBoost Prediction** from the sidebar.<br>"
            "2. Enter temperature, humidity, occupancy and time conditions.<br>"
            "3. Select **Predict Electricity Consumption** to see the estimate.",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_title("🧠 What the system uses")
        st.markdown(
            "Environmental conditions, occupancy, time and calendar-related features "
            "are prepared through the saved preprocessor and passed to the trained XGBoost model.",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    section_title("📌 System Overview")
    a, b, c = st.columns(3)
    with a:
        st.info("**Regression**\n\nThe model predicts a continuous electricity-consumption value.")
    with b:
        st.info("**XGBoost**\n\nThe trained model learns relationships among structured input features.")
    with c:
        st.info("**Streamlit**\n\nThe model is presented through an interactive browser application.")

# -------------------- PREDICTION --------------------
elif page == "🤖 XGBoost Prediction":
    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">🤖 XGBoost <span class="hero-accent">Prediction</span></div>'
        '<div class="hero-text">Enter the conditions and generate an electricity-consumption estimate.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    section_title("📝 Input Conditions")
    c1, c2 = st.columns(2)

    with c1:
        temperature = st.number_input("🌡️ Temperature (°C)", value=25.0)
        humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, 60.0)
        occupancy = st.number_input("🏠 Occupancy (%)", 0.0, 100.0, 50.0)
        hour = st.number_input("🕐 Hour", 0, 23, 12)
        day = st.number_input("📅 Day", 1, 31, 1)

    with c2:
        month = st.number_input("📆 Month", 1, 12, 1)
        day_of_week = st.number_input("📅 Day of Week (0 = Monday, 6 = Sunday)", 0, 6, 3)
        is_weekend = st.selectbox("📅 Is Weekend?", ["No", "Yes"])
        is_peak_hour = st.selectbox("⏰ Is Peak Hour?", ["No", "Yes"])
        season = st.selectbox("🌤️ Season", ["Winter", "Spring", "Summer", "Autumn"])

    st.write("")
    if st.button("⚡ Predict Electricity Consumption", type="primary"):
        try:
            prediction = predict_consumption(
                temperature, humidity, occupancy, hour, day, month,
                day_of_week, is_weekend, season, is_peak_hour
            )
            add_history(
                (temperature, humidity, occupancy, hour, day, month,
                 day_of_week, is_weekend, season, is_peak_hour),
                prediction
            )

            st.success("Prediction completed successfully.")
            st.markdown(
                f'<div class="result-card">'
                f'<div class="result-label">Predicted Electricity Consumption</div>'
                f'<div class="result-number">{prediction:.2f} kWh</div>'
                f'<div class="result-status">{status_for(prediction)}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            fig, ax = plt.subplots(figsize=(8, 3.8))
            ax.bar(["Predicted Consumption"], [prediction])
            ax.set_ylabel("Consumption (kWh)")
            ax.set_title("Current Prediction")
            ax.grid(axis="y", alpha=.20)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error("Prediction could not be completed.")
            with st.expander("Technical details"):
                st.code(str(e))

# -------------------- SIMULATOR --------------------
elif page == "🔬 Energy Impact Simulator":
    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">🔬 Energy Impact <span class="hero-accent">Simulator</span></div>'
        '<div class="hero-text">Compare two conditions using the trained XGBoost model.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    section_title("📌 Current Scenario")
    c1, c2 = st.columns(2)
    with c1:
        ct = st.number_input("🌡️ Current Temperature (°C)", value=25.0, key="ct")
        ch = st.number_input("💧 Current Humidity (%)", 0.0, 100.0, 60.0, key="ch")
        co = st.number_input("🏠 Current Occupancy (%)", 0.0, 100.0, 50.0, key="co")
        chr_ = st.number_input("🕐 Current Hour", 0, 23, 12, key="chr")
        cd = st.number_input("📅 Current Day", 1, 31, 1, key="cd")
    with c2:
        cm = st.number_input("📆 Current Month", 1, 12, 1, key="cm")
        cdw = st.number_input("📅 Current Day of Week", 0, 6, 3, key="cdw")
        cw = st.selectbox("Current Weekend?", ["No", "Yes"], key="cw")
        cp = st.selectbox("Current Peak Hour?", ["No", "Yes"], key="cp")
        cs = st.selectbox("🌤️ Current Season", ["Winter", "Spring", "Summer", "Autumn"], key="cs")

    st.divider()
    section_title("🔬 Simulated Scenario")
    c1, c2 = st.columns(2)
    with c1:
        stt = st.number_input("🌡️ Scenario Temperature (°C)", value=25.0, key="stt")
        sth = st.number_input("💧 Scenario Humidity (%)", 0.0, 100.0, 60.0, key="sth")
        sto = st.number_input("🏠 Scenario Occupancy (%)", 0.0, 100.0, 80.0, key="sto")
        st_hour = st.number_input("🕐 Scenario Hour", 0, 23, 18, key="sthour")
        std = st.number_input("📅 Scenario Day", 1, 31, 1, key="std")
    with c2:
        stm = st.number_input("📆 Scenario Month", 1, 12, 1, key="stm")
        stdw = st.number_input("📅 Scenario Day of Week", 0, 6, 3, key="stdw")
        stw = st.selectbox("Scenario Weekend?", ["No", "Yes"], key="stw")
        stp = st.selectbox("Scenario Peak Hour?", ["No", "Yes"], key="stp")
        sts = st.selectbox("🌤️ Scenario Season", ["Winter", "Spring", "Summer", "Autumn"], key="sts")

    if st.button("🔬 Simulate Energy Impact", type="primary"):
        try:
            current = predict_consumption(ct, ch, co, chr_, cd, cm, cdw, cw, cs, cp)
            simulated = predict_consumption(stt, sth, sto, st_hour, std, stm, stdw, stw, sts, stp)
            diff = simulated - current
            pct = (diff / current * 100) if current else 0

            st.success("Simulation completed successfully.")
            a, b, c = st.columns(3)
            a.metric("Current", f"{current:.2f} kWh")
            b.metric("Simulated", f"{simulated:.2f} kWh")
            c.metric("Impact", f"{diff:+.2f} kWh", delta=f"{pct:+.1f}%")

            if diff > 0:
                st.warning(f"The simulated conditions increase predicted consumption by {diff:.2f} kWh.")
            elif diff < 0:
                st.success(f"The simulated conditions reduce predicted consumption by {abs(diff):.2f} kWh.")
            else:
                st.info("Both scenarios produce the same prediction.")

            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(["Current", "Simulated"], [current, simulated])
            ax.set_ylabel("Consumption (kWh)")
            ax.set_title("Current vs Simulated Consumption")
            ax.grid(axis="y", alpha=.20)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for bar, value in zip(bars, [current, simulated]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                        f"{value:.2f}", ha="center", va="bottom", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error("Simulation could not be completed.")
            with st.expander("Technical details"):
                st.code(str(e))

# -------------------- ANALYTICS --------------------
elif page == "📊 Analytics":
    st.markdown(
        '<div class="hero"><div class="hero-title">📊 Prediction <span class="hero-accent">Analytics</span></div>'
        '<div class="hero-text">Understand the predictions generated during this session.</div></div>',
        unsafe_allow_html=True
    )
    if not st.session_state.prediction_history:
        st.info("No prediction data yet. Make a prediction first.")
    else:
        df = pd.DataFrame(st.session_state.prediction_history)
        vals = df["Prediction (kWh)"]
        a,b,c,d = st.columns(4)
        a.metric("Predictions", len(df))
        b.metric("Average", f"{vals.mean():.2f} kWh")
        c.metric("Highest", f"{vals.max():.2f} kWh")
        d.metric("Lowest", f"{vals.min():.2f} kWh")

        st.divider()
        section_title("📈 Consumption Trend")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(1, len(df)+1), vals, marker="o")
        ax.set_xlabel("Prediction Number")
        ax.set_ylabel("Consumption (kWh)")
        ax.grid(alpha=.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.divider()
        section_title("📋 Prediction Data")
        st.dataframe(df, use_container_width=True, hide_index=True)

# -------------------- FORECAST --------------------
elif page == "🔮 Forecast":
    st.markdown(
        '<div class="hero"><div class="hero-title">🔮 Electricity <span class="hero-accent">Forecast</span></div>'
        '<div class="hero-text">Generate a model-based estimate for upcoming hours.</div></div>',
        unsafe_allow_html=True
    )
    c1,c2 = st.columns(2)
    with c1:
        ft = st.number_input("🌡️ Temperature (°C)", value=25.0, key="ft")
        fh = st.number_input("💧 Humidity (%)", 0.0, 100.0, 60.0, key="fh")
        fo = st.number_input("🏠 Occupancy (%)", 0.0, 100.0, 50.0, key="fo")
        start = st.slider("🕐 Starting Hour", 0, 23, 12)
    with c2:
        hours = st.slider("⏱️ Hours to Forecast", 1, 12, 6)
        fd = st.number_input("📅 Day", 1, 31, 15, key="fd")
        fm = st.number_input("📆 Month", 1, 12, 6, key="fm")
        fs = st.selectbox("🌤️ Season", ["Winter", "Spring", "Summer", "Autumn"], key="fs")
    w,p = st.columns(2)
    weekend = w.selectbox("Weekend?", ["No","Yes"], key="fw")
    peak = p.selectbox("Peak Hour?", ["No","Yes"], key="fp")

    if st.button("🔮 Generate Forecast", type="primary"):
        try:
            results = []
            for i in range(hours):
                h = (start+i) % 24
                pred = predict_consumption(ft, fh, fo, h, fd, fm, 0, weekend, fs, peak)
                results.append({"Hour": f"{h:02d}:00", "Predicted Consumption (kWh)": round(pred,2)})
            fdf = pd.DataFrame(results)
            st.success("Forecast generated successfully.")
            vals = fdf["Predicted Consumption (kWh)"]
            a,b,c = st.columns(3)
            a.metric("Average", f"{vals.mean():.2f} kWh")
            b.metric("Highest", f"{vals.max():.2f} kWh")
            c.metric("Lowest", f"{vals.min():.2f} kWh")

            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(fdf["Hour"], vals, marker="o")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Consumption (kWh)")
            ax.grid(alpha=.20)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.dataframe(fdf, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error("Forecast could not be generated.")
            with st.expander("Technical details"):
                st.code(str(e))

# -------------------- MODEL INSIGHTS --------------------
elif page == "🧠 Model Insights":
    st.markdown(
        '<div class="hero"><div class="hero-title">🧠 Model <span class="hero-accent">Insights</span></div>'
        '<div class="hero-text">Explore the model type and feature importance.</div></div>',
        unsafe_allow_html=True
    )
    a,b,c = st.columns(3)
    a.metric("Model", "XGBoost")
    b.metric("Task", "Regression")
    c.metric("Output", "Electricity Consumption")

    st.divider()
    section_title("📊 Feature Importance")
    try:
        importance = model.feature_importances_
        if len(importance) == len(FEATURE_NAMES):
            imp = pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": importance}).sort_values("Importance")
            fig, ax = plt.subplots(figsize=(9,5))
            ax.barh(imp["Feature"], imp["Importance"])
            ax.set_xlabel("Importance")
            ax.set_title("XGBoost Feature Importance")
            ax.grid(axis="x", alpha=.20)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.dataframe(imp.sort_values("Importance", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.warning("The saved model's feature count does not match the displayed feature names.")
    except Exception as e:
        st.warning("Feature importance is not available for this saved model.")

    st.divider()
    section_title("💡 How XGBoost Predicts")
    st.markdown(
        '<div class="section-card">XGBoost is a tree-based machine-learning algorithm that builds multiple decision trees sequentially. '
        'For this project, the trained regression model uses environmental, occupancy, time and calendar-related inputs '
        'to estimate electricity consumption.</div>',
        unsafe_allow_html=True
    )

# -------------------- ALERTS --------------------
elif page == "🚨 Smart Alerts":
    st.markdown(
        '<div class="hero"><div class="hero-title">🚨 Smart <span class="hero-accent">Alerts</span></div>'
        '<div class="hero-text">Monitor the latest predicted consumption level.</div></div>',
        unsafe_allow_html=True
    )
    if not st.session_state.prediction_history:
        st.info("No prediction data yet. Make a prediction first.")
    else:
        df = pd.DataFrame(st.session_state.prediction_history)
        latest = float(df.iloc[-1]["Prediction (kWh)"])
        section_title("⚡ Current Consumption Status")
        if latest >= 8:
            st.error(f"🔴 HIGH CONSUMPTION — {latest:.2f} kWh")
        elif latest >= 5:
            st.warning(f"🟡 MODERATE CONSUMPTION — {latest:.2f} kWh")
        else:
            st.success(f"🟢 LOW CONSUMPTION — {latest:.2f} kWh")

        high = int((df["Prediction (kWh)"] >= 8).sum())
        moderate = int(((df["Prediction (kWh)"] >= 5) & (df["Prediction (kWh)"] < 8)).sum())
        low = int((df["Prediction (kWh)"] < 5).sum())
        a,b,c = st.columns(3)
        a.metric("🔴 High", high)
        b.metric("🟡 Moderate", moderate)
        c.metric("🟢 Low", low)

        st.divider()
        if latest >= 8:
            st.warning("Consider reducing unnecessary electricity usage and monitoring high-power appliances.")
        elif latest >= 5:
            st.info("Consumption is moderate. Continue monitoring usage, especially during peak hours.")
        else:
            st.success("The latest predicted consumption is relatively low.")

# -------------------- HISTORY --------------------
elif page == "📋 Prediction History":
    st.markdown(
        '<div class="hero"><div class="hero-title">📋 Prediction <span class="hero-accent">History</span></div>'
        '<div class="hero-text">Review predictions generated during the current application session.</div></div>',
        unsafe_allow_html=True
    )
    if not st.session_state.prediction_history:
        st.info("No prediction history available yet.")
    else:
        df = pd.DataFrame(st.session_state.prediction_history)
        vals = df["Prediction (kWh)"]
        a,b,c,d = st.columns(4)
        a.metric("Predictions", len(df))
        b.metric("Average", f"{vals.mean():.2f} kWh")
        c.metric("Highest", f"{vals.max():.2f} kWh")
        d.metric("Lowest", f"{vals.min():.2f} kWh")

        st.divider()
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.write("")
        if st.button("🗑️ Clear Prediction History"):
            st.session_state.prediction_history = []
            st.rerun()
