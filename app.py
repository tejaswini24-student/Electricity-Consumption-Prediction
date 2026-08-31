import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime
from streamlit_option_menu import option_menu

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
# The alert threshold is user-adjustable on the Smart Alerts page, but we
# need a default before that page has ever been visited, so any page can
# check "did this prediction cross the alert line" from the moment the
# app starts.
if "alert_threshold" not in st.session_state:
    st.session_state.alert_threshold = 8.0

# -------------------- THEME DEFINITIONS --------------------
# Kept as a dict (rather than hardcoding hex codes throughout the file)
# even with only one theme now, so every color the app uses still comes
# from one place - the CSS block below just reads THEMES["dark"][...].
THEMES = {
    "dark": dict(
        bg=(
            "radial-gradient(circle at 10% 0%, rgba(14,165,233,.10), transparent 28%),"
            "radial-gradient(circle at 90% 10%, rgba(59,130,246,.10), transparent 25%),"
            "linear-gradient(135deg,#07111f 0%,#0b1220 52%,#101d36 100%)"
        ),
        text="#f8fafc", muted="#94a3b8",
        sidebar_bg="linear-gradient(180deg,#020617,#081426)", sidebar_border="rgba(148,163,184,.14)",
        card_bg="rgba(15,23,42,.70)", card_border="rgba(148,163,184,.12)", card_shadow="0 10px 35px rgba(0,0,0,.16)",
        hero_bg="linear-gradient(135deg,rgba(14,165,233,.12),rgba(30,41,59,.40))",
        hero_border="rgba(56,189,248,.18)", hero_shadow="0 18px 50px rgba(0,0,0,.18)",
        section_bg="rgba(15,23,42,.62)", section_border="rgba(148,163,184,.10)",
        empty_bg="rgba(15,23,42,.55)", empty_border="rgba(148,163,184,.25)",
        metric_bg="rgba(15,23,42,.68)", metric_border="rgba(148,163,184,.11)",
        accent="#38bdf8", accent2="#2563eb",
        chart_text="#e2e8f0", chart_grid="rgba(148,163,184,.15)",
        tooltip_bg="#0f172a", tooltip_text="#f8fafc",
    ),
}

STATUS_STYLES = {
    "dark": {
        "default":  dict(bg="linear-gradient(135deg,rgba(14,165,233,.16),rgba(37,99,235,.10))", border="rgba(56,189,248,.30)", number="#38bdf8"),
        "low":      dict(bg="linear-gradient(135deg,rgba(34,197,94,.16),rgba(21,128,61,.10))",  border="rgba(34,197,94,.35)",  number="#22c55e"),
        "moderate": dict(bg="linear-gradient(135deg,rgba(234,179,8,.16),rgba(180,83,9,.10))",   border="rgba(234,179,8,.35)",  number="#eab308"),
        "high":     dict(bg="linear-gradient(135deg,rgba(239,68,68,.16),rgba(153,27,27,.10))",  border="rgba(239,68,68,.35)",  number="#ef4444"),
    },
}

# -------------------- SIDEBAR --------------------
PAGES = [
    "Dashboard", "XGBoost Prediction", "Energy Impact Simulator",
    "Bill Estimator", "Analytics", "Forecast", "Model Insights",
    "Smart Alerts", "Prediction History",
]
PAGE_ICONS = [
    "house", "cpu", "shuffle", "cash-coin", "bar-chart-line",
    "cloud-sun", "diagram-3", "exclamation-triangle", "clock-history",
]

with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:12px 0 10px;">'
        '<div style="font-size:46px;">⚡</div>'
        '<div style="font-size:24px;font-weight:800;color:#38bdf8;">Electricity AI</div>'
        '<div style="color:#94a3b8;font-size:13px;margin-top:5px;">Consumption Prediction</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.divider()
    page = option_menu(
        menu_title=None,
        options=PAGES,
        icons=PAGE_ICONS,
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": "#38bdf8", "font-size": "16px"},
            "nav-link": {
                "font-size": "14px", "text-align": "left", "margin": "3px 0",
                "border-radius": "10px", "padding": "10px 12px",
            },
            "nav-link-selected": {
                "background-color": "rgba(56,189,248,.16)",
                "color": "#38bdf8", "font-weight": "700",
            },
        },
    )
    st.divider()
    st.caption("AI-powered electricity monitoring")

# The light/dark toggle was removed - it caused plain Streamlit text
# (which doesn't go through our custom CSS classes) to render invisible
# in light mode. The app is fixed back to the single dark theme that
# was working correctly.
t = THEMES["dark"]
s = STATUS_STYLES["dark"]

# These four are read by every Plotly chart further down via style_fig(),
# so switching theme_key above automatically re-colors every chart too.
COLOR_BG = "rgba(0,0,0,0)"
COLOR_TEXT = t["chart_text"]
COLOR_GRID = t["chart_grid"]
COLOR_ACCENT = t["accent"]
COLOR_ACCENT2 = t["accent2"]

# -------------------- PROFESSIONAL UI --------------------
st.markdown(f"""
<style>
.stApp {{
    background: {t['bg']};
    color:{t['text']};
}}
.block-container {{
    max-width:1500px;
    padding-top:3.5rem;
    padding-bottom:2rem;
}}
/* Streamlit's built-in toolbar (hamburger/rerun icons) floats over the
   page at a fixed position. Giving it a solid background matching the
   theme (instead of the default transparent one) stops content from
   looking like it's "poking through" behind it as you scroll. */
header[data-testid="stHeader"] {{
    background:{t['bg']};
}}
section[data-testid="stSidebar"] {{
    background:{t['sidebar_bg']};
    border-right:1px solid {t['sidebar_border']};
}}
section[data-testid="stSidebar"] * {{
    color:{t['text']};
}}
h1,h2,h3,h4 {{
    color:{t['text']} !important;
}}
.small-muted {{
    color:{t['muted']};
    font-size:15px;
}}
.hero {{
    padding:28px 30px;
    border:1px solid {t['hero_border']};
    border-radius:24px;
    background:{t['hero_bg']};
    box-shadow:{t['hero_shadow']};
    margin-bottom:22px;
}}
.hero-title {{
    font-size:38px;
    font-weight:800;
    letter-spacing:-1px;
    color:{t['text']};
}}
.hero-accent {{
    color:{t['accent']};
}}
.hero-text {{
    color:{t['muted']};
    font-size:16px;
    margin-top:8px;
}}
.card {{
    background:{t['card_bg']};
    border:1px solid {t['card_border']};
    border-radius:18px;
    padding:20px;
    min-height:130px;
    box-shadow:{t['card_shadow']};
}}
.card-icon {{
    font-size:27px;
    margin-bottom:10px;
}}
.card-label {{
    color:{t['muted']};
    font-size:13px;
}}
.card-value {{
    color:{t['text']};
    font-size:25px;
    font-weight:750;
    margin-top:4px;
}}
.card-trend {{
    font-size:13px;
    margin-top:6px;
    font-weight:600;
}}
.card-trend.up {{ color:{s['high']['number']}; }}
.card-trend.down {{ color:{s['low']['number']}; }}
.card-trend.flat {{ color:{t['muted']}; }}
.result-card {{
    border-radius:24px;
    padding:30px;
    text-align:center;
    margin:20px 0;
    border:1px solid {s['default']['border']};
    background:{s['default']['bg']};
}}
.result-card.low {{ border-color:{s['low']['border']}; background:{s['low']['bg']}; }}
.result-card.moderate {{ border-color:{s['moderate']['border']}; background:{s['moderate']['bg']}; }}
.result-card.high {{ border-color:{s['high']['border']}; background:{s['high']['bg']}; }}
.result-label {{
    color:{t['muted']};
    font-size:15px;
}}
.result-number {{
    font-size:48px;
    font-weight:850;
    margin:6px 0;
    color:{s['default']['number']};
}}
.result-card.low .result-number {{ color:{s['low']['number']}; }}
.result-card.moderate .result-number {{ color:{s['moderate']['number']}; }}
.result-card.high .result-number {{ color:{s['high']['number']}; }}
.result-status {{
    color:{t['muted']};
    font-size:18px;
}}
.section-card {{
    background:{t['section_bg']};
    border:1px solid {t['section_border']};
    border-radius:18px;
    padding:22px;
    margin-bottom:18px;
}}
.section-card-title {{
    font-size:20px;
    font-weight:700;
    color:{t['text']};
    margin-bottom:6px;
}}
.section-card-body {{
    color:{t['text']};
    font-size:15px;
    line-height:1.6;
}}
.empty-state {{
    text-align:center;
    padding:46px 20px;
    background:{t['empty_bg']};
    border:1px dashed {t['empty_border']};
    border-radius:18px;
}}
.empty-state .icon {{ font-size:40px; margin-bottom:10px; }}
.empty-state .title {{ font-size:17px; font-weight:700; color:{t['text']}; }}
.empty-state .subtitle {{ font-size:14px; color:{t['muted']}; margin-top:4px; }}
.stButton > button {{
    width:100%;
    min-height:46px;
    border-radius:12px;
    font-weight:700;
}}
[data-testid="stMetric"] {{
    background:{t['metric_bg']};
    border:1px solid {t['metric_border']};
    border-radius:15px;
    padding:15px;
}}
[data-testid="stDataFrame"] {{
    border-radius:14px;
    overflow:hidden;
}}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {{
    border-radius:10px;
}}
</style>
""", unsafe_allow_html=True)

# -------------------- HELPERS --------------------
FEATURE_NAMES = [
    "Temperature", "Humidity", "Occupancy", "Hour", "Day",
    "Month", "Day of Week", "Weekend", "Season", "Peak Hour"
]

SEASONS = ["Winter", "Spring", "Summer", "Autumn"]

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
    """Returns a (label, css_class) pair so both the emoji text and the
    result-card color are always derived from the same thresholds -
    change the numbers once here and every page updates consistently."""
    if value < 5:
        return "🟢 Low Consumption", "low"
    if value < 8:
        return "🟡 Moderate Consumption", "moderate"
    return "🔴 High Consumption", "high"

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

def get_recommendations(values, prediction, status_label):
    """Rule-based (not model-based) energy-saving tips. Each 'if' checks
    one input the user just entered, so the tips are specific to what
    they described rather than generic advice - e.g. a peak-hour flag
    always adds the peak-shifting tip, regardless of how high or low the
    predicted number turned out to be."""
    tips = []
    if values.get("is_peak_hour") == "Yes":
        tips.append("This falls in a peak hour. Shifting flexible loads (laundry, charging, dishwashing) to off-peak hours can lower both consumption and cost.")
    if values.get("occupancy", 0) >= 70:
        tips.append("Occupancy is high. Make sure unused rooms have lights, fans, and standby electronics switched off.")
    if values.get("temperature", 0) >= 30 and values.get("season") in ("Summer",):
        tips.append("Temperature is high for summer conditions. Setting AC to 24-26°C instead of lower uses noticeably less energy per hour.")
    if values.get("is_weekend") == "Yes":
        tips.append("Weekend usage patterns differ from weekdays - consider scheduling high-load appliances for off-peak weekend hours too.")
    if "High" in status_label:
        tips.append("Predicted consumption is in the high range. Check for appliances left running unnecessarily (water heaters, old refrigerators, always-on chargers).")
    elif "Moderate" in status_label:
        tips.append("Consumption is moderate. Small changes - LED lighting, unplugging idle electronics - can shift this toward the low range.")
    else:
        tips.append("Consumption is already in the low range. Maintaining current usage habits should keep costs down.")
    return tips

def build_report_text(values, prediction, status_label, bill_lines=None):
    """Builds a plain-text report as one string, which st.download_button
    can hand to the browser as a file. Plain text (not PDF) keeps this
    dependency-free - no extra library needed just to produce a report."""
    lines = [
        "SMARTENERGY AI - PREDICTION REPORT",
        "=" * 40,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "INPUT CONDITIONS",
        "-" * 40,
        f"Temperature       : {values.get('temperature')} C",
        f"Humidity          : {values.get('humidity')} %",
        f"Occupancy         : {values.get('occupancy')} %",
        f"Hour              : {values.get('hour')}",
        f"Day / Month       : {values.get('day')} / {values.get('month')}",
        f"Day of Week       : {values.get('day_of_week')} (0=Mon, 6=Sun)",
        f"Weekend           : {values.get('is_weekend')}",
        f"Peak Hour         : {values.get('is_peak_hour')}",
        f"Season            : {values.get('season')}",
        "",
        "PREDICTION",
        "-" * 40,
        f"Predicted Consumption : {prediction:.2f} kWh",
        f"Status                : {status_label}",
    ]
    if bill_lines:
        lines += ["", "BILL ESTIMATE", "-" * 40] + bill_lines
    lines += ["", "ENERGY-SAVING RECOMMENDATIONS", "-" * 40]
    lines += [f"- {tip}" for tip in get_recommendations(values, prediction, status_label)]
    return "\n".join(lines)

def section_title(title, subtitle=None):
    st.subheader(title)
    if subtitle:
        st.markdown(f'<div class="small-muted">{subtitle}</div>', unsafe_allow_html=True)

def empty_state(icon, title, subtitle):
    st.markdown(
        f'<div class="empty-state"><div class="icon">{icon}</div>'
        f'<div class="title">{title}</div>'
        f'<div class="subtitle">{subtitle}</div></div>',
        unsafe_allow_html=True
    )

def style_fig(fig, y_title="Consumption (kWh)", x_title=None):
    """Applies one consistent theme to every Plotly chart. Because this
    reads the module-level COLOR_* variables (which are set from the
    THEMES dict based on the sidebar toggle), charts automatically
    switch between light and dark styling along with the rest of the app."""
    fig.update_layout(
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT),
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(bgcolor=t["tooltip_bg"], font_color=t["tooltip_text"]),
    )
    fig.update_xaxes(showgrid=False, title=x_title, color=COLOR_TEXT)
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID, title=y_title, color=COLOR_TEXT)
    return fig

def remembered_defaults(store_key, fallback=None):
    """Streamlit deletes a widget's stored value the moment it stops
    being drawn on a run - which happens here every time you switch
    pages, since only the current page's widgets are created. That
    resets inputs to their hardcoded defaults on return. This function
    reads a plain (non-widget) copy of the last values entered, stored
    separately in session_state so page switches can't wipe it.

    store_key groups which forms share state: Prediction, the
    Simulator's Current Scenario, and Forecast all pass "shared" so a
    value typed on any one of them shows up on the others too - they
    represent the same real "current conditions". The Simulator's
    Simulated Scenario uses its own "sim" key instead, since it's meant
    to be a different hypothetical, not a copy of the current one."""
    return st.session_state.get(f"{store_key}_remembered", fallback or {})

def remember_values(store_key, values):
    """Call this right after render_condition_inputs() so whatever the
    user currently has entered is saved under store_key and reappears
    - on this form and on any other form using the same store_key -
    even after switching pages."""
    st.session_state[f"{store_key}_remembered"] = values

def render_condition_inputs(key_prefix, defaults=None, include_hour=True):
    """One shared input form used by the Prediction, Simulator and
    Forecast pages instead of copy-pasting the same 8-10 widgets three
    times. Each call needs a unique key_prefix (e.g. "cur", "sim") so
    Streamlit doesn't raise a duplicate-key error when the same form is
    drawn twice on one page. Returns a dict you can pass straight into
    predict_consumption(**values)."""
    d = defaults or {}
    c1, c2 = st.columns(2)
    with c1:
        temperature = st.number_input(
            "🌡️ Temperature (°C)", value=d.get("temperature", 25.0), key=f"{key_prefix}_temp"
        )
        humidity = st.number_input(
            "💧 Humidity (%)", 0.0, 100.0, d.get("humidity", 60.0), key=f"{key_prefix}_hum"
        )
        occupancy = st.number_input(
            "🏠 Occupancy (%)", 0.0, 100.0, d.get("occupancy", 50.0), key=f"{key_prefix}_occ"
        )
        hour = None
        if include_hour:
            hour = st.number_input(
                "🕐 Hour", 0, 23, d.get("hour", 12), key=f"{key_prefix}_hour"
            )
        day = st.number_input("📅 Day", 1, 31, d.get("day", 1), key=f"{key_prefix}_day")
    with c2:
        month = st.number_input("📆 Month", 1, 12, d.get("month", 1), key=f"{key_prefix}_month")
        day_of_week = st.number_input(
            "📅 Day of Week (0 = Monday, 6 = Sunday)", 0, 6, d.get("day_of_week", 3),
            key=f"{key_prefix}_dow"
        )
        is_weekend = st.selectbox(
            "📅 Is Weekend?", ["No", "Yes"],
            index=["No", "Yes"].index(d.get("is_weekend", "No")), key=f"{key_prefix}_weekend"
        )
        is_peak_hour = st.selectbox(
            "⏰ Is Peak Hour?", ["No", "Yes"],
            index=["No", "Yes"].index(d.get("is_peak_hour", "No")), key=f"{key_prefix}_peak"
        )
        season = st.selectbox(
            "🌤️ Season", SEASONS, index=SEASONS.index(d.get("season", "Winter")),
            key=f"{key_prefix}_season"
        )
    values = dict(
        temperature=temperature, humidity=humidity, occupancy=occupancy,
        day=day, month=month, day_of_week=day_of_week,
        is_weekend=is_weekend, season=season, is_peak_hour=is_peak_hour,
    )
    if include_hour:
        values["hour"] = hour
    return values

# -------------------- DASHBOARD --------------------
if page == "Dashboard":
    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">⚡ Electricity <span class="hero-accent">AI</span></div>'
        '<div class="hero-text">Smart electricity consumption prediction powered by XGBoost regression.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    history = st.session_state.prediction_history
    latest = history[-1]["Prediction (kWh)"] if history else None
    previous = history[-2]["Prediction (kWh)"] if len(history) > 1 else None

    trend_html = ""
    if latest is not None and previous is not None:
        diff = latest - previous
        if diff > 0:
            trend_html = f'<div class="card-trend up">▲ {diff:+.2f} kWh vs last</div>'
        elif diff < 0:
            trend_html = f'<div class="card-trend down">▼ {diff:+.2f} kWh vs last</div>'
        else:
            trend_html = '<div class="card-trend flat">— no change vs last</div>'

    cols = st.columns(4)
    cards = [
        ("🤖", "Prediction Model", "XGBoost", ""),
        ("⚡", "Latest Prediction", f"{latest:.2f} kWh" if latest is not None else "Ready", trend_html),
        ("📊", "Predictions Made", str(len(history)), ""),
        ("🚨", "Monitoring", "Active", ""),
    ]
    for col, (icon, label, value, trend) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="card"><div class="card-icon">{icon}</div>'
                f'<div class="card-label">{label}</div>'
                f'<div class="card-value">{value}</div>{trend}</div>',
                unsafe_allow_html=True
            )

    st.write("")
    left, right = st.columns([1.35, 1])

    with left:
        # Everything for this card - heading, subtitle, and body text - is
        # built into ONE string and passed to a SINGLE st.markdown call.
        # Splitting an opening <div> and closing </div> across separate
        # st.markdown/st.subheader calls doesn't nest them: each call
        # renders as its own independent block, so the div ends up empty
        # and the real content spills out underneath it, unstyled.
        st.markdown(
            '<div class="section-card">'
            '<div class="section-card-title">🚀 Quick Start</div>'
            '<div class="small-muted" style="margin-bottom:10px;">Make a prediction in three simple steps.</div>'
            '<div class="section-card-body">'
            '1. Open <b>XGBoost Prediction</b> from the sidebar.<br>'
            '2. Enter temperature, humidity, occupancy and time conditions.<br>'
            '3. Select <b>Predict Electricity Consumption</b> to see the estimate.'
            '</div></div>',
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            '<div class="section-card">'
            '<div class="section-card-title">🧠 What the system uses</div>'
            '<div class="section-card-body">'
            'Environmental conditions, occupancy, time and calendar-related features '
            'are prepared through the saved preprocessor and passed to the trained XGBoost model.'
            '</div></div>',
            unsafe_allow_html=True
        )

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
elif page == "XGBoost Prediction":
    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">🤖 XGBoost <span class="hero-accent">Prediction</span></div>'
        '<div class="hero-text">Enter the conditions and generate an electricity-consumption estimate.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    section_title("📝 Input Conditions")
    # Reads from - and writes to - the SAME "shared" store used by the
    # Simulator's Current Scenario and the Forecast page, so a value
    # typed here shows up there too, and vice versa. Only the Simulator's
    # Simulated Scenario stays independent, since that's meant to be a
    # different hypothetical, not a copy of the current conditions.
    values = render_condition_inputs("pred", defaults=remembered_defaults("shared"))
    remember_values("shared", values)

    st.write("")
    if st.button("⚡ Predict Electricity Consumption", type="primary"):
        try:
            with st.spinner("Running the XGBoost model..."):
                prediction = predict_consumption(**values)
                add_history(
                    (values["temperature"], values["humidity"], values["occupancy"],
                     values["hour"], values["day"], values["month"], values["day_of_week"],
                     values["is_weekend"], values["season"], values["is_peak_hour"]),
                    prediction
                )
            label, css_class = status_for(prediction)
            # Saved to session_state (not just local variables) because the
            # Download Report button below triggers its own rerun when
            # clicked - if the result only lived in local variables inside
            # this `if st.button(...)` block, it would vanish the instant
            # someone clicked Download, since the Predict button itself
            # resets to False on that rerun.
            st.session_state.last_prediction = dict(
                values=values, prediction=prediction, label=label, css_class=css_class
            )
        except Exception as e:
            st.session_state.pop("last_prediction", None)
            st.error("Prediction could not be completed.")
            with st.expander("Technical details"):
                st.code(str(e))

    if "last_prediction" in st.session_state:
        lp = st.session_state.last_prediction
        values, prediction, label, css_class = lp["values"], lp["prediction"], lp["label"], lp["css_class"]

        st.success("Prediction completed successfully.")
        st.markdown(
            f'<div class="result-card {css_class}">'
            f'<div class="result-label">Predicted Electricity Consumption</div>'
            f'<div class="result-number">{prediction:.2f} kWh</div>'
            f'<div class="result-status">{label}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # 1. High Consumption Alert - compares against the user-adjustable
        # threshold (set on the Smart Alerts page), not the fixed low/
        # moderate/high bands used for the status label above.
        if prediction >= st.session_state.alert_threshold:
            st.error(
                f"🚨 High Consumption Alert: {prediction:.2f} kWh is at or above "
                f"your alert threshold of {st.session_state.alert_threshold:.1f} kWh."
            )

        fig = go.Figure(go.Bar(
            x=["Predicted Consumption"], y=[prediction],
            marker_color=COLOR_ACCENT, text=[f"{prediction:.2f}"], textposition="outside",
        ))
        fig.update_layout(title="Current Prediction")
        st.plotly_chart(style_fig(fig), use_container_width=True)

        # 3. Energy-Saving Recommendations - generated from the specific
        # inputs just entered (see get_recommendations), not generic text.
        st.divider()
        section_title("🌱 Energy-Saving Recommendations")
        for tip in get_recommendations(values, prediction, label):
            st.markdown(f"- {tip}")

        # 4. Download Report - a plain-text file built from this exact
        # prediction, generated fresh each render so it always matches
        # what's on screen. If a bill estimate exists from this session,
        # it's folded in automatically.
        st.divider()
        bill_lines = None
        if "last_bill" in st.session_state:
            lb = st.session_state.last_bill
            bill_lines = [
                f"Tariff             : Rs.{lb['tariff']:.2f} per kWh",
                f"Estimated Bill     : Rs.{lb['cost']:,.2f}",
            ]
        report_text = build_report_text(values, prediction, label, bill_lines=bill_lines)
        st.download_button(
            "📥 Download Report",
            data=report_text,
            file_name=f"electricity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

# -------------------- SIMULATOR --------------------
elif page == "Energy Impact Simulator":
    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">🔬 Energy Impact <span class="hero-accent">Simulator</span></div>'
        '<div class="hero-text">Compare two conditions using the trained XGBoost model.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    section_title("📌 Current Scenario")
    # Same shared store as the Prediction page - this is meant to
    # represent the actual current conditions, not a Simulator-only value.
    current_values = render_condition_inputs("cur", defaults=remembered_defaults("shared"))
    remember_values("shared", current_values)

    st.divider()
    section_title("🔬 Simulated Scenario")
    sim_defaults = {"occupancy": 80.0, "hour": 18}
    simulated_values = render_condition_inputs("sim", defaults=remembered_defaults("sim", sim_defaults))
    remember_values("sim", simulated_values)

    if st.button("🔬 Simulate Energy Impact", type="primary"):
        try:
            with st.spinner("Running both scenarios through the model..."):
                current = predict_consumption(**current_values)
                simulated = predict_consumption(**simulated_values)
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

            colors = [COLOR_ACCENT2, COLOR_ACCENT]
            fig = go.Figure(go.Bar(
                x=["Current", "Simulated"], y=[current, simulated],
                marker_color=colors, text=[f"{current:.2f}", f"{simulated:.2f}"], textposition="outside",
            ))
            fig.update_layout(title="Current vs Simulated Consumption")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        except Exception as e:
            st.error("Simulation could not be completed.")
            with st.expander("Technical details"):
                st.code(str(e))

# -------------------- BILL ESTIMATOR --------------------
elif page == "Bill Estimator":
    st.markdown(
        '<div class="hero"><div class="hero-title">💰 Electricity Bill <span class="hero-accent">Estimator</span></div>'
        '<div class="hero-text">Estimated cost = your most recent predicted consumption × tariff.</div></div>',
        unsafe_allow_html=True
    )

    history = st.session_state.prediction_history
    if not history:
        empty_state("💰", "No prediction data yet", "Go to XGBoost Prediction and make a prediction first, then come back here.")
    else:
        # Always the latest prediction - no checkbox, no "which value am I
        # using" ambiguity. If you want to price a different prediction,
        # make a new one on the Prediction page first.
        prediction = history[-1]["Prediction (kWh)"]

        section_title("⚡ Predicted Consumption", "Taken automatically from your most recent XGBoost prediction.")
        st.metric("Predicted Consumption", f"{prediction:.2f} kWh")

        st.divider()
        section_title("💵 Enter Electricity Tariff")
        # step=0.01 so + / - move by paise (₹0.01), not by 50 paise jumps -
        # real tariffs are usually quoted to 2 decimal places.
        tariff = st.number_input(
            "Electricity tariff (₹ per kWh)", min_value=0.0, value=7.00, step=0.01, format="%.2f"
        )

        cost = prediction * tariff
        st.markdown(
            f'<div class="result-card">'
            f'<div class="result-label">Estimated Electricity Bill</div>'
            f'<div class="result-number">₹{cost:,.2f}</div>'
            f'<div class="result-status">{prediction:.2f} kWh × ₹{tariff:.2f}/kWh</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.caption(
            "This is an estimate based only on your predicted consumption. "
            "Actual electricity bills may include slabs, fixed charges, and taxes."
        )

        # Stored so the Prediction page's downloadable report can include
        # this estimate automatically - build_report_text() just checks
        # whether this key exists.
        st.session_state.last_bill = dict(tariff=tariff, cost=cost)

# -------------------- ANALYTICS --------------------
elif page == "Analytics":
    st.markdown(
        '<div class="hero"><div class="hero-title">📊 Prediction <span class="hero-accent">Analytics</span></div>'
        '<div class="hero-text">Understand the predictions generated during this session.</div></div>',
        unsafe_allow_html=True
    )
    if not st.session_state.prediction_history:
        empty_state("📊", "No prediction data yet", "Make a prediction first to see analytics here.")
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
        fig = go.Figure(go.Scatter(
            x=list(range(1, len(df)+1)), y=vals, mode="lines+markers",
            line=dict(color=COLOR_ACCENT, width=3), marker=dict(size=8, color=COLOR_ACCENT),
        ))
        st.plotly_chart(style_fig(fig, x_title="Prediction Number"), use_container_width=True)

        st.divider()
        section_title("📋 Prediction Data")
        st.dataframe(df, use_container_width=True, hide_index=True)

# -------------------- FORECAST --------------------
elif page == "Forecast":
    st.markdown(
        '<div class="hero"><div class="hero-title">🔮 Electricity <span class="hero-accent">Forecast</span></div>'
        '<div class="hero-text">Generate a model-based estimate for upcoming hours.</div></div>',
        unsafe_allow_html=True
    )
    values = render_condition_inputs("forecast", include_hour=False, defaults=remembered_defaults("shared"))
    remember_values("shared", values)
    c1, c2 = st.columns(2)
    with c1:
        start = st.slider("🕐 Starting Hour", 0, 23, st.session_state.get("forecast_start_remembered", 12))
    with c2:
        hours = st.slider("⏱️ Hours to Forecast", 1, 12, st.session_state.get("forecast_hours_remembered", 6))
    st.session_state.forecast_start_remembered = start
    st.session_state.forecast_hours_remembered = hours

    if st.button("🔮 Generate Forecast", type="primary"):
        try:
            with st.spinner("Generating forecast..."):
                results = []
                for i in range(hours):
                    h = (start + i) % 24
                    pred = predict_consumption(
                        temperature=values["temperature"], humidity=values["humidity"],
                        occupancy=values["occupancy"], hour=h, day=values["day"],
                        month=values["month"], day_of_week=values["day_of_week"],
                        is_weekend=values["is_weekend"], season=values["season"],
                        is_peak_hour=values["is_peak_hour"],
                    )
                    results.append({"Hour": f"{h:02d}:00", "Predicted Consumption (kWh)": round(pred, 2)})
                fdf = pd.DataFrame(results)
            st.success("Forecast generated successfully.")
            vals = fdf["Predicted Consumption (kWh)"]
            a,b,c = st.columns(3)
            a.metric("Average", f"{vals.mean():.2f} kWh")
            b.metric("Highest", f"{vals.max():.2f} kWh")
            c.metric("Lowest", f"{vals.min():.2f} kWh")

            fig = go.Figure(go.Scatter(
                x=fdf["Hour"], y=vals, mode="lines+markers",
                line=dict(color=COLOR_ACCENT, width=3), marker=dict(size=8, color=COLOR_ACCENT),
                fill="tozeroy", fillcolor="rgba(56,189,248,.10)",
            ))
            st.plotly_chart(style_fig(fig, x_title="Hour"), use_container_width=True)
            st.dataframe(fdf, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error("Forecast could not be generated.")
            with st.expander("Technical details"):
                st.code(str(e))

# -------------------- MODEL INSIGHTS --------------------
elif page == "Model Insights":
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
            fig = go.Figure(go.Bar(
                x=imp["Importance"], y=imp["Feature"], orientation="h",
                marker_color=COLOR_ACCENT,
            ))
            fig.update_layout(title="XGBoost Feature Importance")
            st.plotly_chart(style_fig(fig, y_title="Feature", x_title="Importance"), use_container_width=True)
            st.dataframe(imp.sort_values("Importance", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.warning("The saved model's feature count does not match the displayed feature names.")
    except Exception:
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
elif page == "Smart Alerts":
    st.markdown(
        '<div class="hero"><div class="hero-title">🚨 Smart <span class="hero-accent">Alerts</span></div>'
        '<div class="hero-text">Monitor the latest predicted consumption level.</div></div>',
        unsafe_allow_html=True
    )

    section_title("🎚️ Alert Threshold", "Predictions at or above this value trigger a High Consumption Alert.")
    # key="alert_threshold" ties this slider directly to the session_state
    # value initialized at the top of the file, so the XGBoost Prediction
    # page's alert check always reads whatever was last set here.
    st.slider("Alert threshold (kWh)", 1.0, 15.0, key="alert_threshold", step=0.5)

    st.divider()
    if not st.session_state.prediction_history:
        empty_state("🚨", "No prediction data yet", "Make a prediction first to see alerts here.")
    else:
        df = pd.DataFrame(st.session_state.prediction_history)
        latest = float(df.iloc[-1]["Prediction (kWh)"])
        threshold = st.session_state.alert_threshold
        section_title("⚡ Current Consumption Status")
        if latest >= threshold:
            st.error(f"🔴 ALERT — latest prediction {latest:.2f} kWh is at or above the {threshold:.1f} kWh threshold")
        elif latest >= 5:
            st.warning(f"🟡 MODERATE CONSUMPTION — {latest:.2f} kWh")
        else:
            st.success(f"🟢 LOW CONSUMPTION — {latest:.2f} kWh")

        high = int((df["Prediction (kWh)"] >= 8).sum())
        moderate = int(((df["Prediction (kWh)"] >= 5) & (df["Prediction (kWh)"] < 8)).sum())
        low = int((df["Prediction (kWh)"] < 5).sum())
        exceeded = int((df["Prediction (kWh)"] >= threshold).sum())
        a,b,c,d = st.columns(4)
        a.metric("🔴 High", high)
        b.metric("🟡 Moderate", moderate)
        c.metric("🟢 Low", low)
        d.metric("🚨 Over Threshold", exceeded)

        st.divider()
        if latest >= threshold:
            st.warning("Consider reducing unnecessary electricity usage and monitoring high-power appliances.")
        elif latest >= 5:
            st.info("Consumption is moderate. Continue monitoring usage, especially during peak hours.")
        else:
            st.success("The latest predicted consumption is relatively low.")

# -------------------- HISTORY --------------------
elif page == "Prediction History":
    st.markdown(
        '<div class="hero"><div class="hero-title">📋 Prediction <span class="hero-accent">History</span></div>'
        '<div class="hero-text">Review predictions generated during the current application session.</div></div>',
        unsafe_allow_html=True
    )
    if not st.session_state.prediction_history:
        empty_state("📋", "No prediction history yet", "Predictions you make will be listed here.")
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
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "📥 Download Full History (CSV)",
                data=df.to_csv(index=False),
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        with d2:
            if st.button("🗑️ Clear Prediction History"):
                st.session_state.prediction_history = []
                st.rerun()
