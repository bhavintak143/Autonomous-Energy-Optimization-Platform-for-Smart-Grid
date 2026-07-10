"""

===============================================================================

 AUTONOMOUS ENERGY OPTIMIZATION PLATFORM FOR SMART GRID  —  app.py

===============================================================================

"""
import os
import datetime as dt
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Autonomous Energy Optimization Platform | Smart Grid",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at 20% 0%, #10233b 0%, #0b1120 45%, #0b1120 100%);
    }
    #MainMenu, footer, header {visibility: hidden;}
    h1, h2, h3, h4 { color: #e6f1ff; font-family: 'Segoe UI', sans-serif; }
    p, li, span, label, .stMarkdown { color: #c7d2e0; }

    @keyframes glow {
        0%   { box-shadow: 0 0 18px rgba(47,217,196,0.15); }
        50%  { box-shadow: 0 0 30px rgba(47,217,196,0.35); }
        100% { box-shadow: 0 0 18px rgba(47,217,196,0.15); }
    }
    @keyframes flow {
        0%   { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    .hero {
        position: relative;
        padding: 1.8rem 2.2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #0f2b4a 0%, #123a52 25%, #0a5c56 50%, #123a52 75%, #0f2b4a 100%);
        background-size: 200% 200%;
        animation: glow 4s ease-in-out infinite, flow 12s linear infinite;
        border: 1px solid #1f3b57;
        margin-bottom: 1.4rem;
        overflow: hidden;
    }
    .hero h1 { margin: 0; font-size: 2.3rem; letter-spacing: 0.01em; }
    .hero p { margin: 0.4rem 0 0 0; color: #9fd8d1; font-size: 1.05rem; }
    .hero .badge {
        display: inline-block; margin-top: 0.6rem; margin-right: 0.4rem; padding: 0.25rem 0.7rem;
        background: rgba(47,217,196,0.12); border: 1px solid #2fd9c4;
        border-radius: 999px; color: #2fd9c4; font-size: 0.75rem; letter-spacing: 0.03em;
    }

    .cube-scene { perspective: 700px; width: 64px; height: 64px; margin: 4px auto 14px auto; }
    .cube {
        width: 64px; height: 64px; position: relative;
        transform-style: preserve-3d;
        animation: spin3d 9s linear infinite;
    }
    .cube-face {
        position: absolute; width: 64px; height: 64px;
        display: flex; align-items: center; justify-content: center;
        font-size: 26px; color: #eaf6ff;
        background: linear-gradient(135deg, #14807a, #0e5c56);
        border: 1px solid #2fd9c4; border-radius: 10px;
        box-shadow: 0 0 14px rgba(47,217,196,0.25);
    }
    .cf-front  { transform: translateZ(32px); }
    .cf-back   { transform: translateZ(-32px) rotateY(180deg); }
    .cf-right  { transform: rotateY(90deg) translateZ(32px); }
    .cf-left   { transform: rotateY(-90deg) translateZ(32px); }
    .cf-top    { transform: rotateX(90deg) translateZ(32px); }
    .cf-bottom { transform: rotateX(-90deg) translateZ(32px); }
    @keyframes spin3d {
        from { transform: rotateY(0deg) rotateX(12deg); }
        to   { transform: rotateY(360deg) rotateX(12deg); }
    }
    .cube-caption { text-align: center; color: #6fd3c7; font-size: 0.72rem; letter-spacing: 0.05em; margin-top: -8px; margin-bottom: 10px; text-transform: uppercase; }

    .metric-card {
        background: linear-gradient(160deg, #101c33 0%, #0d1830 100%);
        border: 1px solid #1f3b57;
        border-radius: 16px;
        padding: 1rem 1.15rem;
        height: 100%;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover { transform: translateY(-3px); border-color: #2fd9c4; }
    .metric-card .label { color: #8ea3c0; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { color: #eaf6ff; font-size: 1.65rem; font-weight: 700; margin-top: 0.15rem; }
    .metric-card .sub { color: #6fd3c7; font-size: 0.78rem; margin-top: 0.2rem; }

    .tip-card {
        background: linear-gradient(90deg, #0f1f36 0%, #0d1c30 100%);
        border-left: 4px solid #2fd9c4;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        color: #dbe8f5;
        font-size: 0.96rem;
    }

    .beginner-box {
        background: rgba(138,127,247,0.08); border: 1px dashed #8a7ff7;
        border-radius: 10px; padding: 0.65rem 1rem; margin: 0.5rem 0 1rem 0;
        color: #cfc9fb; font-size: 0.85rem;
    }

    .status-ok { color: #2fd9c4; font-weight: 600; }
    .status-bad { color: #ff6b6b; font-weight: 600; }

    section[data-testid="stSidebar"] { background-color: #0a121f; border-right: 1px solid #1f3b57; }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] {
        background-color: #101c33; border-radius: 10px 10px 0 0; color: #9fb3cc; padding: 0.55rem 1.1rem;
    }
    .stTabs [aria-selected="true"] { background-color: #123a52 !important; color: #eaf6ff !important; }

    div.stButton > button, div.stDownloadButton > button {
        background: linear-gradient(135deg, #14807a, #0e5c56);
        color: #eaf6ff; border: none; border-radius: 10px; font-weight: 600;
        width: 100%;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background: linear-gradient(135deg, #1ba69d, #147a71);
    }

    button[data-testid="collapsedControl"],
    button[data-testid="stSidebarCollapsedControl"],
    div[data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
        pointer-events: auto !important;
    }

    #sg-sidebar-toggle {
        position: fixed;
        top: 0.7rem;
        left: 0.7rem;
        z-index: 999999;
        width: 38px;
        height: 38px;
        border-radius: 8px;
        background: linear-gradient(135deg, #14807a, #0e5c56);
        border: 1px solid #2fd9c4;
        box-shadow: 0 0 10px rgba(47,217,196,0.35);
        color: #eaf6ff;
        font-size: 18px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        user-select: none;
        transition: background 0.15s ease;
    }
    #sg-sidebar-toggle:hover {
        background: linear-gradient(135deg, #1ba69d, #147a71);
    }

    @media (max-width: 768px) {
        .hero { padding: 1.2rem 1.3rem; }
        .hero h1 { font-size: 1.5rem; }
        .hero p { font-size: 0.9rem; }
        .metric-card .value { font-size: 1.25rem; }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

SIDEBAR_TOGGLE_HTML = """
<div id="sg-sidebar-toggle" title="Show / hide sidebar">☰</div>
<script>
    function sgFindNativeToggle() {
        return window.parent.document.querySelector('button[data-testid="collapsedControl"]')
            || window.parent.document.querySelector('button[data-testid="stSidebarCollapsedControl"]')
            || window.parent.document.querySelector('[data-testid="stSidebarNav"] button')
            || window.parent.document.querySelector('section[data-testid="stSidebar"] button[kind="header"]');
    }

    function sgToggleSidebar() {
        let btn = sgFindNativeToggle();
        if (btn) {
            btn.click();
            return;
        }
        let sidebar = window.parent.document.querySelector('section[data-testid="stSidebar"]');
        if (sidebar) {
            let expanded = sidebar.getAttribute('aria-expanded');
            if (expanded === 'true') {
                sidebar.setAttribute('aria-expanded', 'false');
                sidebar.style.marginLeft = '-350px';
            } else {
                sidebar.setAttribute('aria-expanded', 'true');
                sidebar.style.marginLeft = '0px';
            }
        }
    }

    document.getElementById('sg-sidebar-toggle').addEventListener('click', sgToggleSidebar);
</script>
"""
st.markdown(SIDEBAR_TOGGLE_HTML, unsafe_allow_html=True)

FEATURE_COLUMNS = [
    "day_of_week", "is_weekend", "month", "is_holiday",
    "energy_yesterday", "temperatureMax", "temperatureMin",
    "humidity", "windSpeed", "cloudCover",
]

DEFAULT_OUTPUT_DIR = "outputs"
MODEL_FILE = "energy_forecast_model.pkl"
MERGED_FILE = "cleaned_merged_energy_data.csv.gz"
PROFILE_FILE = "household_usage_groups.csv"

FIXED_UK_HOLIDAYS = {(1, 1), (12, 25), (12, 26)}


@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_merged_data(path):
    df = pd.read_csv(path)
    if "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"])
    return df


@st.cache_data(show_spinner=False)
def load_house_profile(path):
    df = pd.read_csv(path)
    id_col = df.columns[0]
    return df.rename(columns={id_col: "LCLid"})


@st.cache_data(show_spinner=False)
def compute_model_metrics(_model, merged_data):
    model_data = merged_data.dropna(subset=FEATURE_COLUMNS + ["energy_sum"])
    x = model_data[FEATURE_COLUMNS]
    y = model_data["energy_sum"]
    _, test_x, _, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
    preds = _model.predict(test_x)

    mae = mean_absolute_error(test_y, preds)
    rmse = mean_squared_error(test_y, preds) ** 0.5
    r2 = r2_score(test_y, preds)

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": _model.feature_importances_,
    }).sort_values("importance", ascending=True)

    return {"mae": mae, "rmse": rmse, "r2": r2, "test_y": test_y, "preds": preds, "importance": importance}


def build_energy_tips(group_summary, forecast_mae, peak_hour=None):
    tips = []
    if peak_hour is not None:
        tips.append(
            f"Peak demand happens around {int(peak_hour)}:00. Shifting heavy appliances "
            f"(washing machine, dishwasher, EV charging) away from this hour can help "
            f"reduce strain on the grid and may lower costs on dynamic tariffs."
        )
    else:
        tips.append(
            "Upload a half-hourly usage file from the sidebar to unlock the peak-hour "
            "shifting recommendation for this dataset."
        )
    heavy_group = group_summary["avg_energy"].idxmax()
    light_group = group_summary["avg_energy"].idxmin()
    tips.append(
        f"Usage group {heavy_group} uses the most energy on average. These households "
        f"are good targets for efficiency programs (insulation checks, smart thermostats, "
        f"appliance upgrades)."
    )
    tips.append(
        f"Usage group {light_group} already uses energy efficiently and could be a "
        f"reference group to benchmark others against."
    )
    tips.append(
        f"The forecasting model is accurate to within about {round(forecast_mae, 2)} kWh "
        f"per household per day on average — good enough to plan next-day grid load and "
        f"pre-emptively warn high-usage households."
    )
    weekend_vs_weekday = group_summary["weekend_energy"].mean() - group_summary["weekday_energy"].mean()
    if weekend_vs_weekday > 0:
        tips.append(
            "On average, households use more energy on weekends than weekdays. "
            "Weekend-specific pricing or reminders could help flatten demand."
        )
    else:
        tips.append(
            "On average, households use more energy on weekdays than weekends. "
            "This lines up with work-from-home and evening routines."
        )
    return tips


def predict_scenario(model, params):
    row = pd.DataFrame([params])[FEATURE_COLUMNS]
    return float(model.predict(row)[0])


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_city(city_name):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    resp = requests.get(url, params={"name": city_name, "count": 1}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("results"):
        return None
    r = data["results"][0]
    return {"lat": r["latitude"], "lon": r["longitude"], "label": f"{r['name']}, {r.get('country', '')}"}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather_forecast(lat, lon, days=7):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,cloudcover",
        "forecast_days": days, "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date

    daily = df.groupby("date").agg(
        temperatureMax=("temperature_2m", "max"),
        temperatureMin=("temperature_2m", "min"),
        humidity=("relativehumidity_2m", "mean"),
        windSpeed=("windspeed_10m", "mean"),
        cloudCover=("cloudcover", "mean"),
    ).reset_index()

    daily["humidity"] = daily["humidity"] / 100.0
    daily["cloudCover"] = daily["cloudCover"] / 100.0
    return daily


def recursive_forecast(model, weather_daily, last_known_energy):
    rows = []
    energy_yesterday = last_known_energy
    for _, row in weather_daily.iterrows():
        d = pd.Timestamp(row["date"])
        month, day_of_week = d.month, d.dayofweek
        is_weekend = int(day_of_week in (5, 6))
        is_holiday = int((d.month, d.day) in FIXED_UK_HOLIDAYS)

        features = {
            "day_of_week": day_of_week, "is_weekend": is_weekend, "month": month,
            "is_holiday": is_holiday, "energy_yesterday": energy_yesterday,
            "temperatureMax": row["temperatureMax"], "temperatureMin": row["temperatureMin"],
            "humidity": row["humidity"], "windSpeed": row["windSpeed"], "cloudCover": row["cloudCover"],
        }
        pred = predict_scenario(model, features)
        rows.append({"date": d, "predicted_energy": pred, **row.drop("date").to_dict()})
        energy_yesterday = pred

    return pd.DataFrame(rows)


def metric_card(col, label, value, sub=""):
    col.markdown(
        f"""<div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
                <div class="sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def beginner_box(text):
    if st.session_state.get("beginner_mode", True):
        st.markdown(f"<div class='beginner-box'>📘 <b>In plain English:</b> {text}</div>", unsafe_allow_html=True)


def cube_logo():
    st.markdown(
        """
        <div class="cube-scene">
          <div class="cube">
            <div class="cube-face cf-front">⚡</div>
            <div class="cube-face cf-back">⚡</div>
            <div class="cube-face cf-right">⚡</div>
            <div class="cube-face cf-left">⚡</div>
            <div class="cube-face cf-top">⚡</div>
            <div class="cube-face cf-bottom">⚡</div>
          </div>
        </div>
        <div class="cube-caption">Smart Grid AI</div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    cube_logo()
    st.markdown("### ⚡ Control Panel")

    st.session_state["beginner_mode"] = st.toggle("🧠 Beginner explanations", value=True)

    output_dir = DEFAULT_OUTPUT_DIR
    with st.expander("⚙️ Advanced: data source", expanded=False):
        output_dir = st.text_input("Outputs folder", value=DEFAULT_OUTPUT_DIR)

    model_path = os.path.join(output_dir, MODEL_FILE)
    merged_path = os.path.join(output_dir, MERGED_FILE)
    profile_path = os.path.join(output_dir, PROFILE_FILE)
    missing_any = not (os.path.exists(model_path) and os.path.exists(merged_path) and os.path.exists(profile_path))
    if missing_any:
        st.error("Required data files not found. Run the training notebook first, or set the correct folder above.")

    st.markdown("---")
    with st.expander("📂 Optional: hourly usage file"):
        hh_upload = st.file_uploader("Upload a half-hourly block CSV", type=["csv"], label_visibility="collapsed")
        st.caption("Unlocks the peak-hour recommendation on the Insights tab.")

    st.markdown("---")
    st.markdown("#### 🧪 Scenario Simulator")
    st.caption("Change any parameter — Trends & Model tabs update instantly.")

    day_of_week = st.selectbox("Day of week", options=list(range(7)),
                                format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
    is_weekend = int(day_of_week in (5, 6))
    month = st.slider("Month", 1, 12, 6)
    is_holiday = int(st.checkbox("Bank holiday"))
    energy_yesterday = st.number_input("Yesterday's energy (kWh)", value=10.0, step=1.0)
    temperatureMax = st.slider("Max temperature (°C)", -5.0, 40.0, 18.0)
    temperatureMin = st.slider("Min temperature (°C)", -10.0, 30.0, 10.0)
    humidity = st.slider("Humidity (0–1)", 0.0, 1.0, 0.7)
    windSpeed = st.slider("Wind speed", 0.0, 15.0, 4.0)
    cloudCover = st.slider("Cloud cover (0–1)", 0.0, 1.0, 0.5)

    scenario_params = {
        "day_of_week": day_of_week, "is_weekend": is_weekend, "month": month, "is_holiday": is_holiday,
        "energy_yesterday": energy_yesterday, "temperatureMax": temperatureMax, "temperatureMin": temperatureMin,
        "humidity": humidity, "windSpeed": windSpeed, "cloudCover": cloudCover,
    }

    st.markdown("---")
    st.markdown("#### 🌦️ Live Forecast")
    city = st.text_input("City", value="London")
    run_forecast = st.button("🔮 Get 7-Day Forecast", use_container_width=True)

if missing_any:
    st.markdown(
        """<div class="hero">
            <h1>⚡ Autonomous Energy Optimization Platform</h1>
            <p>Smart Grid Insight Dashboard</p>
           </div>""",
        unsafe_allow_html=True,
    )
    st.info("Waiting for model artifacts. Run the training notebook, or point to the correct folder in the sidebar's Advanced section.")
    st.stop()


with st.spinner("Loading model and data..."):
    model = load_model(model_path)
    merged_data = load_merged_data(merged_path)
    house_profile = load_house_profile(profile_path)
    metrics = compute_model_metrics(model, merged_data)
    group_summary = house_profile.groupby("usage_group").mean(numeric_only=True)

peak_hour = None
hourly_pattern = None
if hh_upload is not None:
    hh_df = pd.read_csv(hh_upload)
    time_col = "tstp" if "tstp" in hh_df.columns else hh_df.columns[1]
    hh_df[time_col] = pd.to_datetime(hh_df[time_col], errors="coerce")
    hh_df["hour"] = hh_df[time_col].dt.hour
    energy_col = [c for c in hh_df.columns if "energy" in c.lower()][0]
    hh_df[energy_col] = pd.to_numeric(hh_df[energy_col], errors="coerce")
    hourly_pattern = hh_df.groupby("hour")[energy_col].mean()
    peak_hour = hourly_pattern.idxmax()

tips = build_energy_tips(group_summary, metrics["mae"], peak_hour)

n_households = merged_data["LCLid"].nunique() if "LCLid" in merged_data.columns else house_profile.shape[0]
avg_daily_energy = merged_data["energy_sum"].mean() if "energy_sum" in merged_data.columns else np.nan

scenario_prediction = predict_scenario(model, scenario_params)


st.markdown(
    """<div class="hero">
        <h1>⚡ Autonomous Energy Optimization Platform</h1>
        <p>AI-powered smart-grid dashboard — explore historical trends, simulate scenarios, and forecast demand live.</p>
        <span class="badge">Random Forest Forecast Model</span>
        <span class="badge">London Smart Meter Data</span>
       </div>""",
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)
metric_card(k1, "Households", f"{n_households:,}")
metric_card(k2, "Daily Records", f"{merged_data.shape[0]:,}")
metric_card(k3, "Avg Daily Energy", f"{avg_daily_energy:.2f} kWh")
metric_card(k4, "Model R²", f"{metrics['r2']:.3f}", "Random Forest")
metric_card(k5, "Your Scenario", f"{scenario_prediction:.2f} kWh", "live, from sidebar")

beginner_box(
    "R² tells you how well the model explains energy usage (closer to 1 = better). "
    "\"Your Scenario\" is the model's live prediction for whatever settings you choose in the sidebar."
)

st.markdown("####")

tab_trends, tab_model, tab_clusters, tab_tips, tab_live, tab_export = st.tabs(
    ["📈 Trends & Scenario", "🤖 Model Performance", "🧩 Household Clusters",
     "💡 Optimization Insights", "🔮 Live Forecast", "📤 Export Report"]
)

with tab_trends:
    st.markdown("##### 🎯 Your Scenario vs. Historical Average")
    beginner_box(
        "Move any slider in the sidebar (month, temperature, humidity...) and watch this gauge "
        "and comparison update immediately — that's the model reacting to your inputs in real time."
    )

    low = merged_data["energy_sum"].quantile(0.33)
    high = merged_data["energy_sum"].quantile(0.66)
    max_range = merged_data["energy_sum"].quantile(0.99)

    gc1, gc2 = st.columns([1.1, 1])
    with gc1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=scenario_prediction,
            delta={"reference": avg_daily_energy, "increasing": {"color": "#ff6b6b"}, "decreasing": {"color": "#2fd9c4"}},
            title={"text": "Predicted Energy for Your Scenario (kWh)"},
            gauge={
                "axis": {"range": [0, max_range]},
                "bar": {"color": "#2fd9c4"},
                "steps": [
                    {"range": [0, low], "color": "#123a52"},
                    {"range": [low, high], "color": "#0e5c56"},
                    {"range": [high, max_range], "color": "#7a3b3b"},
                ],
            },
        ))
        fig.update_layout(template="plotly_dark", height=340)
        st.plotly_chart(fig, use_container_width=True)
    with gc2:
        compare_df = pd.DataFrame({
            "series": ["Historical Average", "Your Scenario"],
            "energy": [avg_daily_energy, scenario_prediction],
        })
        fig = px.bar(compare_df, x="series", y="energy", color="series", text_auto=".2f",
                     template="plotly_dark", title="Scenario vs. Average Household Day",
                     color_discrete_sequence=["#8a7ff7", "#2fd9c4"])
        fig.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 📈 Historical Trends (no live parameters — full dataset)")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        daily_trend = merged_data.groupby("day")["energy_sum"].mean().reset_index()
        fig = px.line(daily_trend, x="day", y="energy_sum", template="plotly_dark",
                      title="Average Daily Energy Use Over Time")
        fig.update_traces(line_color="#2fd9c4")
        fig.add_hline(y=scenario_prediction, line_dash="dot", line_color="#ff9f5b",
                       annotation_text="Your scenario", annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "Acorn_grouped" in merged_data.columns:
            grp = merged_data.groupby("Acorn_grouped")["energy_sum"].mean().sort_values().reset_index()
            fig = px.bar(grp, x="energy_sum", y="Acorn_grouped", orientation="h",
                         template="plotly_dark", title="Avg Energy Use by ACORN Group",
                         color="energy_sum", color_continuous_scale="Teal")
            st.plotly_chart(fig, use_container_width=True)
    beginner_box(
        "ACORN is a UK household classification (e.g. by income/lifestyle). This chart shows which "
        "types of households use more or less energy on average."
    )

    if "temperatureMax" in merged_data.columns:
        fig = px.scatter(merged_data.sample(min(5000, len(merged_data)), random_state=1),
                          x="temperatureMax", y="energy_sum", opacity=0.35,
                          template="plotly_dark", title="Energy Use vs Max Temperature",
                          color_discrete_sequence=["#ff9f5b"])
        fig.add_vline(x=temperatureMax, line_dash="dot", line_color="#2fd9c4",
                       annotation_text="Your scenario temp")
        st.plotly_chart(fig, use_container_width=True)

    if hourly_pattern is not None:
        fig = px.line(hourly_pattern.reset_index(), x="hour", y=hourly_pattern.name,
                      markers=True, template="plotly_dark",
                      title="Average Energy Use by Hour of Day (uploaded sample)")
        fig.update_traces(line_color="#ff6b6b")
        st.plotly_chart(fig, use_container_width=True)

with tab_model:
    m1, m2, m3 = st.columns(3)
    metric_card(m1, "MAE", f"{metrics['mae']:.3f} kWh")
    metric_card(m2, "RMSE", f"{metrics['rmse']:.3f} kWh")
    metric_card(m3, "R² Score", f"{metrics['r2']:.3f}")
    beginner_box(
        "MAE = average prediction error in kWh. RMSE punishes big misses more. "
        "R² closer to 1 means the model explains the data well."
    )

    c1, c2 = st.columns(2)
    with c1:
        pv = pd.DataFrame({"actual": metrics["test_y"], "predicted": metrics["preds"]})
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=pv["actual"], y=pv["predicted"], mode="markers",
                                    marker=dict(color="#8a7ff7", opacity=0.35), name="Households"))
        lims = [pv["actual"].min(), pv["actual"].max()]
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(color="#ff6b6b", dash="dash"), name="Ideal"))
        fig.update_layout(template="plotly_dark", title="Predicted vs Actual Daily Energy Use",
                           xaxis_title="Actual (kWh)", yaxis_title="Predicted (kWh)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(metrics["importance"], x="importance", y="feature", orientation="h",
                     template="plotly_dark", title="What Drives the Forecast Most",
                     color="importance", color_continuous_scale="Tealgrn")
        st.plotly_chart(fig, use_container_width=True)
    beginner_box(
        "The left chart shows how close predictions are to reality (closer to the dashed line = better). "
        "The right chart ranks which inputs matter most to the model — try changing the top one in the sidebar!"
    )

with tab_clusters:
    c1, c2 = st.columns([1, 1.3])
    with c1:
        counts = house_profile["usage_group"].value_counts().sort_index().reset_index()
        counts.columns = ["usage_group", "count"]
        fig = px.pie(counts, names="usage_group", values="count", template="plotly_dark",
                     title="Households per Usage Group", hole=0.45,
                     color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(house_profile, x="avg_energy", y="energy_spread", color="usage_group",
                          template="plotly_dark", title="Household Profiles by Usage Group",
                          color_continuous_scale="Turbo",
                          labels={"avg_energy": "Avg Daily Energy (kWh)", "energy_spread": "Variability (std)"})
        st.plotly_chart(fig, use_container_width=True)
    beginner_box(
        "Clustering groups households with similar usage patterns — useful for targeting "
        "efficiency programs at the right customers."
    )

    st.markdown("##### Usage Group Summary")
    st.dataframe(group_summary.style.format("{:.2f}").background_gradient(cmap="BuGn"), use_container_width=True)

    weekday_weekend = group_summary[["weekday_energy", "weekend_energy"]].reset_index()
    weekday_weekend_m = weekday_weekend.melt(id_vars="usage_group", var_name="period", value_name="energy")
    fig = px.bar(weekday_weekend_m, x="usage_group", y="energy", color="period", barmode="group",
                 template="plotly_dark", title="Weekday vs Weekend Energy by Group",
                 color_discrete_sequence=["#2fd9c4", "#ff9f5b"])
    st.plotly_chart(fig, use_container_width=True)

with tab_tips:
    st.markdown("##### Plain-English Optimization Insights")
    for i, tip in enumerate(tips, start=1):
        st.markdown(f"<div class='tip-card'>{i}. {tip}</div>", unsafe_allow_html=True)
    tips_text = "\n".join(f"{i}. {t}" for i, t in enumerate(tips, start=1))
    st.download_button("⬇️ Download insights as .txt", tips_text, file_name="energy_optimization_insights.txt")

with tab_live:
    st.caption(f"Live forecast for **{city}** — triggered from the sidebar.")

    if run_forecast:
        try:
            with st.spinner("Fetching live weather and running the model..."):
                loc = geocode_city(city)
                if loc is None:
                    st.error(f"Could not find a location for '{city}'. Try a different spelling.")
                else:
                    weather_daily = fetch_weather_forecast(loc["lat"], loc["lon"], days=7)
                    last_known_energy = merged_data.sort_values("day")["energy_sum"].tail(7).mean()
                    forecast_df = recursive_forecast(model, weather_daily, last_known_energy)
                    st.session_state["last_forecast"] = (loc, forecast_df)
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the weather service: {e}")

    if "last_forecast" in st.session_state:
        loc, forecast_df = st.session_state["last_forecast"]
        st.success(f"Forecast generated for **{loc['label']}**")

        f1, f2, f3 = st.columns(3)
        metric_card(f1, "7-Day Avg Predicted Energy", f"{forecast_df['predicted_energy'].mean():.2f} kWh")
        peak_row = forecast_df.loc[forecast_df["predicted_energy"].idxmax()]
        metric_card(f2, "Predicted Peak Day", peak_row["date"].strftime("%a %d %b"),
                    f"{peak_row['predicted_energy']:.2f} kWh")
        metric_card(f3, "±Model Error Margin", f"{metrics['mae']:.2f} kWh")
        beginner_box(
            "This tab pulls a real 7-day weather forecast for your chosen city and feeds it into the "
            "trained model, one day at a time, to project future household energy demand."
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["predicted_energy"] + metrics["mae"],
            line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["predicted_energy"] - metrics["mae"],
            fill="tonexty", fillcolor="rgba(47,217,196,0.15)", line=dict(width=0),
            name="Error margin"))
        fig.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["predicted_energy"],
            mode="lines+markers", line=dict(color="#2fd9c4", width=3),
            name="Predicted energy"))
        fig.update_layout(template="plotly_dark", title=f"7-Day Energy Demand Projection — {loc['label']}",
                           xaxis_title="Date", yaxis_title="Predicted Avg Energy (kWh)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            forecast_df[["date", "predicted_energy", "temperatureMax", "temperatureMin", "humidity", "windSpeed", "cloudCover"]]
            .style.format({"predicted_energy": "{:.2f}", "temperatureMax": "{:.1f}", "temperatureMin": "{:.1f}",
                           "humidity": "{:.2f}", "windSpeed": "{:.1f}", "cloudCover": "{:.2f}"}),
            use_container_width=True,
        )
    else:
        st.info("Enter a city and click **🔮 Get 7-Day Forecast** in the sidebar to generate a live projection.")

with tab_export:
    st.markdown("##### Export a snapshot of this dashboard")

    report_lines = [
        "AUTONOMOUS ENERGY OPTIMIZATION PLATFORM — REPORT SNAPSHOT",
        f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "-" * 60,
        f"Households analyzed: {n_households}",
        f"Daily records: {merged_data.shape[0]}",
        f"Average daily energy: {avg_daily_energy:.2f} kWh",
        f"Model R²: {metrics['r2']:.3f} | MAE: {metrics['mae']:.3f} | RMSE: {metrics['rmse']:.3f}",
        f"Current scenario prediction: {scenario_prediction:.2f} kWh",
        "-" * 60,
        "OPTIMIZATION INSIGHTS:",
    ] + [f"{i}. {t}" for i, t in enumerate(tips, start=1)]

    report_text = "\n".join(report_lines)
    st.text_area("Preview", report_text, height=300)

    e1, e2 = st.columns(2)
    e1.download_button("⬇️ Download report (.txt)", report_text, file_name="smart_grid_report.txt", use_container_width=True)
    e2.download_button("⬇️ Download cluster table (.csv)", house_profile.to_csv(index=False),
                        file_name="household_usage_groups_export.csv", use_container_width=True)
