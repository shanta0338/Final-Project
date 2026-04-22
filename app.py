"""
Global Weather Temperature Prediction — Streamlit App
Loads the MLflow-logged WeatherPredictor model and predicts temperature.
Run: streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import mlflow.pytorch
import pickle
import os
from datetime import datetime

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Global Weather Predictor",
    page_icon="🌤️",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Model definition  (must match your notebook)
# ─────────────────────────────────────────────
class WeatherPredictor(nn.Module):
    def __init__(self, input_dim):
        super(WeatherPredictor, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu1  = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2  = nn.ReLU()
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return self.output_layer(x)


# ─────────────────────────────────────────────
#  Load model & scaler (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(run_id: str):
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/weather_predictor_model")
    model.eval()
    return model

@st.cache_resource
def load_scaler(scaler_path: str):
    with open(scaler_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_label_encoders(le_path: str):
    with open(le_path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
#  Sidebar — model loading
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Model Configuration")

run_id     = st.sidebar.text_input("MLflow Run ID", placeholder="Paste your run_id here")
scaler_path = st.sidebar.text_input("Scaler path (.pkl)", value="scaler.pkl")
le_path     = st.sidebar.text_input("Label encoders path (.pkl)", value="label_encoders.pkl")

model, scaler, label_encoders = None, None, None

if run_id:
    try:
        model = load_model(run_id)
        st.sidebar.success("✅ Model loaded")
    except Exception as e:
        st.sidebar.error(f"Model load failed: {e}")

if os.path.exists(scaler_path):
    scaler = load_scaler(scaler_path)
    st.sidebar.success("✅ Scaler loaded")
else:
    st.sidebar.warning("Scaler file not found — predictions require it.")

if os.path.exists(le_path):
    label_encoders = load_label_encoders(le_path)
    st.sidebar.success("✅ Label encoders loaded")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Save scaler & encoders from your notebook:\n"
    "```python\n"
    "import pickle\n"
    "pickle.dump(scaler, open('scaler.pkl','wb'))\n"
    "pickle.dump(label_encoders, open('label_encoders.pkl','wb'))\n"
    "```"
)


# ─────────────────────────────────────────────
#  Main header
# ─────────────────────────────────────────────
st.title("🌤️ Global Weather Temperature Predictor")
st.markdown("Enter weather conditions below to predict **temperature (°C)** and **feels-like temperature**.")
st.divider()

# ─────────────────────────────────────────────
#  Input form — grouped by category
# ─────────────────────────────────────────────
with st.form("prediction_form"):

    # ── Location ──────────────────────────────
    st.subheader("📍 Location")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        latitude  = st.number_input("Latitude",  value=23.72, min_value=-90.0,  max_value=90.0)
    with col2:
        longitude = st.number_input("Longitude", value=90.41, min_value=-180.0, max_value=180.0)
    with col3:
        country_raw  = st.text_input("Country",  value="Bangladesh")
    with col4:
        location_raw = st.text_input("Location", value="Dhaka")

    timezone_raw = st.text_input("Timezone", value="Asia/Dhaka")

    st.divider()

    # ── Date & Time ───────────────────────────
    st.subheader("🕐 Date & Time")
    col1, col2, col3, col4 = st.columns(4)
    now = datetime.now()
    with col1:
        year  = st.number_input("Year",  value=now.year,  min_value=2000, max_value=2100)
    with col2:
        month = st.number_input("Month", value=now.month, min_value=1,    max_value=12)
    with col3:
        day   = st.number_input("Day",   value=now.day,   min_value=1,    max_value=31)
    with col4:
        hour  = st.number_input("Hour",  value=now.hour,  min_value=0,    max_value=23)

    st.divider()

    # ── Weather Conditions ─────────────────────
    st.subheader("🌬️ Weather Conditions")
    col1, col2, col3 = st.columns(3)
    with col1:
        condition_raw  = st.selectbox("Condition", [
            "Sunny", "Partly cloudy", "Cloudy", "Overcast",
            "Mist", "Patchy rain possible", "Light rain",
            "Moderate rain", "Heavy rain", "Thundery outbreaks possible",
            "Blizzard", "Fog", "Freezing drizzle", "Blowing snow"
        ])
        humidity    = st.slider("Humidity (%)",   0, 100, 70)
        cloud       = st.slider("Cloud Cover (%)", 0, 100, 50)
    with col2:
        wind_kph    = st.number_input("Wind Speed (kph)",  value=15.0, min_value=0.0)
        wind_degree = st.number_input("Wind Degree",       value=180,  min_value=0, max_value=360)
        wind_dir_raw = st.selectbox("Wind Direction", [
            "N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"
        ])
    with col3:
        pressure_mb   = st.number_input("Pressure (mb)",     value=1013.0, min_value=800.0, max_value=1100.0)
        precip_mm     = st.number_input("Precipitation (mm)", value=0.0,   min_value=0.0)
        visibility_km = st.number_input("Visibility (km)",   value=10.0,  min_value=0.0)

    col1, col2 = st.columns(2)
    with col1:
        gust_kph  = st.number_input("Gust Speed (kph)", value=20.0, min_value=0.0)
    with col2:
        uv_index  = st.number_input("UV Index", value=5.0, min_value=0.0, max_value=20.0)

    st.divider()

    # ── Air Quality ────────────────────────────
    st.subheader("🌫️ Air Quality")
    col1, col2, col3 = st.columns(3)
    with col1:
        co   = st.number_input("CO (μg/m³)",   value=233.0, min_value=0.0)
        o3   = st.number_input("Ozone (μg/m³)", value=60.0, min_value=0.0)
    with col2:
        no2  = st.number_input("NO₂ (μg/m³)",  value=10.0, min_value=0.0)
        so2  = st.number_input("SO₂ (μg/m³)",  value=5.0,  min_value=0.0)
    with col3:
        pm25 = st.number_input("PM2.5 (μg/m³)", value=15.0, min_value=0.0)
        pm10 = st.number_input("PM10 (μg/m³)",  value=25.0, min_value=0.0)

    col1, col2 = st.columns(2)
    with col1:
        us_epa    = st.selectbox("US EPA Index",    [1, 2, 3, 4, 5, 6])
    with col2:
        gb_defra  = st.selectbox("GB DEFRA Index",  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    st.divider()

    # ── Astronomical ───────────────────────────
    st.subheader("🌙 Astronomical")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sunrise_raw  = st.text_input("Sunrise",  value="06:00 AM")
    with col2:
        sunset_raw   = st.text_input("Sunset",   value="06:00 PM")
    with col3:
        moonrise_raw = st.text_input("Moonrise", value="06:30 PM")
    with col4:
        moonset_raw  = st.text_input("Moonset",  value="05:00 AM")

    col1, col2 = st.columns(2)
    with col1:
        moon_phase_raw  = st.selectbox("Moon Phase", [
            "New Moon", "Waxing Crescent", "First Quarter",
            "Waxing Gibbous", "Full Moon", "Waning Gibbous",
            "Last Quarter", "Waning Crescent"
        ])
    with col2:
        moon_illumination = st.slider("Moon Illumination (%)", 0, 100, 50)

    st.divider()
    submitted = st.form_submit_button("🔮 Predict Temperature", use_container_width=True)


# ─────────────────────────────────────────────
#  Prediction logic
# ─────────────────────────────────────────────
def encode(le_dict, col, value):
    """Label-encode a value; fall back to 0 if unseen."""
    if le_dict and col in le_dict:
        le = le_dict[col]
        if value in le.classes_:
            return int(le.transform([value])[0])
        else:
            st.warning(f"⚠️ '{value}' not seen during training for '{col}'. Using 0.")
            return 0
    return 0  # no encoder available


if submitted:
    if model is None:
        st.error("❌ Please load a model first (enter your MLflow Run ID in the sidebar).")
    elif scaler is None:
        st.error("❌ Scaler not found. Save it from your notebook and place it next to app.py.")
    else:
        # Build feature row in the EXACT column order used during training
        feature_row = {
            "country":                        encode(label_encoders, "country",       country_raw),
            "location_name":                  encode(label_encoders, "location_name", location_raw),
            "latitude":                       latitude,
            "longitude":                      longitude,
            "timezone":                       encode(label_encoders, "timezone",      timezone_raw),
            "condition_text":                 encode(label_encoders, "condition_text", condition_raw),
            "wind_kph":                       wind_kph,
            "wind_degree":                    wind_degree,
            "wind_direction":                 encode(label_encoders, "wind_direction", wind_dir_raw),
            "pressure_mb":                    pressure_mb,
            "precip_mm":                      precip_mm,
            "humidity":                       humidity,
            "cloud":                          cloud,
            "visibility_km":                  visibility_km,
            "uv_index":                       uv_index,
            "gust_kph":                       gust_kph,
            "air_quality_Carbon_Monoxide":    co,
            "air_quality_Ozone":              o3,
            "air_quality_Nitrogen_dioxide":   no2,
            "air_quality_Sulphur_dioxide":    so2,
            "air_quality_PM2.5":              pm25,
            "air_quality_PM10":               pm10,
            "air_quality_us-epa-index":       us_epa,
            "air_quality_gb-defra-index":     gb_defra,
            "sunrise":                        encode(label_encoders, "sunrise",    sunrise_raw),
            "sunset":                         encode(label_encoders, "sunset",     sunset_raw),
            "moonrise":                       encode(label_encoders, "moonrise",   moonrise_raw),
            "moonset":                        encode(label_encoders, "moonset",    moonset_raw),
            "moon_phase":                     encode(label_encoders, "moon_phase", moon_phase_raw),
            "moon_illumination":              moon_illumination,
            "year":                           int(year),
            "month":                          int(month),
            "day":                            int(day),
            "hour":                           int(hour),
        }

        row_df     = pd.DataFrame([feature_row])
        row_scaled = scaler.transform(row_df.values)
        tensor     = torch.tensor(row_scaled, dtype=torch.float32)

        with torch.no_grad():
            temp_pred = model(tensor).item()

        # Rough feels-like estimate (wind-chill style)
        feels_like = temp_pred - (wind_kph * 0.05) + (humidity * 0.01)

        # ── Result display ─────────────────────
        st.divider()
        st.subheader("🌡️ Prediction Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Temperature (°C)",       f"{temp_pred:.2f} °C")
        c2.metric("Feels Like (°C)",        f"{feels_like:.2f} °C")
        c3.metric("Location",               f"{location_raw}, {country_raw}")

        # Context card
        with st.expander("📋 Full input summary"):
            st.dataframe(row_df.T.rename(columns={0: "Value"}), use_container_width=True)