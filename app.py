import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Predicción de Consumo Energético", page_icon="⚡", layout="centered")

models_dir = "/content/drive/MyDrive/Ejercicios IA/Trabajo final IA"

@st.cache_resource
def load_artifacts():
    label_encoder = joblib.load(os.path.join(models_dir, "labelencoder_energy.joblib"))
    model         = joblib.load(os.path.join(models_dir, "best_stacking_model.joblib"))
    scaler_target = joblib.load(os.path.join(models_dir, "minmaxscaler_energyconsumption.joblib"))
    return label_encoder, model, scaler_target

try:
    label_encoder_energy, best_stacking_model, scaler_target = load_artifacts()
    st.success("✅ Modelos cargados correctamente.")
except Exception as e:
    st.error(f"❌ Error al cargar modelos: {e}")
    st.stop()

FEATURE_COLUMNS = [
    "Temperature", "Humidity", "SquareFootage", "Occupancy", "HVACUsage",
    "LightingUsage", "RenewableEnergy", "Holiday", "solo_la_hora",
    "DayOfWeek_Friday", "DayOfWeek_Monday", "DayOfWeek_Saturday",
    "DayOfWeek_Sunday", "DayOfWeek_Thursday", "DayOfWeek_Tuesday", "DayOfWeek_Wednesday"
]

st.title("⚡ Predicción de Consumo Energético")
st.write("Ingresa las características del edificio para predecir su consumo energético.")

st.subheader("🌡️ Variables ambientales")
col1, col2 = st.columns(2)
with col1:
    temperature    = st.number_input("Temperatura (°C)",        value=22.0, step=0.5)
    humidity       = st.number_input("Humedad (%)",             value=50.0, step=1.0)
with col2:
    square_footage = st.number_input("Área del edificio (m²)", value=1500.0, step=10.0)
    occupancy      = st.number_input("Ocupación (personas)",    value=10, step=1, min_value=0)

st.subheader("🔋 Energía y sistemas")
col3, col4 = st.columns(2)
with col3:
    renewable_energy = st.number_input("Energía renovable (kWh)", value=0.0, step=0.5, min_value=0.0)
    hvac_usage       = st.selectbox("Uso HVAC",       options=["No", "Yes"])
with col4:
    lighting_usage   = st.selectbox("Uso iluminación", options=["No", "Yes"])
    holiday          = st.selectbox("¿Es festivo?",    options=["No", "Yes"])

st.subheader("📅 Fecha y hora")
col5, col6 = st.columns(2)
with col5:
    hour_of_day = st.slider("Hora del día (0–23)", min_value=0, max_value=23, value=12)
with col6:
    day_of_week = st.selectbox(
        "Día de la semana",
        options=["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]
    )

def preprocess_inputs():
    binary_map = {"No": 0, "Yes": 1}
    hvac_enc     = binary_map[hvac_usage]
    lighting_enc = binary_map[lighting_usage]
    try:
        holiday_enc = int(label_encoder_energy.transform([holiday])[0])
    except Exception:
        holiday_enc = binary_map[holiday]

    all_days = ["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]
    day_ohe  = {f"DayOfWeek_{d}": (1.0 if d == day_of_week else 0.0) for d in all_days}

    row = {
        "Temperature":     temperature,
        "Humidity":        humidity,
        "SquareFootage":   square_footage,
        "Occupancy":       float(occupancy),
        "HVACUsage":       float(hvac_enc),
        "LightingUsage":   float(lighting_enc),
        "RenewableEnergy": renewable_energy,
        "Holiday":         float(holiday_enc),
        "solo_la_hora":    float(hour_of_day),
        **day_ohe
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)

if st.button("🔮 Predecir consumo energético", type="primary"):
    try:
        input_df = preprocess_inputs()
        st.write("**Input procesado:**")
        st.dataframe(input_df)

        prediction = best_stacking_model.predict(input_df)
        pred_value = float(prediction[0])

        st.success(f"⚡ Consumo energético predicho: **{pred_value:.4f}** (escala 0–1 normalizada)")
        real_kwh = scaler_target.inverse_transform([[pred_value]])[0][0]
        st.info(f"🔌 Equivale aproximadamente a: **{real_kwh:.2f} kWh**")

    except Exception as e:
        st.error(f"❌ Error durante la predicción: {e}")
        st.exception(e)
