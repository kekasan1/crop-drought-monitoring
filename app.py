from datetime import datetime, timedelta
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac_client
import requests
import streamlit as st
from catboost import CatBoostClassifier

st.set_page_config(page_title="Мониторинг засухи полей", layout="wide")


@st.cache_resource
def load_model(model_path="drought_catboost_model.cbm"):
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def fetch_weather_data(lat: float, lon: float, past_days: int = 90) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,soil_moisture_0_to_7cm",
        "past_days": past_days,
        "forecast_days": 5,
        "timezone": "auto",
    }

    resp = requests.get(url, params=params, timeout=12)
    resp.raise_for_status()
    data = resp.json()

    if "hourly" not in data:
        raise ValueError(f"Некорректный ответ погодного API: {data}")

    hourly = data["hourly"]
    df = pd.DataFrame({
        "date": pd.to_datetime(hourly["time"]),
        "temp": hourly["temperature_2m"],
        "precip": hourly["precipitation"],
        "soil_moisture": hourly["soil_moisture_0_to_7cm"],
    })

    df.set_index("date", inplace=True)
    daily = pd.DataFrame({
        "temp_max": df["temp"].resample("D").max(),
        "temp_mean": df["temp"].resample("D").mean(),
        "precip_sum": df["precip"].resample("D").sum(),
        "soil_moisture_mean": df["soil_moisture"].resample("D").mean(),
    }).reset_index()

    return daily


def fetch_sentinel_ndvi(lat: float, lon: float, days_back: int = 120):
    bbox = [lon - 0.015, lat - 0.012, lon + 0.015, lat + 0.012]
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    time_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": 25}},
    )

    items = list(search.items())
    if not items:
        return None, None

    items.sort(key=lambda x: x.datetime, reverse=True)
    item = items[0]

    ds = odc.stac.load([item], bands=["red", "nir"], bbox=bbox, resolution=10)
    red = ds.red.astype(np.float32) / 10000.0
    nir = ds.nir.astype(np.float32) / 10000.0

    denominator = nir + red
    ndvi = np.where(denominator == 0, 0, (nir - red) / denominator)
    valid_ndvi = ndvi[~np.isnan(ndvi)]
    mean_ndvi = float(valid_ndvi.mean()) if len(valid_ndvi) > 0 else 0.0

    return mean_ndvi, item.datetime


st.title("Система мониторинга и прогнозирования засухи")
st.markdown("Оценка погодных условий и вегетационного индекса (Sentinel-2 NDVI) с предсказанием риска водного стресса.")

st.sidebar.header("Параметры участка")
lat = st.sidebar.number_input("Широта (Latitude)", value=55.827, format="%.4f")
lon = st.sidebar.number_input("Долгота (Longitude)", value=37.558, format="%.4f")

if st.sidebar.button("Запустить анализ поля", type="primary"):
    with st.spinner("Загрузка метеоданных и спутниковых снимков..."):
        daily_w = None
        try:
            daily_w = fetch_weather_data(lat, lon)
            st.success("Метеорологические данные успешно получены")
        except Exception as e:
            st.error(f"Ошибка загрузки метеоданных: {e}")

        current_ndvi = None
        try:
            current_ndvi, capture_time = fetch_sentinel_ndvi(lat, lon)
            if current_ndvi is not None:
                st.info(f"Снимок Sentinel-2 от {capture_time.strftime('%Y-%m-%d')}. Средний NDVI: {current_ndvi:.4f}")
            else:
                current_ndvi = 0.65
                st.warning("Свежих безоблачных снимков не найдено. Применено базовое значение NDVI: 0.65")
        except Exception as e:
            current_ndvi = 0.65
            st.warning(f"Не удалось подключиться к Planetary Computer: {e}. Применено значение по умолчанию (0.65).")

        if daily_w is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Динамика максимальной температуры (°C)")
                st.line_chart(daily_w.set_index("date")["temp_max"])

            with col2:
                st.subheader("Средняя влажность почвы (0-7 см)")
                st.line_chart(daily_w.set_index("date")["soil_moisture_mean"])

            try:
                model = load_model()
                last_row = daily_w.iloc[-1]
                x_pred = pd.DataFrame([{
                    "temp_max": last_row["temp_max"],
                    "temp_mean": last_row["temp_mean"],
                    "precip_sum": last_row["precip_sum"],
                    "soil_moisture_mean": last_row["soil_moisture_mean"],
                    "ndvi": current_ndvi,
                }])

                drought_prob = model.predict_proba(x_pred)[0][1]

                st.markdown("---")
                st.subheader("Оценка риска засухи (CatBoost)")

                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Вероятность засухи", f"{drought_prob:.1%}")
                with metric_col2:
                    st.metric("Текущий NDVI", f"{current_ndvi:.3f}")

                if drought_prob > 0.5:
                    st.error(f"Высокий риск засухи и водного стресса растительности (вероятность: {drought_prob:.1%}). Рекомендуется полив.")
                else:
                    st.success(f"Риск засухи минимален ({drought_prob:.1%}). Влажность почвы и состояние растительности в норме.")

            except Exception:
                st.info("Файл весов модели (drought_catboost_model.cbm) не найден. Запустите model_builder.ipynb для обучения.")
