import requests
import pandas as pd


def get_historical_meteo_data(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Загрузка исторических метеоданных с Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "precipitation", "soil_moisture_0_to_7cm"],
        "timezone": "auto",
    }

    print(f"Запрос архива погоды ({lat}, {lon}) с {start_date} по {end_date}...")
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    hourly_data = response.json()["hourly"]

    df = pd.DataFrame({
        "date": pd.to_datetime(hourly_data["time"]),
        "temperature_2m": hourly_data["temperature_2m"],
        "precipitation": hourly_data["precipitation"],
        "soil_moisture_0_7cm": hourly_data["soil_moisture_0_to_7cm"],
    })

    df.set_index("date", inplace=True)
    daily = pd.DataFrame()
    daily["temp_max"] = df["temperature_2m"].resample("D").max()
    daily["temp_mean"] = df["temperature_2m"].resample("D").mean()
    daily["precip_sum"] = df["precipitation"].resample("D").sum()
    daily["soil_moisture_mean"] = df["soil_moisture_0_7cm"].resample("D").mean()
    daily["precip_14d_sum"] = daily["precip_sum"].rolling(window=14, min_periods=1).sum()

    daily = daily.reset_index()
    print(f"Обработано {len(daily)} дней.")
    return daily


if __name__ == "__main__":
    lat, lon = 55.50946, 37.82086
    start_date, end_date = "2026-01-01", "2026-09-20"

    history_df = get_historical_meteo_data(lat, lon, start_date, end_date)
    if history_df is not None:
        csv_filename = "meteo_history.csv"
        history_df.to_csv(csv_filename, index=False)
        print(f"Данные сохранены в {csv_filename}")
        print(history_df.head())
