import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac_client


def collect_ndvi_timeseries(bbox, time_range, max_cloud=20):
    """Сбор временного ряда среднего NDVI по безоблачным снимкам Sentinel-2."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    print(f"Поиск сцен Sentinel-2 для bbox {bbox} ({time_range})...")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )

    items = list(search.items())
    print(f"Найдено снимков: {len(items)}")
    if not items:
        return None

    records = []
    for item in items:
        date_str = item.datetime.strftime("%Y-%m-%d")
        print(f"Обработка снимка за {date_str}...")

        try:
            ds = odc.stac.load(
                [item],
                bands=["red", "nir"],
                bbox=bbox,
                resolution=10,
            )

            red = ds.red.astype(np.float32) / 10000.0
            nir = ds.nir.astype(np.float32) / 10000.0

            denominator = nir + red
            ndvi = np.where(denominator == 0, 0, (nir - red) / denominator)

            valid_ndvi = ndvi[~np.isnan(ndvi)]
            mean_ndvi = float(valid_ndvi.mean()) if len(valid_ndvi) > 0 else 0.0

            records.append({
                "date": pd.to_datetime(date_str),
                "ndvi": mean_ndvi,
            })
        except Exception as e:
            print(f"Ошибка при обработке {date_str}: {e}")

    df_ndvi = pd.DataFrame(records)
    if not df_ndvi.empty:
        df_ndvi = df_ndvi.groupby("date", as_index=False)["ndvi"].mean()
    return df_ndvi


if __name__ == "__main__":
    test_bbox = [37.545, 55.820, 37.575, 55.832]
    test_time = "2026-01-01/2026-09-20"

    ndvi_df = collect_ndvi_timeseries(test_bbox, test_time)

    if ndvi_df is not None and not ndvi_df.empty:
        print(f"\nСобрано наблюдений NDVI: {len(ndvi_df)}")

        try:
            meteo_df = pd.read_csv("meteo_history.csv")
            meteo_df["date"] = pd.to_datetime(meteo_df["date"])

            # Объединение метеорологических данных и спутниковых наблюдений
            dataset = pd.merge(meteo_df, ndvi_df, on="date", how="left")

            # Интерполяция пропусков между пролетами спутника
            dataset["ndvi"] = dataset["ndvi"].interpolate(method="linear").bfill().ffill()

            output_filename = "final_drought_dataset.csv"
            dataset.to_csv(output_filename, index=False)
            print(f"Итоговый датасет сохранен в {output_filename}")
            print(dataset.head(10))

        except FileNotFoundError:
            print("Файл meteo_history.csv не найден. Запустите meteo_archive.py для сбора погоды.")
