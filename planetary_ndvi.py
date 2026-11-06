import matplotlib.pyplot as plt
import numpy as np
import odc.stac
import planetary_computer
import pystac_client


def fetch_sentinel2_ndvi(bbox, time_range, max_cloud=15, show_plot=True):
    """
    Поиск снимков Sentinel-2 L2A через Planetary Computer STAC,
    расчет пространственного распределения NDVI и среднего индекса.
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    print(f"Поиск снимков Sentinel-2 для bbox {bbox} ({time_range})...")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )

    items = list(search.items())
    print(f"Найдено снимков: {len(items)}")

    if not items:
        print("Подходящих безоблачных снимков не найдено.")
        return None

    item = items[-1]
    print(f"Используется снимок от {item.datetime.strftime('%Y-%m-%d %H:%M')}")

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
    print(f"Средний NDVI для выбранного участка: {mean_ndvi:.4f}")

    if show_plot:
        plt.figure(figsize=(7, 6))
        plt.imshow(ndvi.squeeze(), cmap="RdYlGn", vmin=-0.1, vmax=0.9)
        plt.colorbar(label="NDVI")
        plt.title(f"Sentinel-2 NDVI ({item.datetime.strftime('%Y-%m-%d')})")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return mean_ndvi


if __name__ == "__main__":
    test_bbox = [37.80, 55.49, 37.84, 55.52]
    test_time = "2026-05-01/2026-09-01"

    fetch_sentinel2_ndvi(test_bbox, test_time)
