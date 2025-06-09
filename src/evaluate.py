import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import utils


def run_evaluation():
    print("Running evaluation...")

    config = utils.load_config()
    demo_nc_path = config["demo_nc_path"]
    ds = xr.open_dataset(demo_nc_path)

    demo_comparison(
        ds=ds,
        variable_name="swvl1",
        true_time="2020-01-15T12:00",
        pred_time="2020-01-16T12:00"
    )
    print("Evaluation completed.")
    return

def demo_comparison(ds, variable_name, true_time, pred_time):
    true_data = ds[variable_name].sel(valid_time=true_time)
    pred_data = ds[variable_name].sel(valid_time=pred_time)

    # Flatten data for calculating evaluation indicators (excluding NaN values)
    mask = np.isfinite(true_data) & np.isfinite(pred_data)
    y_true = true_data.values[mask]
    y_pred = pred_data.values[mask]

    # calcuate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print("Evaluation Metrics:")
    print(f"RMSE: {rmse:.5f}")
    print(f"R²  : {r2:.5f}")
    print(f"MAE : {mae:.5f}")
    print(f"MSE : {mse:.5f}")

    # visualize the error map
    error_map = np.abs(pred_data - true_data)
    visualize_error_map(error_map, ds.latitude, ds.longitude, title=f"Absolute Error ({true_time} vs {pred_time})")


def visualize_error_map(error_data, lat, lon, title="Error Map"):
   
    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.pcolormesh(lon, lat, error_data, cmap='Reds', shading='auto')
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.set_title(title)
    plt.colorbar(im, orientation='vertical', label='Absolute Error')
    plt.tight_layout()
    plt.show()