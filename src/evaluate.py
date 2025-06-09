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

    demo_sequence_evaluation(
        ds=ds,
        variable_name="swvl1",
        start_time="2020-01-15T12:00",
        sequence_length=11
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

def demo_sequence_evaluation(ds, variable_name, start_time, sequence_length=11):
    print(f"Running sequence evaluation from {start_time} over {sequence_length} days...")
    
    start_idx = int(np.where(ds.valid_time.values == np.datetime64(start_time))[0])
    if start_idx + sequence_length > len(ds.valid_time):
        raise ValueError("Not enough data for sequence evaluation.")

    base_data = ds[variable_name].isel(valid_time=start_idx)
    r2_list, mae_list, time_list = [], [], []

    for offset in range(1, sequence_length):
        pred_data = ds[variable_name].isel(valid_time=start_idx + offset)

        mask = np.isfinite(base_data) & np.isfinite(pred_data)
        y_true = base_data.values[mask]
        y_pred = pred_data.values[mask]

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        r2_list.append(r2)
        mae_list.append(mae)
        time_str = str(ds.valid_time[start_idx + offset * 24].values)[:10]
        time_list.append(time_str)

        print(f"{time_str} => R²: {r2:.4f}, MAE: {mae:.4f}")

    visualize_sequence_metrics(time_list, r2_list, mae_list)

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

def visualize_sequence_metrics(time_list, r2_list, mae_list):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = list(range(1, len(time_list) + 1))

    ax1.set_xlabel("Prediction Time (Day)")
    ax1.set_ylabel("R²", color="tab:blue")
    ax1.plot(x, r2_list, label="R²", color="tab:blue", marker='o')
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0.9, 1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("MAE", color="tab:red")
    ax2.plot(x, mae_list, label="MAE", color="tab:red", marker='x')
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Evaluation Metrics Over 10 Days")
    fig.tight_layout()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()