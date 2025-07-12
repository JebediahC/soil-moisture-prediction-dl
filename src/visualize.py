import utils
import matplotlib.pyplot as plt
import os
import pyarrow.parquet as pq
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

config = utils.load_config()
VARS = ["swvl1", "ro", "tp", "e"]



# def visualize_data():
#     print("Visualizing data...")
#     variable_name = "swvl1"
#     time_str = "2020-01-15T12:00"
#     demo_nc_path = config["demo_nc_path"]
#     ds = xr.open_dataset(demo_nc_path)
#     print(ds)

#     visualize_single_layer(ds, variable_name, time_str)
    
#     print("Visualization completed.")

    


# def visualize_single_layer(ds, variable_name, time_str):
#     print(ds)
    
#     data = ds[variable_name].sel(valid_time=time_str)  
#     lat = ds.latitude
#     lon = ds.longitude

#     plt.figure(figsize=(12, 6))
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     im = ax.pcolormesh(lon, lat, data, cmap='YlGnBu', shading='auto')

#     ax.coastlines()
#     ax.add_feature(cfeature.BORDERS)
#     ax.set_title(f"{variable_name} at time {time_str}")
#     plt.colorbar(im, orientation='vertical', label=variable_name)
#     plt.tight_layout()
#     plt.show()

class Visualizer:
    """
    General visualization tools for soil moisture and related variables.
    """

    def __init__(self, lat=None, lon=None, var_names=None):
        """
        lat, lon: 1D or 2D arrays for latitude and longitude
        var_names: list of variable/channel names (optional, for labeling)
        """
        self.lat = lat
        self.lon = lon
        self.var_names = var_names

    def plot_variable_day(self, data, lat=None, lon=None, day_index=0, channel_index=0, title=None, cmap='YlGnBu'):
        """
        Visualize a single day's single channel variable.
        data: 3D array (lat, lon, channels) or (H, W, C)
        lat, lon: arrays, if None use self.lat/self.lon
        day_index: for title only
        channel_index: which channel to plot
        """
        lat = lat if lat is not None else self.lat
        lon = lon if lon is not None else self.lon
        var_data = data[..., channel_index]
        plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = ax.pcolormesh(lon, lat, var_data, cmap=cmap, shading='auto')
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        var_label = self.var_names[channel_index] if self.var_names else f"Var {channel_index}"
        ax.set_title(title or f"Day {day_index}, {var_label}")
        plt.colorbar(im, orientation='vertical', label=var_label)
        plt.tight_layout()
        plt.show()

    def plot_tensor_sample(self, x, y, lat=None, lon=None):
        """
        Visualize a sample's input and output.
        x: (input_days, C, H, W)
        y: (predict_days, 1, H, W)
        """
        lat = lat if lat is not None else self.lat
        lon = lon if lon is not None else self.lon
        # Plot input sequence
        for t in range(x.shape[0]):
            # (C, H, W) -> (H, W, C)
            self.plot_variable_day(np.transpose(x[t], (1, 2, 0)), lat, lon, t+1, 0, f"Input day {t+1}")
        # Plot output sequence
        for t in range(y.shape[0]):
            self.plot_variable_day(np.transpose(y[t], (1, 2, 0)), lat, lon, t+1, 0, f"Target day {t+1}")

    def plot_metric_map(self, metric_map, title="Metric Map", cmap="coolwarm", vmin=None, vmax=None):
        """
        Plot a 2D metric map (e.g., RMSE, MAE) on the lat/lon grid.
        metric_map: 2D array (lat, lon)
        """
        lat = self.lat
        lon = self.lon
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        im = ax.pcolormesh(lon, lat, metric_map, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(title)
        plt.colorbar(im, orientation='vertical', label=title)
        plt.tight_layout()
        plt.show()

    def plot_metric_trend(self, metric_list, metric_name="Metric", predict_days=None):
        """
        Plot metric (e.g., RMSE, MAE, R2) trend over prediction days.
        metric_list: list or array of metric values
        """
        days = predict_days if predict_days is not None else len(metric_list)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, days+1), metric_list, marker='o', label=metric_name)
        plt.xlabel("Prediction Day")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} Over Prediction Days")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_box(self, data_list, labels, title="Metric Distribution", ylabel="Value"):
        """
        Boxplot for metric distributions.
        data_list: list of arrays (e.g., [all_rmse, all_mae])
        labels: list of str
        """
        plt.figure(figsize=(8, 6))
        plt.boxplot([d[~np.isnan(d)] for d in data_list], labels=labels, showfliers=False)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


