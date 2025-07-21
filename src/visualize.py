from . import utils
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

    def plot_metric_map(self, metric_map, title="Metric Map", cmap="coolwarm", vmin=None, vmax=None, 
                       outlier_percentile=95, outlier_color='red'):
        """
        Plot a 2D metric map (e.g., RMSE, MAE) on the lat/lon grid.
        metric_map: 2D array (lat, lon)
        outlier_percentile: percentile threshold for outliers (default 95th percentile)
        outlier_color: color for outlier pixels
        """
        lat = self.lat
        lon = self.lon
        
        # 计算有效值的统计信息
        valid_values = metric_map[~np.isnan(metric_map)]
        if len(valid_values) == 0:
            print("Warning: No valid values in metric map")
            return
            
        # 如果没有指定vmax，使用指定百分位数作为上限
        if vmax is None:
            vmax = np.nanpercentile(valid_values, outlier_percentile)
        if vmin is None:
            vmin = np.nanmin(valid_values)
            
        # 创建离群点掩码
        outlier_mask = metric_map > vmax
        
        # 创建用于绘图的数据，将离群点设置为vmax
        plot_data = metric_map.copy()
        plot_data[outlier_mask] = vmax
        
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # 绘制主要数据
        im = ax.pcolormesh(lon, lat, plot_data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        
        # 如果有离群点，用特殊颜色标记
        if np.any(outlier_mask):
            outlier_data = np.full_like(metric_map, np.nan)
            outlier_data[outlier_mask] = metric_map[outlier_mask]
            ax.pcolormesh(lon, lat, outlier_data, cmap='Reds', shading='auto', vmin=vmax, alpha=0.8)
            
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.set_title(f"{title} (>{outlier_percentile}th percentile in red)")
        
        # 添加颜色条
        cbar = plt.colorbar(im, orientation='vertical', label=title, shrink=0.8)
        
        # 添加离群点信息
        n_outliers = np.sum(outlier_mask)
        total_valid = np.sum(~np.isnan(metric_map))
        outlier_ratio = n_outliers / total_valid * 100 if total_valid > 0 else 0
        
        plt.figtext(0.02, 0.02, f"Outliers (>{vmax:.3f}): {n_outliers} pixels ({outlier_ratio:.1f}%)", 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print(f"Metric statistics:")
        print(f"  Valid pixels: {total_valid}")
        print(f"  Min: {vmin:.4f}, Max: {np.nanmax(metric_map):.4f}")
        print(f"  {outlier_percentile}th percentile: {vmax:.4f}")
        print(f"  Outliers: {n_outliers} pixels ({outlier_ratio:.1f}%)")
    
    def plot_comparison(self, y_true, y_pred, day_index=0, channel_index=0, title_prefix="", cmap='YlGnBu'):
        """
        Plot true, predicted, and difference maps side by side.
        y_true, y_pred: (1, H, W) or (H, W)
        """
        lat = self.lat
        lon = self.lon
        
        # Ensure 2D arrays
        if y_true.ndim == 3:
            y_true = y_true.squeeze()
        if y_pred.ndim == 3:
            y_pred = y_pred.squeeze()
            
        # Calculate difference
        diff = y_pred - y_true
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # True values
        im1 = axes[0].pcolormesh(lon, lat, y_true, cmap=cmap, shading='auto')
        axes[0].coastlines()
        axes[0].add_feature(cfeature.BORDERS)
        axes[0].set_title(f"{title_prefix}True Day {day_index+1}")
        plt.colorbar(im1, ax=axes[0], orientation='vertical', label='Value')
        
        # Predicted values
        im2 = axes[1].pcolormesh(lon, lat, y_pred, cmap=cmap, shading='auto', 
                                vmin=np.nanmin(y_true), vmax=np.nanmax(y_true))
        axes[1].coastlines()
        axes[1].add_feature(cfeature.BORDERS)
        axes[1].set_title(f"{title_prefix}Predicted Day {day_index+1}")
        plt.colorbar(im2, ax=axes[1], orientation='vertical', label='Value')
        
        # Difference
        im3 = axes[2].pcolormesh(lon, lat, diff, cmap='RdBu_r', shading='auto')
        axes[2].coastlines()
        axes[2].add_feature(cfeature.BORDERS)
        axes[2].set_title(f"{title_prefix}Difference (Pred - True) Day {day_index+1}")
        plt.colorbar(im3, ax=axes[2], orientation='vertical', label='Difference')
        
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

    def animate_tensor_sample(self, x, y, lat=None, lon=None, interval=500, save_path=None, channel_index=0):
        """
        Display the whole input and output period as a video to show the changing.
        x: (input_days, C, H, W)
        y: (predict_days, 1 or C, H, W)
        interval: delay between frames in ms
        save_path: if provided, save the animation to this path (e.g., 'output.mp4' or 'output.gif')
        channel_index: which channel to display (default 0)
        """
        import matplotlib.animation as animation
        lat = lat if lat is not None else self.lat
        lon = lon if lon is not None else self.lon
        input_days = x.shape[0]
        output_days = y.shape[0]

        def get_var_label():
            if self.var_names and channel_index < len(self.var_names):
                return self.var_names[channel_index]
            return f"Var {channel_index}"

        # Helper to animate a sequence
        def animate_sequence(frames, titles, vmin=None, vmax=None):
            fig = plt.figure(figsize=(10, 5))
            ax = plt.axes(projection=ccrs.PlateCarree())
            im = ax.pcolormesh(lon, lat, frames[0], cmap='YlGnBu', shading='auto', vmin=vmin, vmax=vmax)
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS)
            cb = plt.colorbar(im, orientation='vertical', label=get_var_label())
            title = ax.set_title(titles[0])

            def update(frame_idx):
                im.set_array(frames[frame_idx].ravel())
                title.set_text(titles[frame_idx])
                return [im, title]

            ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
            plt.tight_layout()
            if save_path:
                plt.close(fig)
                return ani
            else:
                plt.close(fig)
                return ani

        # Prepare frames for input and output
        input_frames = []
        input_titles = []
        for t in range(input_days):
            data = np.transpose(x[t], (1, 2, 0))
            input_frames.append(data[..., channel_index])
            input_titles.append(f"Input day {t+1}")

        output_frames = []
        output_titles = []
        for t in range(output_days):
            data = np.transpose(y[t], (1, 2, 0))
            # If output only has 1 channel, always use channel 0
            if data.shape[-1] == 1:
                output_frames.append(data[..., 0])
            else:
                output_frames.append(data[..., channel_index])
            output_titles.append(f"Target day {t+1}")

        if channel_index == 0:
            # Combine input and output for same scale
            all_frames = input_frames + output_frames
            all_titles = input_titles + output_titles
            vmin = np.nanmin([np.nanmin(f) for f in all_frames])
            vmax = np.nanmax([np.nanmax(f) for f in all_frames])
            return animate_sequence(all_frames, all_titles, vmin, vmax)
        else:
            # Display input and output separately (scales may differ)
            vmin_in = np.nanmin([np.nanmin(f) for f in input_frames])
            vmax_in = np.nanmax([np.nanmax(f) for f in input_frames])
            vmin_out = np.nanmin([np.nanmin(f) for f in output_frames])
            vmax_out = np.nanmax([np.nanmax(f) for f in output_frames])
            ani_in = animate_sequence(input_frames, input_titles, vmin_in, vmax_in)
            ani_out = animate_sequence(output_frames, output_titles, vmin_out, vmax_out)
            return ani_in, ani_out


class ArrayInfoDisplayer:
    """
    Utility to log basic info for numpy arrays: shape, dtype, min, max, mean, std.
    """

    def __init__(self, logger):
        """
        logger: a logging.Logger instance or similar object with .info() method
        """
        self.logger = logger

    def print_info(self, arr, name="Array"):
        """
        Log info about the numpy array.
        """
        if arr is None:
            self.logger.info(f"{name}: None")
            return
        self.logger.info(f"{name} info:")
        self.logger.info(f"  shape: {arr.shape}")
        self.logger.info(f"  dtype: {arr.dtype}")
        self.logger.info(f"  min: {np.nanmin(arr):.4f}, max: {np.nanmax(arr):.4f}")
        self.logger.info(f"  mean: {np.nanmean(arr):.4f}, std: {np.nanstd(arr):.4f}")
    
    def downsample_channels(self, arr, target_hw=30):
        """
        对T,H,W,C的每个通道做邻近插值降分辨率到target_hw（边长），返回降分辨率后的ndarray，shape=(T, target_hw, target_hw, C)
        """
        T, H, W, C = arr.shape
        # 计算缩放因子
        scale_h = H // target_hw
        scale_w = W // target_hw
        if scale_h < 1 or scale_w < 1:
            raise ValueError(f"原始分辨率太低，无法降到{target_hw}")
        # 邻近插值：直接取每scale_h, scale_w步的像元
        arr_down = arr[:, ::scale_h, ::scale_w, :]
        # 若不是正好target_hw，裁剪
        arr_down = arr_down[:, :target_hw, :target_hw, :]
        self.logger.info(f"Downsampled array shape: {arr_down.shape}")
        return arr_down

