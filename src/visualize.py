import utils
import matplotlib.pyplot as plt
import os
import pyarrow.parquet as pq
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

config = utils.load_config()
VARS = ["swvl1", "ro", "tp", "e"]



def visualize_data():
    print("Visualizing data...")
    variable_name = "swvl1"
    time_str = "2020-01-15T12:00"
    demo_nc_path = config["demo_nc_path"]
    ds = xr.open_dataset(demo_nc_path)
    print(ds)

    visualize_single_layer(ds, variable_name, time_str)
    
    print("Visualization completed.")

    


def visualize_single_layer(ds, variable_name, time_str):
    print(ds)
    
    data = ds[variable_name].sel(valid_time=time_str)  
    lat = ds.latitude
    lon = ds.longitude

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.pcolormesh(lon, lat, data, cmap='YlGnBu', shading='auto')

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.set_title(f"{variable_name} at time {time_str}")
    plt.colorbar(im, orientation='vertical', label=variable_name)
    plt.tight_layout()
    plt.show()