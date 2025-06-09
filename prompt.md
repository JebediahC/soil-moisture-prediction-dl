我们准备进行Soil Moisture Prediction系统的编写，请你首先帮我生成一个代码框架，然后我会引导你一步步完善这个系统。

这个系统由以下部分组成
- Raw Dataset Download
- Preprocess Dataset
- Virtualization
- Input trained models and data
- Prediction
- Evaluation

系统配置：
- 语言：python
- 可以单独运行其中一部分
- 

数据详情：
- Ingest ERA5-Land input data
- Layers: 4(volumetric soil water layer 1, total runoff, total evaporation, total precipitation)
- spatial resolution(0.1°)
- Input window 10 days
- format: NetCDF
- Region: Europe (35°N–72°N, 25°W–45°E)


## Raw Dataset Download

```pseudo
# Generate pseudo codes
```
参考[download data](#download-data-from-cdsclimate-data-store-api)

## Preprocess Dataset
```pseudo
# Generate pseudo codes
```
参考[proprocess](#preprocess-data)

## Virtualization
data format(after preprocessing)：
- shape: 371(latitude 37° and 0.1° resolution)×701(longitude 70° and 0.1° resolution)×4(4 parameters)×10(10 days input window and )
- raw data's format(hourly, averaging to daily needed) is as [dateset format](#dataset-format)

Display maps for each layer


## Input trained models and data
Input data format(after preprocessing)

Load model


## Prediction
input: 
data format(after preprocessing)：
- shape: 371(latitude 37° and 0.1° resolution)×701(longitude 70° and 0.1° resolution)×4(4 parameters)×10(10 days input window and )
- raw data's format(hourly, averaging to daily needed) is as [dateset format](#dataset-format)


## Evaluation
As a demo or test, please select 2 moments of swvl1 data from the raw dateset with 1 day offset(such as 2020-01-15T12:00:00 and 2020-01-15T12:00:00), and compare the 2 moments as the (fake) ground true and prediction, and calculate the Evaluation Metrics and virtualize the error heatmap.

use this Evaluation Metrics to evaluate the accuracy:
- RMSE
- R2
- MAE
- MSE

data format: refer as raw data's format(hourly, averaging to daily needed) is as [dateset format](#dataset-format)

And lay the foundation for continuous expansion in the future.

## Reference

### dataset format
```
<xarray.Dataset> Size: 3GB
Dimensions:     (valid_time: 744, latitude: 371, longitude: 701)
Coordinates:
    number      int64 8B ...
  * valid_time  (valid_time) datetime64[ns] 6kB 2020-01-01 ... 2020-01-31T23:...
  * latitude    (latitude) float64 3kB 72.0 71.9 71.8 71.7 ... 35.2 35.1 35.0
  * longitude   (longitude) float64 6kB -25.0 -24.9 -24.8 ... 44.8 44.9 45.0
    expver      (valid_time) <U4 12kB ...
Data variables:
    swvl1       (valid_time, latitude, longitude) float32 774MB ...
    ro          (valid_time, latitude, longitude) float32 774MB ...
    e           (valid_time, latitude, longitude) float32 774MB ...
    tp          (valid_time, latitude, longitude) float32 774MB ...
Attributes:
    GRIB_centre:             ecmf
    GRIB_centreDescription:  European Centre for Medium-Range Weather Forecasts
    GRIB_subCentre:          0
    Conventions:             CF-1.7
    institution:             European Centre for Medium-Range Weather Forecasts
    history:                 2025-05-11T20:16 GRIB to CDM+CF via cfgrib-0.9.1...
```

### Download data from CDS(Climate Data Store) API

```python
import cdsapi

dataset = "reanalysis-era5-land"
request = {
    "variable": [
        "volumetric_soil_water_layer_1",
        "runoff",
        "total_evaporation",
        "total_precipitation"
    ],
    "year": "2025",
    "month": "05",
    "day": [
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [72, -25, 35, 45]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
```

### Preprocess data

```pseudo
# some pseudo code needed here

```