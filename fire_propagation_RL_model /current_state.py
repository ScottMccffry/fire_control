import ee
import numpy as np

ee.Initialize()

def get_fire_state(date, region):
    fire_data = ee.ImageCollection('FIRMS').filterDate(date).mean().clip(region)
    fire_matrix = fire_data.select('T21').reduceRegion(ee.Reducer.toList(), region, scale=1000).get('T21').getInfo()
    return np.array(fire_matrix)

def get_elevation(region):
    elevation_data = ee.Image('USGS/SRTMGL1_003').clip(region)
    elevation_matrix = elevation_data.reduceRegion(ee.Reducer.toList(), region, scale=1000).get('elevation').getInfo()
    return np.array(elevation_matrix)

def get_wind(date, region):
    wind_data = ee.ImageCollection('NASA/NLDAS/FORA0125_H002').filterDate(date).mean().clip(region)
    wind_u = wind_data.select('wind_u').reduceRegion(ee.Reducer.toList(), region, scale=1000).get('wind_u').getInfo()
    wind_v = wind_data.select('wind_v').reduceRegion(ee.Reducer.toList(), region, scale=1000).get('wind_v').getInfo()
    return np.array(wind_u), np.array(wind_v)

def get_humidity(date, region):
    humidity_data = ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD').filterDate(date).mean().clip(region)
    humidity_matrix = humidity_data.reduceRegion(ee.Reducer.toList(), region, scale=1000).get('precipitation').getInfo()
    return np.array(humidity_matrix)

# Define your region of interest
region = ee.Geometry.Rectangle([left, bottom, right, top])

# Define the date for data extraction
date = '2023-01-01'

fire_state = get_fire_state(date, region)
elevation = get_elevation(region)
wind_u, wind_v = get_wind(date, region)
humidity = get_humidity(date, region)

# Example matrices
print(fire_state.shape)
print(elevation.shape)
print(wind_u.shape)
print(wind_v.shape)
print(humidity.shape)
