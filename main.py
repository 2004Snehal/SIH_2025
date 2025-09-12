# fastapi_lstm_weather.py

from fastapi import FastAPI
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict
import datetime
import io
import base64
import matplotlib.pyplot as plt

# import your analyze_farmland function
from services.msi_analyze import analyze_farmland
app = FastAPI()

# Load trained LSTM model
model = load_model("models/weather_lstm_model.h5", compile=False)

# Define input schema
class GeoPoints(BaseModel):
    points: list  # List of 4 points, each as [latitude, longitude]

# Function to compute center coordinate
def get_center_coordinate(points):
    lats = [pt[0] for pt in points]
    lons = [pt[1] for pt in points]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    return center_lat, center_lon

# Function to fetch historical weather data from Open-Meteo
def fetch_historical_weather(lat, lon, start_date, end_date):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,rain,soil_temperature_18cm&start_date={start_date}&end_date={end_date}"
    response = requests.get(url)
    data = response.json()
    
    df = pd.DataFrame({
        "time": data['hourly']['time'],
        "rain (mm)": data['hourly']['rain'],
        "temperature_2m (°C)": data['hourly']['temperature_2m'],
        "soil_temperature_18cm (°C)": data['hourly']['soil_temperature_18cm']
    })
    df['time'] = pd.to_datetime(df['time'])
    return df

# Preprocess data for LSTM
def preprocess_for_lstm(df, scaler, time_step=30):
    data = df[['rain (mm)', 'temperature_2m (°C)', 'soil_temperature_18cm (°C)']].values
    scaled_data = scaler.transform(data)
    # Take the last 'time_step' rows
    X_input = scaled_data[-time_step:]
    X_input = X_input.reshape(1, time_step, 3)
    return X_input

@app.post("/predict_weather")
def predict_weather(geo: GeoPoints):
    # Compute center coordinate
    center_lat, center_lon = get_center_coordinate(geo.points)
    
    # Historical dates for last 30 days
    today = datetime.utcnow().date()
    start_date = today - timedelta(days=30)
    start_date_str = start_date.isoformat()
    end_date_str = today.isoformat()
    
    # Fetch historical weather
    df_weather = fetch_historical_weather(center_lat, center_lon, start_date_str, end_date_str)
    
    # Drop missing values if any
    df_weather.dropna(inplace=True)
    
    # Scale using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_weather[['rain (mm)', 'temperature_2m (°C)', 'soil_temperature_18cm (°C)']])
    
    # Prepare LSTM input
    time_step = 30
    X_input = preprocess_for_lstm(df_weather, scaler, time_step)
    
    # Predict next time step
    pred_scaled = model.predict(X_input)
    pred = scaler.inverse_transform(pred_scaled)
    
    return {
        "predicted_rain_mm": float(pred[0,0]),
        "predicted_temperature_2m_C": float(pred[0,1]),
        "predicted_soil_temperature_18cm_C": float(pred[0,2]),
    }


class FarmlandResponse(BaseModel):
    timestamp: str
    images: Dict[str, str]  # Dictionary of base64 encoded images

def indices_to_images(indices_data, best_timestamp):
    """Convert indices data to base64 encoded images"""
    index_info = {
        'NDVI': {'data': indices_data[..., 0], 'cmap': 'RdYlGn', 'vmin': 0, 'vmax': 1, 'label': 'Vegetation Health'},
        'EVI':  {'data': indices_data[..., 1], 'cmap': 'RdYlGn', 'vmin': 0, 'vmax': 1, 'label': 'Vegetation Density'},
        'SAVI': {'data': indices_data[..., 2], 'cmap': 'RdYlGn', 'vmin': 0, 'vmax': 1, 'label': 'Soil-Adjusted Veg.'},
        'NDRE': {'data': indices_data[..., 3], 'cmap': 'viridis', 'vmin': 0, 'vmax': 0.8, 'label': 'Chlorophyll Content'},
        'MSI':  {'data': indices_data[..., 4], 'cmap': 'RdYlGn_r','vmin': 0.3, 'vmax': 2.0, 'label': 'Moisture Stress'},
        'NDMI': {'data': indices_data[..., 5], 'cmap': 'BrBG_r',  'vmin': -0.5, 'vmax': 0.5, 'label': 'Vegetation Moisture'},
        'SOC':  {'data': indices_data[..., 6], 'cmap': 'YlOrBr',  'vmin': 0, 'vmax': 5, 'label': 'Est. SOC (%)'},
        'OM':   {'data': indices_data[..., 7], 'cmap': 'YlOrBr',  'vmin': 0, 'vmax': 8.6, 'label': 'Est. OM (%)'}
    }
    
    images = {}
    for name, info in index_info.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = plt.get_cmap(info['cmap']).copy()
        if name in ['SOC', 'OM']:
            cmap.set_bad('lightgray', 1.)
        
        im = ax.imshow(info['data'], cmap=cmap, vmin=info['vmin'], vmax=info['vmax'])
        ax.set_title(f'{name}\n({info["label"]})', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(name, rotation=270, labelpad=15)
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        images[name] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
    
    return images

@app.get("/analyze", response_model=FarmlandResponse)
def analyze(
    min_lon: float = Query(..., description="Minimum longitude"),
    min_lat: float = Query(..., description="Minimum latitude"),
    max_lon: float = Query(..., description="Maximum longitude"),
    max_lat: float = Query(..., description="Maximum latitude"),
    start_date: str = Query("2025-07-01", description="Start date (YYYY-MM-DD)"),
    end_date: str = Query("2025-07-15", description="End date (YYYY-MM-DD)")
):
    """
    Analyze farmland health and soil indices for a given bounding box.
    Returns vegetation indices and soil organic content as base64 encoded images.
    """

    # time range: last `days_back` days till today
    # today = datetime.date.today()
    # start_date = (today - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
    # end_date = today.strftime("%Y-%m-%d")
    # time_range = (start_date, end_date)
    time_range = (start_date,end_date)

    coords = [min_lon, min_lat, max_lon, max_lat]

    result = analyze_farmland(coords, time_range)

    if "error" in result:
        # Raise a 404 if no data found
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=result["error"])

    # Extract indices data and timestamp
    timestamp = result["timestamp"]
    indices_data = np.stack([
        result["NDVI"], result["EVI"], result["SAVI"], result["NDRE"],
        result["MSI"], result["NDMI"], result["SOC"], result["OM"]
    ], axis=-1)

    # Generate images
    images = indices_to_images(indices_data, timestamp)

    return FarmlandResponse(timestamp=timestamp, images=images)