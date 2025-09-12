# fastapi_lstm_weather.py

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Dict
import requests
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime, timedelta
import io
import base64
import matplotlib.pyplot as plt

# import your analyze_farmland function
from services.msi_analyze import analyze_farmland
import joblib

app = FastAPI()

# Load trained LSTM model for temperature & rainfall
weather_model = load_model("models/yearly_weather_lstm_model.h5", compile=False)

# Load soil temperature linear model
soil_model = joblib.load("models/soil_temp_linear_model.pkl")
soil_scaler = joblib.load("models/soil_temp_scaler.pkl")

# Define input schema
class GeoPoints(BaseModel):
    points: list  # List of 4 points, each as [latitude, longitude]

# Function to compute center coordinate
def get_center_coordinate(points):
    lats = [float(pt[0]) for pt in points]
    lons = [float(pt[1]) for pt in points]
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
    data = df[['temperature_2m (°C)', 'rain (mm)']].values
    scaled_data = scaler.transform(data)
    X_input = scaled_data[-time_step:].reshape(1, time_step, 2)
    return X_input

@app.post("/predict_weather")
def predict_weather(geo: GeoPoints):
    # Compute center coordinate
    center_lat, center_lon = get_center_coordinate(geo.points)
    
    # Historical dates for last 30 days
    today = datetime.utcnow().date()
    start_date = today - timedelta(days=30)
    
    # Fetch historical weather
    df_weather = fetch_historical_weather(center_lat, center_lon, start_date.isoformat(), today.isoformat())
    df_weather.dropna(inplace=True)
    
    # Scale using MinMaxScaler for LSTM
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(df_weather[['temperature_2m (°C)','rain (mm)']])
    
    # Prepare last 30 days sequence
    last_sequence = preprocess_for_lstm(df_weather, scaler, time_step=30)
    
    # Predict next 7 days
    predictions = []
    current_seq = last_sequence.copy()
    for _ in range(7):
        pred_7 = weather_model.predict(current_seq)
        pred_7 = pred_7.reshape((1,7,2))
        day_pred = pred_7[0,0,:]
        predictions.append(day_pred)
        current_seq = np.concatenate([current_seq[:,1:,:], day_pred.reshape(1,1,2)], axis=1)
    predictions = np.array(predictions)
    predictions_inv = scaler.inverse_transform(predictions)
    
    # Predict soil temperature using soil model
    soil_preds = []
    for day in predictions_inv:
        soil_input = soil_scaler.transform([day])  # [rain_mm, temperature]
        soil_temp = soil_model.predict(soil_input)
        soil_preds.append(soil_temp[0])
    soil_preds = np.array(soil_preds)
    
    # Dates for next 7 days
    date_range = [today + timedelta(days=i+1) for i in range(7)]
    
    # Plot results
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(date_range, predictions_inv[:,1], label="2m Temp (°C)", marker='o')
    ax.plot(date_range, predictions_inv[:,0], label="Rainfall (mm)", marker='o')
    ax.plot(date_range, soil_preds, label="Soil Temp (°C)", marker='o')
    ax.set_title("Next 7 Days Weather Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    
    # Convert plot to base64
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # Return numeric predictions + plot
    return {
        "dates": [d.isoformat() for d in date_range],
        "temperature_2m_C": predictions_inv[:,1].tolist(),
        "rain_mm": predictions_inv[:,0].tolist(),
        "soil_temperature_C": soil_preds.tolist(),
        "plot_base64": plot_base64
    }


# ---------- Existing GET /analyze stays unchanged ----------
class FarmlandResponse(BaseModel):
    timestamp: str
    images: Dict[str, str]

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
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(name, rotation=270, labelpad=15)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        images[name] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
    return images

@app.get("/analyze", response_model=FarmlandResponse)
def analyze(
    min_lon: float = Query(...),
    min_lat: float = Query(...),
    max_lon: float = Query(...),
    max_lat: float = Query(...),
    start_date: str = Query("2024-07-01"),
    end_date: str = Query("2024-07-15")
):
    coords = [min_lon, min_lat, max_lon, max_lat]
    time_range = (start_date, end_date)
    result = analyze_farmland(coords, time_range)
    if "error" in result:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=result["error"])
    timestamp = result["timestamp"]
    indices_data = np.stack([
        result["NDVI"], result["EVI"], result["SAVI"], result["NDRE"],
        result["MSI"], result["NDMI"], result["SOC"], result["OM"]
    ], axis=-1)
    images = indices_to_images(indices_data, timestamp)
    return FarmlandResponse(timestamp=timestamp, images=images)
