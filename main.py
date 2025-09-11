# fastapi_lstm_weather.py

from fastapi import FastAPI
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

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
