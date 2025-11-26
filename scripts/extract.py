"""
Data Extraction Script for OpenWeatherMap API
Fetches current weather data and historical forecast data
"""
import os
import json
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
BASE_URL = "http://api.openweathermap.org/data/2.5"
CITY = "London"  # Default city, can be parameterized
COUNTRY_CODE = "GB"


def fetch_current_weather(city: str = CITY, country_code: str = COUNTRY_CODE) -> dict:
    """Fetch current weather data from OpenWeatherMap API"""
    url = f"{BASE_URL}/weather"
    params = {
        "q": f"{city},{country_code}",
        "appid": API_KEY,
        "units": "metric"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch current weather: {str(e)}")


def fetch_forecast(city: str = CITY, country_code: str = COUNTRY_CODE) -> dict:
    """Fetch 5-day forecast data from OpenWeatherMap API"""
    url = f"{BASE_URL}/forecast"
    params = {
        "q": f"{city},{country_code}",
        "appid": API_KEY,
        "units": "metric"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch forecast: {str(e)}")


def extract_weather_data(city: str = CITY, country_code: str = COUNTRY_CODE) -> pd.DataFrame:
    """
    Extract and combine current weather and forecast data
    Returns a DataFrame with weather features
    """
    current = fetch_current_weather(city, country_code)
    forecast = fetch_forecast(city, country_code)
    
    # Extract current weather features
    current_data = {
        "timestamp": datetime.now().isoformat(),
        "collection_time": datetime.now().isoformat(),
        "city": city,
        "country": country_code,
        "temp": current["main"]["temp"],
        "feels_like": current["main"]["feels_like"],
        "temp_min": current["main"]["temp_min"],
        "temp_max": current["main"]["temp_max"],
        "pressure": current["main"]["pressure"],
        "humidity": current["main"]["humidity"],
        "wind_speed": current["wind"].get("speed", 0),
        "wind_deg": current["wind"].get("deg", 0),
        "clouds": current["clouds"]["all"],
        "visibility": current.get("visibility", 10000),
        "weather_main": current["weather"][0]["main"],
        "weather_description": current["weather"][0]["description"],
        "is_current": True
    }
    
    # Extract forecast data
    forecast_records = []
    for item in forecast.get("list", [])[:10]:  # Get next 10 forecast points
        forecast_data = {
            "timestamp": datetime.fromtimestamp(item["dt"]).isoformat(),
            "collection_time": datetime.now().isoformat(),
            "city": city,
            "country": country_code,
            "temp": item["main"]["temp"],
            "feels_like": item["main"]["feels_like"],
            "temp_min": item["main"]["temp_min"],
            "temp_max": item["main"]["temp_max"],
            "pressure": item["main"]["pressure"],
            "humidity": item["main"]["humidity"],
            "wind_speed": item["wind"].get("speed", 0),
            "wind_deg": item["wind"].get("deg", 0),
            "clouds": item["clouds"]["all"],
            "visibility": item.get("visibility", 10000),
            "weather_main": item["weather"][0]["main"],
            "weather_description": item["weather"][0]["description"],
            "is_current": False
        }
        forecast_records.append(forecast_data)
    
    # Combine current and forecast data
    all_data = [current_data] + forecast_records
    df = pd.DataFrame(all_data)
    
    # Convert timestamp to datetime (handle ISO format without microseconds)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed', errors='coerce')
    df["collection_time"] = pd.to_datetime(df["collection_time"], format='mixed', errors='coerce')
    
    return df


def save_raw_data(df: pd.DataFrame, output_dir: str = "data/raw") -> str:
    """Save raw extracted data with timestamp"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"weather_data_{timestamp}.parquet"
    filepath = os.path.join(output_dir, filename)
    
    df.to_parquet(filepath, index=False)
    
    # Also save metadata
    metadata = {
        "collection_time": datetime.now().isoformat(),
        "records_count": len(df),
        "columns": list(df.columns),
        "filepath": filepath
    }
    
    metadata_path = os.path.join(output_dir, f"metadata_{timestamp}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return filepath


if __name__ == "__main__":
    print("Starting data extraction...")
    
    if not API_KEY:
        raise ValueError("OPENWEATHER_API_KEY not found in environment variables")
    
    # Extract data
    df = extract_weather_data()
    print(f"Extracted {len(df)} records")
    
    # Save raw data
    filepath = save_raw_data(df)
    print(f"Saved raw data to: {filepath}")
    
    print("Data extraction completed successfully!")

