#!/usr/bin/env python3
"""
Real-time Weather Prediction Client
Interacts with the FastAPI prediction service programmatically
"""
import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, Optional

class WeatherPredictionClient:
    """Client for interacting with Weather Forecasting API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check if the API is healthy and model is loaded"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "unhealthy"}
    
    def predict(
        self,
        temp: float,
        humidity: float,
        pressure: float,
        wind_speed: float,
        wind_deg: Optional[float] = None,
        clouds: Optional[float] = None,
        hour: Optional[int] = None,
        day_of_week: Optional[int] = None,
        month: Optional[int] = None
    ) -> Dict:
        """
        Make a weather prediction
        
        Args:
            temp: Temperature in Celsius
            humidity: Humidity percentage (0-100)
            pressure: Atmospheric pressure in hPa
            wind_speed: Wind speed in m/s
            wind_deg: Wind direction in degrees (optional)
            clouds: Cloud coverage percentage (optional)
            hour: Hour of day (0-23, optional)
            day_of_week: Day of week (0-6, optional)
            month: Month (1-12, optional)
        
        Returns:
            Prediction result with forecasted temperature
        """
        payload = {
            "temp": temp,
            "humidity": humidity,
            "pressure": pressure,
            "wind_speed": wind_speed
        }
        
        if wind_deg is not None:
            payload["wind_deg"] = wind_deg
        if clouds is not None:
            payload["clouds"] = clouds
        if hour is not None:
            payload["hour"] = hour
        if day_of_week is not None:
            payload["day_of_week"] = day_of_week
        if month is not None:
            payload["month"] = month
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=30
            )
            elapsed = time.time() - start_time
            response.raise_for_status()
            
            result = response.json()
            result["inference_time_ms"] = round(elapsed * 1000, 2)
            return result
        except requests.exceptions.Timeout:
            return {"error": "Request timed out", "status": "error"}
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "error"}
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics"""
        try:
            response = self.session.get(f"{self.base_url}/metrics", timeout=5)
            return response.text
        except Exception as e:
            return f"Error: {e}"


def main():
    """Example usage"""
    client = WeatherPredictionClient()
    
    # Check health
    print("🔍 Checking API health...")
    health = client.health_check()
    print(f"Status: {health.get('status', 'unknown')}")
    print(f"Model loaded: {health.get('model_loaded', False)}")
    print()
    
    if not health.get('model_loaded', False):
        print("⚠️  Model not loaded. Please ensure the DAG has completed training.")
        print("   You can restart FastAPI: docker compose restart fastapi-service")
        return
    
    # Make predictions
    print("🌡️  Making weather predictions...")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Normal Day",
            "temp": 20.5,
            "humidity": 65.0,
            "pressure": 1013.25,
            "wind_speed": 5.2
        },
        {
            "name": "Hot Day",
            "temp": 35.0,
            "humidity": 45.0,
            "pressure": 1015.0,
            "wind_speed": 8.0
        },
        {
            "name": "Cold Day",
            "temp": 5.0,
            "humidity": 80.0,
            "pressure": 1020.0,
            "wind_speed": 3.0
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['name']}:")
        print(f"   Input: {test['temp']}°C, {test['humidity']}% humidity, {test['pressure']} hPa, {test['wind_speed']} m/s wind")
        
        result = client.predict(**{k: v for k, v in test.items() if k != 'name'})
        
        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            predicted_temp = result.get('predicted_temp', 'N/A')
            inference_time = result.get('inference_time_ms', 'N/A')
            print(f"   ✅ Predicted temperature: {predicted_temp}°C")
            print(f"   ⏱️  Inference time: {inference_time}ms")
    
    print("\n" + "=" * 60)
    print("✅ Predictions completed!")


if __name__ == "__main__":
    main()

