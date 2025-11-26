"""
FastAPI Prediction Service with Prometheus Metrics
Serves weather forecasting predictions via REST API
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response, JSONResponse
import mlflow
MLFLOW_AVAILABLE = True
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

app = FastAPI(title="Weather Forecasting API", version="1.0.0")

# Prometheus Metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint"]
)

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request latency in seconds",
    ["method", "endpoint"]
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total number of predictions made"
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors"
)

DATA_DRIFT_RATIO = Gauge(
    "data_drift_ratio",
    "Ratio of requests with out-of-distribution features"
)

MODEL_VERSION = Gauge(
    "model_version",
    "Current model version"
)

# Global model and feature columns
model = None
feature_columns = None
model_loaded = False


class PredictionRequest(BaseModel):
    temp: float
    humidity: float
    pressure: float
    wind_speed: float
    wind_deg: float = 0.0
    clouds: float = 0.0
    hour: int = None
    day_of_week: int = None
    month: int = None


def load_model():
    """Load the trained model and feature columns"""
    global model, feature_columns, model_loaded
    
    if model_loaded and model is not None and feature_columns is not None:
        return
    
    try:
        # Try to load from local models directory
        model_path = Path("models/weather_model.joblib")
        feature_path = Path("models/feature_columns.joblib")
        
        if model_path.exists() and feature_path.exists():
            model = joblib.load(model_path)
            feature_columns = joblib.load(feature_path)
            print(f"✓ Model loaded from local path: {model_path}")
            print(f"✓ Feature columns loaded: {len(feature_columns)} columns")
        else:
            # Try to load from MLflow (if available)
            if not MLFLOW_AVAILABLE:
                raise Exception("MLflow not available. Please train a model first or install MLflow.")
            
            MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
            # Get latest model from MLflow
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name("weather_forecasting")
            
            if experiment:
                runs = client.search_runs(
                    experiment.experiment_id,
                    order_by=["metrics.val_rmse ASC"],
                    max_results=1
                )
                
                if runs:
                    run_id = runs[0].info.run_id
                    model_uri = f"runs:/{run_id}/model"
                    model = mlflow.sklearn.load_model(model_uri)
                    print(f"✓ Model loaded from MLflow run: {run_id}")
                else:
                    raise Exception("No model runs found in MLflow")
            else:
                raise Exception("MLflow experiment not found")
        
        model_loaded = True
        MODEL_VERSION.set(1.0)
        print("✓ Model loaded successfully")
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        # Don't raise - allow service to start without model
        model_loaded = False


def create_features(request: PredictionRequest) -> pd.DataFrame:
    """Create feature vector from request"""
    now = datetime.now()
    
    # Use provided time features or current time
    hour = request.hour if request.hour is not None else now.hour
    day_of_week = request.day_of_week if request.day_of_week is not None else now.weekday()
    month = request.month if request.month is not None else now.month
    
    # Create base features
    features = {
        "temp": request.temp,
        "feels_like": request.temp,  # Approximate
        "temp_min": request.temp - 2,  # Approximate
        "temp_max": request.temp + 2,  # Approximate
        "pressure": request.pressure,
        "humidity": request.humidity,
        "wind_speed": request.wind_speed,
        "wind_deg": request.wind_deg if request.wind_deg is not None else 0,
        "clouds": request.clouds if request.clouds is not None else 0,
        "visibility": 10000,  # Default
        "hour": hour,
        "day_of_week": day_of_week,
        "day_of_month": now.day,
        "month": month,
        "is_weekend": 1 if day_of_week >= 5 else 0,
    }
    
    # Note: Weather dummy variables are NOT included because the model
    # was trained without them (model expects 42 features, not 44)
    # If weather features are needed, retrain the model with them included
    
    # Cyclical encoding
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    
    # Derived features
    features["temp_range"] = 4  # Approximate
    features["temp_feels_diff"] = 0
    features["heat_index"] = 0.5 * (request.temp + 61.0 + ((request.temp - 68.0) * 1.2) + (request.humidity * 0.094))
    features["wind_chill"] = 13.12 + 0.6215 * request.temp - 11.37 * (request.wind_speed ** 0.16) + 0.3965 * request.temp * (request.wind_speed ** 0.16)
    
    # Lag features (use current temp as approximation)
    for lag in [1, 2, 3, 6, 12]:
        features[f"temp_lag_{lag}"] = request.temp
    
    # Rolling features (use current temp as approximation)
    for window in [3, 6, 12]:
        features[f"temp_rolling_mean_{window}"] = request.temp
        features[f"temp_rolling_std_{window}"] = 0
        features[f"temp_rolling_min_{window}"] = request.temp
        features[f"temp_rolling_max_{window}"] = request.temp
    
    # Pressure change (approximate)
    features["pressure_change"] = 0
    features["pressure_change_pct"] = 0
    
    # Create DataFrame
    df = pd.DataFrame([features])
    
    # Add weather dummy variables if they're expected by the model
    if feature_columns:
        # Check for weather_ prefix columns in feature_columns
        weather_cols = [col for col in feature_columns if col.startswith('weather_')]
        for col in weather_cols:
            if col not in df.columns:
                df[col] = 0
    
    # Use model's expected features if available (most accurate)
    if model is not None and hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
        expected_features = list(model.feature_names_in_)
        # Ensure all expected features exist
        for feat in expected_features:
            if feat not in df.columns:
                df[feat] = 0
        # Select in exact order model expects
        df = df[expected_features]
    elif feature_columns is not None and len(feature_columns) > 0:
        # Fallback to feature_columns file
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]
    else:
        print("⚠️  Warning: No feature specification available")
    
    return df


def check_data_drift(features: pd.DataFrame) -> bool:
    """Simple data drift detection - check if features are out of expected ranges"""
    # Define expected ranges
    ranges = {
        "temp": (-50, 60),
        "humidity": (0, 100),
        "pressure": (800, 1100),
        "wind_speed": (0, 200),
    }
    
    drift_detected = False
    for col, (min_val, max_val) in ranges.items():
        if col in features.columns:
            if (features[col] < min_val).any() or (features[col] > max_val).any():
                drift_detected = True
                break
    
    return drift_detected


@app.on_event("startup")
async def startup_event():
    """Load model on startup (optional)"""
    try:
        load_model()
    except Exception as e:
        print(f"⚠️  Model not loaded: {str(e)}")
        print("⚠️  Service will start but predictions won't work until model is available")
        global model_loaded
        model_loaded = False


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make a weather forecast prediction"""
    if not model_loaded:
        # Try to load model if not loaded
        try:
            load_model()
        except:
            pass
        
        if not model_loaded:
            raise HTTPException(
                status_code=503, 
                detail="Model not loaded. Please train a model first or ensure model files are available."
            )
    
    # Track request
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    
    with REQUEST_LATENCY.labels(method="POST", endpoint="/predict").time():
        try:
            # Create features
            features = create_features(request)
            
            # Check for data drift
            drift_detected = check_data_drift(features)
            if drift_detected:
                DATA_DRIFT_RATIO.set(1.0)
            else:
                DATA_DRIFT_RATIO.set(0.0)
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            PREDICTION_COUNT.inc()
            
            return {
                "predicted_temp": float(prediction),
                "prediction": float(prediction),  # Keep both for compatibility
                "forecast_hours": 4,
                "timestamp": datetime.now().isoformat(),
                "data_drift_detected": drift_detected
            }
            
        except Exception as e:
            PREDICTION_ERRORS.inc()
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Weather Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

