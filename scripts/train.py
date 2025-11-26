"""
Model Training Script with MLflow Tracking
Trains a time-series forecasting model for weather prediction
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv
import dagshub

load_dotenv()

# Configure MLflow and Dagshub
# Default to Docker service name, fallback to localhost for local runs
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "")

# Force MLflow to use server (Docker service)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

MLFLOW_AVAILABLE = True
try:
    # Try Dagshub first if configured
    if DAGSHUB_USERNAME and DAGSHUB_REPO and DAGSHUB_TOKEN:
        try:
            # Use token in MLflow URI for authentication (format: https://USERNAME:TOKEN@dagshub.com/...)
            dagshub_uri = os.getenv("MLFLOW_TRACKING_URI", f"https://{DAGSHUB_USERNAME}:{DAGSHUB_TOKEN}@dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
            MLFLOW_TRACKING_URI = dagshub_uri
            print(f"✅ Using Dagshub MLflow: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
            print(f"✅ Authentication configured with token")
        except Exception as e:
            print(f"⚠️  Dagshub configuration failed: {e}, using MLflow server")
            # Continue with server URI
    
    # Always try MLflow server
    print(f"Connecting to MLflow at: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Test connection by setting/getting experiment
    try:
        mlflow.set_experiment("weather_forecasting")
        # Verify connection works
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("weather_forecasting")
        print(f"✅ Connected to MLflow server successfully!")
        print(f"✅ Experiment ID: {exp.experiment_id}")
        MLFLOW_AVAILABLE = True
    except Exception as e:
        print(f"❌ Failed to connect to MLflow server: {e}")
        print(f"❌ URI attempted: {MLFLOW_TRACKING_URI}")
        print("❌ MLflow logging will be disabled")
        MLFLOW_AVAILABLE = False
        raise  # Re-raise to prevent fallback to file-based
        
except Exception as e:
    print(f"❌ MLflow setup failed: {e}")
    print("❌ Continuing without MLflow tracking")
    MLFLOW_AVAILABLE = False


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare feature and target arrays"""
    # Exclude non-feature columns
    exclude_cols = [
        "timestamp", "collection_time", "city", "country",
        "weather_main", "weather_description", "target_temp",
        "is_current"
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Handle categorical columns
    if "weather_main" in df.columns:
        weather_dummies = pd.get_dummies(df["weather_main"], prefix="weather")
        feature_cols.extend(weather_dummies.columns)
        df = pd.concat([df, weather_dummies], axis=1)
    
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df["target_temp"].fillna(df["temp"].mean())
    
    return X, y, feature_cols


def train_model(X_train, y_train, X_val, y_val, model_type: str = "random_forest"):
    """Train a model and return it with metrics"""
    
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    metrics = {
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "train_r2": train_r2,
        "val_r2": val_r2
    }
    
    return model, metrics


def train_pipeline(data_path: str, model_type: str = "random_forest"):
    """Complete training pipeline with MLflow tracking"""
    
    print(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Data shape: {df.shape}")
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    print(f"Feature count: {len(feature_cols)}")
    print(f"Sample count: {len(X)}")
    
    # Time-series split (use last 20% for validation)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    
    # Start MLflow run (if available)
    if not MLFLOW_AVAILABLE:
        print("⚠️  MLflow not available, training without tracking")
        # Train model without MLflow
        print("Training model...")
        model, metrics = train_model(X_train, y_train, X_val, y_val, model_type)
        
        print("Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save model locally
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "weather_model.joblib"
        joblib.dump(model, model_path)
        
        # Save feature columns for inference
        feature_cols_path = model_dir / "feature_columns.joblib"
        joblib.dump(feature_cols, feature_cols_path)
        
        print(f"\n✓ Model training completed!")
        print(f"  Model saved to: {model_path}")
        return model, metrics, "local"
    
    try:
        with mlflow.start_run(run_name=f"weather_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            mlflow.log_param("forecast_hours", 4)
            
            # Train model
            print("Training model...")
            model, metrics = train_model(X_train, y_train, X_val, y_val, model_type)
            
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            print("Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # Log model using MLflow's sklearn flavor (handles permissions correctly)
            try:
                # Use MLflow's built-in model logging (works via API, no file system permissions needed)
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name="weather_forecast_model"
                )
                print("✅ Model logged to MLflow using sklearn flavor")
            except Exception as e:
                print(f"⚠️  Sklearn model logging failed: {e}")
                # Fallback: Try simple artifact logging
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='wb', suffix='.joblib', delete=False) as tmp:
                        joblib.dump(model, tmp.name)
                        mlflow.log_artifact(tmp.name, "model")
                        os.unlink(tmp.name)
                    print("✅ Model logged as artifact (fallback)")
                except Exception as e2:
                    print(f"⚠️  Model artifact logging failed (non-critical): {e2}")
                    # Continue - metrics and params are more important
            
            # Log feature columns as artifact
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.joblib', delete=False) as tmp:
                    joblib.dump(feature_cols, tmp.name)
                    mlflow.log_artifact(tmp.name, "model")
                    os.unlink(tmp.name)
                print("✅ Feature columns logged to MLflow")
            except Exception as e:
                print(f"⚠️  Feature columns logging failed (non-critical): {e}")
            
            # Log feature importance
            if hasattr(model, "feature_importances_"):
                try:
                    feature_importance = pd.DataFrame({
                        "feature": feature_cols[:len(model.feature_importances_)],
                        "importance": model.feature_importances_
                    }).sort_values("importance", ascending=False)
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                        importance_path = tmp.name
                        feature_importance.to_csv(importance_path, index=False)
                        mlflow.log_artifact(importance_path, "model")
                        os.unlink(importance_path)
                    print("✅ Feature importance logged to MLflow")
                except Exception as e:
                    print(f"⚠️  Feature importance logging failed: {e}")
            
            # Save model locally
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / "weather_model.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path))
            
            # Save feature columns for inference
            feature_cols_path = model_dir / "feature_columns.joblib"
            joblib.dump(feature_cols, feature_cols_path)
            
            print(f"\n✓ Model training completed!")
            print(f"  Model saved to: {model_path}")
            print(f"  MLflow run ID: {mlflow.active_run().info.run_id}")
            
            return model, metrics, mlflow.active_run().info.run_id
    except Exception as e:
        print(f"⚠️  MLflow logging failed: {e}")
        print("⚠️  Saving model locally without MLflow tracking")
        # Train model without MLflow
        print("Training model...")
        model, metrics = train_model(X_train, y_train, X_val, y_val, model_type)
        
        print("Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save model locally
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "weather_model.joblib"
        joblib.dump(model, model_path)
        
        # Save feature columns for inference
        feature_cols_path = model_dir / "feature_columns.joblib"
        joblib.dump(feature_cols, feature_cols_path)
        
        print(f"\n✓ Model training completed!")
        print(f"  Model saved to: {model_path}")
        return model, metrics, "local"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <processed_data_path> [model_type]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "random_forest"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
    try:
        train_pipeline(data_path, model_type)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

