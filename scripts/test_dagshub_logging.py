"""
Test script to verify Dagshub MLflow logging works
Run this to manually create test runs in Dagshub
"""
import os
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import dagshub

load_dotenv()

# Get Dagshub credentials
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "")

print("=" * 80)
print("Testing Dagshub MLflow Logging")
print("=" * 80)
print(f"Username: {DAGSHUB_USERNAME}")
print(f"Repo: {DAGSHUB_REPO}")
print(f"Token: {'*' * 10 if DAGSHUB_TOKEN else 'NOT SET'}")

if not DAGSHUB_USERNAME or not DAGSHUB_REPO or not DAGSHUB_TOKEN:
    print("\n❌ ERROR: Dagshub credentials not set!")
    print("Please set the following environment variables:")
    print("  - DAGSHUB_USERNAME")
    print("  - DAGSHUB_REPO")
    print("  - DAGSHUB_TOKEN")
    exit(1)

# Initialize Dagshub
try:
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True, token=DAGSHUB_TOKEN)
    print("\n✅ Dagshub initialized")
except Exception as e:
    print(f"\n⚠️  Dagshub init warning: {e}")

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = f"https://{DAGSHUB_USERNAME}:{DAGSHUB_TOKEN}@dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
print(f"MLflow URI: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("weather_forecasting")
    
    print("\n" + "=" * 80)
    print("Test 1: Creating a data profiling run")
    print("=" * 80)
    
    # Create sample data for profiling
    np.random.seed(42)
    df = pd.DataFrame({
        'temp': np.random.normal(20, 5, 100),
        'humidity': np.random.uniform(30, 90, 100),
        'pressure': np.random.normal(1013, 10, 100),
        'wind_speed': np.random.exponential(5, 100)
    })
    
    with mlflow.start_run(run_name=f"data_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("data_shape", f"{df.shape[0]}x{df.shape[1]}")
        mlflow.log_param("collection_date", datetime.now().isoformat())
        mlflow.log_param("total_columns", len(df.columns))
        mlflow.log_param("missing_values", int(df.isnull().sum().sum()))
        mlflow.log_param("report_type", "test_profiling")
        
        # Log metrics
        mlflow.log_metric("data_quality_score", 1.0)
        mlflow.log_metric("mean_temp", float(df['temp'].mean()))
        mlflow.log_metric("mean_humidity", float(df['humidity'].mean()))
        
        # Create a simple text report
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write("=" * 80 + "\n")
            tmp.write("TEST DATA PROFILE REPORT\n")
            tmp.write("=" * 80 + "\n\n")
            tmp.write(f"Data Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
            tmp.write(f"Collection Time: {datetime.now().isoformat()}\n\n")
            tmp.write("-" * 80 + "\n")
            tmp.write("SUMMARY STATISTICS\n")
            tmp.write("-" * 80 + "\n")
            tmp.write(f"{df.describe().to_string()}\n\n")
            report_path = tmp.name
        
        # Log artifact
        mlflow.log_artifact(report_path, "data_profiling")
        os.unlink(report_path)
        
        print("✅ Data profiling run created successfully!")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")
    
    print("\n" + "=" * 80)
    print("Test 2: Creating a model training run")
    print("=" * 80)
    
    with mlflow.start_run(run_name=f"weather_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("model_type", "random_forest")
        mlflow.log_param("n_features", 4)
        mlflow.log_param("train_size", 80)
        mlflow.log_param("val_size", 20)
        mlflow.log_param("forecast_hours", 4)
        
        # Log metrics (simulated)
        mlflow.log_metric("train_rmse", 2.5)
        mlflow.log_metric("val_rmse", 2.8)
        mlflow.log_metric("train_mae", 1.9)
        mlflow.log_metric("val_mae", 2.1)
        mlflow.log_metric("train_r2", 0.85)
        mlflow.log_metric("val_r2", 0.82)
        
        print("✅ Model training run created successfully!")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")
    
    print("\n" + "=" * 80)
    print("✅ SUCCESS: Both test runs created!")
    print("=" * 80)
    print(f"\nView your runs at:")
    print(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
    print(f"\nLook for:")
    print(f"  - Runs named 'data_profile_YYYYMMDD_HHMMSS'")
    print(f"  - Runs named 'weather_model_YYYYMMDD_HHMMSS'")
    
except Exception as e:
    print(f"\n❌ ERROR: Failed to log to Dagshub: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


