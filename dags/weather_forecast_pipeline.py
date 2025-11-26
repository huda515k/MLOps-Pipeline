"""
Apache Airflow DAG for Weather Forecasting MLOps Pipeline
Runs daily ETL and model retraining workflow
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import sys
import os

# Add scripts directory to path
sys.path.append("/opt/airflow/scripts")

default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "weather_forecast_pipeline",
    default_args=default_args,
    description="Complete MLOps pipeline for weather forecasting",
    schedule_interval=timedelta(days=1),  # Run daily
    start_date=days_ago(1),
    catchup=False,
    tags=["mlops", "weather", "forecasting"],
)


def extract_data(**context):
    """Extract weather data from API"""
    from extract import extract_weather_data, save_raw_data
    
    print("Starting data extraction...")
    df = extract_weather_data()
    filepath = save_raw_data(df)
    
    # Push filepath to XCom for next task
    context["ti"].xcom_push(key="raw_data_path", value=filepath)
    return filepath


def check_quality(**context):
    """Run data quality checks"""
    from quality_check import check_data_quality
    import pandas as pd
    
    # Pull filepath from previous task
    filepath = context["ti"].xcom_pull(key="raw_data_path", task_ids="extract_data")
    
    print(f"Running quality checks on: {filepath}")
    df = pd.read_parquet(filepath)
    
    # Run quality checks (will raise exception if fails)
    results = check_data_quality(df, strict=True)
    
    print("✓ All quality checks passed!")
    return results


def transform_data_task(**context):
    """Transform and engineer features"""
    from transform import transform_data as transform_func, save_processed_data
    import pandas as pd
    
    # Pull filepath from previous task
    filepath = context["ti"].xcom_pull(key="raw_data_path", task_ids="extract_data")
    
    print(f"Transforming data from: {filepath}")
    df = pd.read_parquet(filepath)
    
    # Transform
    df_transformed = transform_func(df, forecast_hours=4)
    
    # Save processed data
    processed_path = save_processed_data(df_transformed)
    
    # Push to XCom
    context["ti"].xcom_push(key="processed_data_path", value=processed_path)
    return processed_path


def generate_profiling_report(**context):
    """Generate pandas profiling report and log to MLflow"""
    import pandas as pd
    import mlflow
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Pull processed data path
    filepath = context["ti"].xcom_pull(key="processed_data_path", task_ids="transform_data")
    
    print(f"Generating profiling report for: {filepath}")
    
    try:
        df = pd.read_parquet(filepath)
        print(f"Data loaded: {df.shape}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        raise
    
    # Create comprehensive data profile report (pandas-profiling has compatibility issues)
    report_path = "/tmp/data_profile.txt"
    try:
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("WEATHER DATA PROFILE REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Data Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
            f.write(f"Collection Time: {datetime.now().isoformat()}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("COLUMNS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total columns: {len(df.columns)}\n")
            f.write(f"Column names: {', '.join(df.columns)}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("DATA TYPES\n")
            f.write("-" * 80 + "\n")
            f.write(f"{df.dtypes.to_string()}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("MISSING VALUES\n")
            f.write("-" * 80 + "\n")
            missing = df.isnull().sum()
            f.write(f"{missing[missing > 0].to_string() if missing.sum() > 0 else 'No missing values'}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{df.describe().to_string()}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("SAMPLE DATA (First 5 rows)\n")
            f.write("-" * 80 + "\n")
            f.write(f"{df.head().to_string()}\n\n")
        
        print("✅ Data profile report generated")
    except Exception as e:
        print(f"❌ Failed to create profile report: {e}")
        raise
    
    # Log to MLflow (if available)
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("weather_forecasting")
        
        with mlflow.start_run(run_name=f"data_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_artifact(report_path, "data_profiling")
            mlflow.log_param("data_shape", f"{df.shape[0]}x{df.shape[1]}")
            mlflow.log_param("collection_date", datetime.now().isoformat())
        
        print(f"✓ Profiling report generated and logged to MLflow")
    except Exception as e:
        print(f"⚠️  MLflow logging failed: {e}")
        print(f"⚠️  Profiling report saved to: {report_path}")
        print("⚠️  Continuing without MLflow tracking")
        # Don't fail the task if MLflow is unavailable
    
    return report_path


def train_model_task(**context):
    """Train model with MLflow tracking"""
    import subprocess
    import sys
    
    # Pull processed data path
    filepath = context["ti"].xcom_pull(key="processed_data_path", task_ids="transform_data")
    
    if not filepath:
        raise Exception("No processed data path found from transform_data task")
    
    print(f"Training model on: {filepath}")
    
    # Run training script with timeout handling
    try:
        result = subprocess.run(
            ["python", "/opt/airflow/scripts/train.py", filepath, "random_forest"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            print(f"Training failed with return code: {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise Exception(f"Model training failed: {result.stderr}")
        
        print(result.stdout)
        print("✓ Model training completed")
    except subprocess.TimeoutExpired:
        print("❌ Training timed out after 10 minutes")
        raise Exception("Model training timed out")
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        raise


def version_data(**context):
    """Version processed data with DVC"""
    import subprocess
    
    # Pull processed data path
    filepath = context["ti"].xcom_pull(key="processed_data_path", task_ids="transform_data")
    
    print(f"Versioning data: {filepath}")
    
    # Add to DVC (assuming data is in data/processed/)
    result = subprocess.run(
        ["dvc", "add", filepath],
        cwd="/opt/airflow",
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"DVC add failed: {result.stderr}")
        # Don't fail the DAG if DVC fails, just log it
        print("Warning: DVC versioning failed, continuing...")
    else:
        print("✓ Data versioned with DVC")


# Define tasks
extract_task = PythonOperator(
    task_id="extract_data",
    python_callable=extract_data,
    dag=dag,
)

quality_check_task = PythonOperator(
    task_id="check_quality",
    python_callable=check_quality,
    dag=dag,
)

transform_task = PythonOperator(
    task_id="transform_data",
    python_callable=transform_data_task,
    dag=dag,
)

profiling_task = PythonOperator(
    task_id="generate_profiling_report",
    python_callable=generate_profiling_report,
    dag=dag,
)

train_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model_task,
    dag=dag,
)

version_task = PythonOperator(
    task_id="version_data",
    python_callable=version_data,
    dag=dag,
)

# Define task dependencies
extract_task >> quality_check_task >> transform_task >> [profiling_task, train_task]
train_task >> version_task

