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
    """Generate pandas profiling report and log to MLflow/Dagshub"""
    import pandas as pd
    import mlflow
    import os
    import tempfile
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
    
    # Configure Dagshub/MLflow tracking
    DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "")
    DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "")
    DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "")
    
    # Set MLflow tracking URI (Dagshub if configured, otherwise local)
    if DAGSHUB_USERNAME and DAGSHUB_REPO and DAGSHUB_TOKEN:
        try:
            # Initialize Dagshub for proper MLflow integration
            import dagshub
            dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True, token=DAGSHUB_TOKEN)
            print(f"✅ Dagshub initialized")
        except Exception as e:
            print(f"⚠️  Dagshub init failed (non-critical): {e}")
        
        MLFLOW_TRACKING_URI = f"https://{DAGSHUB_USERNAME}:{DAGSHUB_TOKEN}@dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
        print(f"✅ Using Dagshub MLflow: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
    else:
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        print(f"Using local MLflow: {MLFLOW_TRACKING_URI}")
        if not DAGSHUB_USERNAME:
            print("⚠️  DAGSHUB_USERNAME not set")
        if not DAGSHUB_REPO:
            print("⚠️  DAGSHUB_REPO not set")
        if not DAGSHUB_TOKEN:
            print("⚠️  DAGSHUB_TOKEN not set")
    
    # Generate Pandas Profiling HTML report
    report_path = None
    use_profiling = False
    try:
        # Try to use pandas-profiling (or ydata-profiling if available)
        try:
            from ydata_profiling import ProfileReport
            use_profiling = True
            print("✅ Using ydata-profiling (maintained fork)")
        except ImportError:
            try:
                from pandas_profiling import ProfileReport
                use_profiling = True
                print("✅ Using pandas-profiling")
            except ImportError:
                print("⚠️  Neither ydata-profiling nor pandas-profiling found, using basic report")
                ProfileReport = None
                use_profiling = False
        
        if use_profiling and ProfileReport:
            # Generate comprehensive HTML profile report
            print("Generating comprehensive data profile report...")
            profile = ProfileReport(
                df,
                title="Weather Data Profile Report",
                minimal=False,
                explorative=True,
                html={"style": {"full_width": True}}
            )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp:
                report_path = tmp.name
                profile.to_file(report_path)
            
            print(f"✅ Pandas Profiling HTML report generated: {report_path}")
        else:
            # Fallback: Create basic text report if profiling library not available
            print("⚠️  Creating basic text report (profiling library not available)")
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
                report_path = tmp.name
                tmp.write("=" * 80 + "\n")
                tmp.write("WEATHER DATA PROFILE REPORT\n")
                tmp.write("=" * 80 + "\n\n")
                tmp.write(f"Data Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
                tmp.write(f"Collection Time: {datetime.now().isoformat()}\n\n")
                tmp.write("-" * 80 + "\n")
                tmp.write("SUMMARY STATISTICS\n")
                tmp.write("-" * 80 + "\n")
                tmp.write(f"{df.describe().to_string()}\n\n")
            
    except Exception as e:
        print(f"❌ Failed to create profile report: {e}")
        raise
    
    # Log to MLflow/Dagshub
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("weather_forecasting")
        
        with mlflow.start_run(run_name=f"data_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log the profiling report as artifact
            mlflow.log_artifact(report_path, "data_profiling")
            
            # Log metadata as parameters
            mlflow.log_param("data_shape", f"{df.shape[0]}x{df.shape[1]}")
            mlflow.log_param("collection_date", datetime.now().isoformat())
            mlflow.log_param("total_columns", len(df.columns))
            mlflow.log_param("missing_values", int(df.isnull().sum().sum()))
            mlflow.log_param("report_type", "pandas_profiling" if use_profiling else "basic_text")
            
            # Log data quality metrics
            mlflow.log_metric("data_quality_score", 1.0 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])))
        
        print(f"✅ Profiling report logged to MLflow/Dagshub")
        print(f"✅ Report available in MLflow UI under 'data_profiling' artifact")
        
        # Clean up temporary file
        if report_path and os.path.exists(report_path):
            os.unlink(report_path)
            print(f"✅ Temporary report file cleaned up")
            
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
        print(f"❌ DVC add failed: {result.stderr}")
        print(f"STDOUT: {result.stdout}")
        raise Exception(f"DVC versioning failed: {result.stderr}")
    else:
        print("✓ Data versioned with DVC")
        print(f"DVC output: {result.stdout}")


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

