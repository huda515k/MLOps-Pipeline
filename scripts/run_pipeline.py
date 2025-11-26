"""
Standalone script to run the complete pipeline locally
Useful for testing before deploying to Airflow
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add scripts to path
sys.path.append(str(Path(__file__).parent))

load_dotenv()

from extract import extract_weather_data, save_raw_data
from quality_check import check_data_quality
from transform import transform_data, save_processed_data
from train import train_pipeline
import pandas as pd


def run_complete_pipeline():
    """Run the complete MLOps pipeline"""
    print("=" * 60)
    print("Starting Complete MLOps Pipeline")
    print("=" * 60)
    
    # Step 1: Extract
    print("\n[1/5] Extracting data from API...")
    try:
        df_raw = extract_weather_data()
        raw_path = save_raw_data(df_raw)
        print(f"✓ Raw data saved to: {raw_path}")
    except Exception as e:
        print(f"✗ Extraction failed: {str(e)}")
        return False
    
    # Step 2: Quality Check
    print("\n[2/5] Running data quality checks...")
    try:
        results = check_data_quality(df_raw, strict=True)
        print("✓ All quality checks passed!")
    except Exception as e:
        print(f"✗ Quality check failed: {str(e)}")
        return False
    
    # Step 3: Transform
    print("\n[3/5] Transforming data and engineering features...")
    try:
        df_transformed = transform_data(df_raw, forecast_hours=4)
        processed_path = save_processed_data(df_transformed)
        print(f"✓ Processed data saved to: {processed_path}")
    except Exception as e:
        print(f"✗ Transformation failed: {str(e)}")
        return False
    
    # Step 4: Train Model
    print("\n[4/5] Training model...")
    try:
        model, metrics, run_id = train_pipeline(processed_path, model_type="random_forest")
        print(f"✓ Model trained successfully!")
        print(f"  Run ID: {run_id}")
        print(f"  Validation RMSE: {metrics['val_rmse']:.4f}")
        print(f"  Validation MAE: {metrics['val_mae']:.4f}")
        print(f"  Validation R²: {metrics['val_r2']:.4f}")
    except Exception as e:
        print(f"✗ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Summary
    print("\n[5/5] Pipeline Summary")
    print("=" * 60)
    print("✓ Data extraction: SUCCESS")
    print("✓ Quality checks: SUCCESS")
    print("✓ Data transformation: SUCCESS")
    print("✓ Model training: SUCCESS")
    print("=" * 60)
    print("\n🎉 Pipeline completed successfully!")
    
    return True


if __name__ == "__main__":
    success = run_complete_pipeline()
    sys.exit(0 if success else 1)

