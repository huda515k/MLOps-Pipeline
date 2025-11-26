"""
Data Transformation and Feature Engineering
Creates time-series features for weather forecasting
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import os


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features"""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Cyclical encoding for time features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    return df


def create_lag_features(df: pd.DataFrame, target_col: str = "temp", lags: list = [1, 2, 3, 6]) -> pd.DataFrame:
    """Create lag features for time-series prediction"""
    df = df.copy()
    df = df.sort_values("timestamp")
    
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str = "temp", windows: list = [3, 6, 12]) -> pd.DataFrame:
    """Create rolling window statistics"""
    df = df.copy()
    df = df.sort_values("timestamp")
    
    for window in windows:
        df[f"{target_col}_rolling_mean_{window}"] = df[target_col].rolling(window=window, min_periods=1).mean()
        df[f"{target_col}_rolling_std_{window}"] = df[target_col].rolling(window=window, min_periods=1).std()
        df[f"{target_col}_rolling_min_{window}"] = df[target_col].rolling(window=window, min_periods=1).min()
        df[f"{target_col}_rolling_max_{window}"] = df[target_col].rolling(window=window, min_periods=1).max()
    
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived/engineered features"""
    df = df.copy()
    
    # Temperature difference features
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["temp_feels_diff"] = df["feels_like"] - df["temp"]
    
    # Weather interaction features
    df["heat_index"] = (
        0.5 * (df["temp"] + 61.0 + ((df["temp"] - 68.0) * 1.2) + (df["humidity"] * 0.094))
    )
    
    # Wind chill (approximate)
    df["wind_chill"] = (
        13.12 + 0.6215 * df["temp"] - 11.37 * (df["wind_speed"] ** 0.16) +
        0.3965 * df["temp"] * (df["wind_speed"] ** 0.16)
    )
    
    # Pressure change rate (if we have historical data)
    if "pressure" in df.columns:
        df["pressure_change"] = df["pressure"].diff()
        df["pressure_change_pct"] = df["pressure"].pct_change()
    
    return df


def create_target_variable(df: pd.DataFrame, forecast_hours: int = 4) -> pd.DataFrame:
    """
    Create target variable: temperature forecast_hours ahead
    For training, we need to shift the target backwards
    """
    df = df.copy()
    df = df.sort_values("timestamp")
    
    # For prediction: we want to predict temp at t+forecast_hours
    # So we create a target that is the temp shifted forward by forecast_hours
    df["target_temp"] = df["temp"].shift(-forecast_hours)
    
    return df


def transform_data(df: pd.DataFrame, forecast_hours: int = 4) -> pd.DataFrame:
    """
    Complete transformation pipeline
    """
    print("Starting data transformation...")
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Create time features
    print("  Creating time features...")
    df = create_time_features(df)
    
    # Create lag features
    print("  Creating lag features...")
    df = create_lag_features(df, target_col="temp", lags=[1, 2, 3, 6, 12])
    
    # Create rolling features
    print("  Creating rolling features...")
    df = create_rolling_features(df, target_col="temp", windows=[3, 6, 12])
    
    # Create derived features
    print("  Creating derived features...")
    df = create_derived_features(df)
    
    # Create target variable
    print("  Creating target variable...")
    df = create_target_variable(df, forecast_hours=forecast_hours)
    
    # Drop rows with NaN target (these are the last forecast_hours rows)
    df = df.dropna(subset=["target_temp"])
    
    # Fill remaining NaN values in features with forward fill and backward fill
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    print(f"Transformation complete. Final shape: {df.shape}")
    
    return df


def save_processed_data(df: pd.DataFrame, output_dir: str = "data/processed") -> str:
    """Save processed data"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"processed_weather_{timestamp}.parquet"
    filepath = os.path.join(output_dir, filename)
    
    df.to_parquet(filepath, index=False)
    print(f"Saved processed data to: {filepath}")
    
    return filepath


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python transform.py <input_file_path> [output_dir]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    
    print(f"Loading data from: {input_file}")
    df = pd.read_parquet(input_file)
    
    # Transform
    df_transformed = transform_data(df, forecast_hours=4)
    
    # Save
    output_path = save_processed_data(df_transformed, output_dir)
    print(f"✓ Transformation completed. Output: {output_path}")

