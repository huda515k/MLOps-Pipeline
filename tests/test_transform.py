"""
Unit tests for data transformation module
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scripts.transform import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    transform_data
)


def test_create_time_features():
    """Test time feature creation"""
    dates = pd.date_range(start="2024-01-01", periods=24, freq="H")
    df = pd.DataFrame({"timestamp": dates, "temp": range(24)})
    
    df_result = create_time_features(df)
    
    assert "hour" in df_result.columns
    assert "day_of_week" in df_result.columns
    assert "month" in df_result.columns
    assert "is_weekend" in df_result.columns
    assert "hour_sin" in df_result.columns
    assert "hour_cos" in df_result.columns


def test_create_lag_features():
    """Test lag feature creation"""
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=10, freq="H"),
        "temp": range(10)
    })
    
    df_result = create_lag_features(df, target_col="temp", lags=[1, 2])
    
    assert "temp_lag_1" in df_result.columns
    assert "temp_lag_2" in df_result.columns
    assert pd.isna(df_result["temp_lag_1"].iloc[0])
    assert df_result["temp_lag_1"].iloc[1] == 0


def test_create_rolling_features():
    """Test rolling feature creation"""
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=10, freq="H"),
        "temp": range(10)
    })
    
    df_result = create_rolling_features(df, target_col="temp", windows=[3])
    
    assert "temp_rolling_mean_3" in df_result.columns
    assert "temp_rolling_std_3" in df_result.columns
    assert not pd.isna(df_result["temp_rolling_mean_3"].iloc[0])


def test_transform_data():
    """Test complete transformation pipeline"""
    dates = pd.date_range(start="2024-01-01", periods=20, freq="H")
    df = pd.DataFrame({
        "timestamp": dates,
        "temp": np.random.uniform(15, 25, 20),
        "humidity": np.random.uniform(40, 80, 20),
        "pressure": np.random.uniform(1000, 1020, 20),
        "wind_speed": np.random.uniform(0, 15, 20),
        "temp_min": np.random.uniform(10, 20, 20),
        "temp_max": np.random.uniform(20, 30, 20),
        "feels_like": np.random.uniform(15, 25, 20),
        "wind_deg": np.random.uniform(0, 360, 20),
        "clouds": np.random.uniform(0, 100, 20)
    })
    
    df_result = transform_data(df, forecast_hours=4)
    
    assert "target_temp" in df_result.columns
    assert "hour" in df_result.columns
    assert "temp_lag_1" in df_result.columns
    assert len(df_result) > 0

