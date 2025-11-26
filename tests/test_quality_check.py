"""
Unit tests for data quality check module
"""
import pytest
import pandas as pd
import numpy as np
from scripts.quality_check import (
    check_null_values,
    check_schema,
    check_value_ranges,
    check_data_quality,
    DataQualityError
)


def test_check_null_values_pass():
    """Test null value check with valid data"""
    df = pd.DataFrame({
        "temp": [20, 21, 22],
        "humidity": [50, 55, 60],
        "pressure": [1013, 1014, 1015]
    })
    
    is_valid, report = check_null_values(df, threshold=0.01)
    assert is_valid is True
    assert report["is_valid"] is True


def test_check_null_values_fail():
    """Test null value check with too many nulls"""
    df = pd.DataFrame({
        "temp": [20, None, None],
        "humidity": [50, 55, 60],
        "pressure": [1013, 1014, 1015]
    })
    
    is_valid, report = check_null_values(df, threshold=0.01)
    assert is_valid is False
    assert "temp" in report["violations"]


def test_check_schema_pass():
    """Test schema check with correct schema"""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "temp": [20.0, 21.0],
        "humidity": [50.0, 55.0],
        "pressure": [1013.0, 1014.0],
        "wind_speed": [5.0, 6.0]
    })
    
    is_valid, report = check_schema(df)
    assert is_valid is True


def test_check_schema_fail_missing_column():
    """Test schema check with missing column"""
    df = pd.DataFrame({
        "temp": [20.0, 21.0],
        "humidity": [50.0, 55.0]
    })
    
    is_valid, report = check_schema(df)
    assert is_valid is False
    assert "timestamp" in report["missing_columns"]


def test_check_value_ranges_pass():
    """Test value range check with valid ranges"""
    df = pd.DataFrame({
        "temp": [20, 25, 30],
        "humidity": [50, 60, 70],
        "pressure": [1000, 1013, 1020],
        "wind_speed": [5, 10, 15]
    })
    
    is_valid, report = check_value_ranges(df)
    assert is_valid is True


def test_check_value_ranges_fail():
    """Test value range check with out-of-range values"""
    df = pd.DataFrame({
        "temp": [200, 25, 30],  # Too high
        "humidity": [50, 60, 70],
        "pressure": [1000, 1013, 1020],
        "wind_speed": [5, 10, 15]
    })
    
    is_valid, report = check_value_ranges(df)
    assert is_valid is False
    assert "temp" in report["violations"]


def test_check_data_quality_strict_pass():
    """Test complete quality check with valid data"""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "temp": [20.0, 21.0, 22.0],
        "humidity": [50.0, 55.0, 60.0],
        "pressure": [1013.0, 1014.0, 1015.0],
        "wind_speed": [5.0, 6.0, 7.0]
    })
    
    results = check_data_quality(df, strict=True)
    assert results["overall_valid"] is True


def test_check_data_quality_strict_fail():
    """Test complete quality check with invalid data (should raise exception)"""
    df = pd.DataFrame({
        "temp": [200.0, 21.0, 22.0],  # Out of range
        "humidity": [None, 55.0, 60.0],  # Null value
    })
    
    with pytest.raises(DataQualityError):
        check_data_quality(df, strict=True)

