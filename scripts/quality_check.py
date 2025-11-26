"""
Data Quality Check Module
Implements strict quality gates for weather data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


class DataQualityError(Exception):
    """Custom exception for data quality failures"""
    pass


def check_null_values(df: pd.DataFrame, threshold: float = 0.01) -> Tuple[bool, Dict]:
    """
    Check if null values exceed threshold (default 1%)
    Returns: (is_valid, report_dict)
    """
    total_rows = len(df)
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / total_rows * 100).round(2)
    
    # Key columns that must not have nulls
    key_columns = ["temp", "timestamp", "humidity", "pressure", "wind_speed"]
    
    violations = {}
    for col in key_columns:
        if col in df.columns:
            null_pct = null_percentages[col]
            if null_pct > (threshold * 100):
                violations[col] = {
                    "null_count": int(null_counts[col]),
                    "null_percentage": float(null_pct),
                    "threshold": threshold * 100
                }
    
    is_valid = len(violations) == 0
    
    report = {
        "is_valid": is_valid,
        "total_rows": total_rows,
        "null_percentages": null_percentages.to_dict(),
        "violations": violations,
        "threshold": threshold * 100
    }
    
    return is_valid, report


def check_schema(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    Validate that required columns exist and have correct data types
    """
    required_columns = {
        "timestamp": "datetime64[ns]",
        "temp": "float64",
        "humidity": "float64",
        "pressure": "float64",
        "wind_speed": "float64"
    }
    
    missing_columns = []
    type_mismatches = {}
    
    for col, expected_dtype in required_columns.items():
        if col not in df.columns:
            missing_columns.append(col)
        else:
            actual_dtype = str(df[col].dtype)
            if expected_dtype not in actual_dtype and "int" not in actual_dtype:
                type_mismatches[col] = {
                    "expected": expected_dtype,
                    "actual": actual_dtype
                }
    
    is_valid = len(missing_columns) == 0 and len(type_mismatches) == 0
    
    report = {
        "is_valid": is_valid,
        "missing_columns": missing_columns,
        "type_mismatches": type_mismatches
    }
    
    return is_valid, report


def check_value_ranges(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    Check that numeric values are within reasonable ranges
    """
    range_checks = {
        "temp": (-50, 60),  # Celsius
        "humidity": (0, 100),  # Percentage
        "pressure": (800, 1100),  # hPa
        "wind_speed": (0, 200),  # m/s
    }
    
    violations = {}
    
    for col, (min_val, max_val) in range_checks.items():
        if col in df.columns:
            out_of_range = (df[col] < min_val) | (df[col] > max_val)
            if out_of_range.any():
                violations[col] = {
                    "min_allowed": min_val,
                    "max_allowed": max_val,
                    "min_actual": float(df[col].min()),
                    "max_actual": float(df[col].max()),
                    "out_of_range_count": int(out_of_range.sum())
                }
    
    is_valid = len(violations) == 0
    
    report = {
        "is_valid": is_valid,
        "violations": violations
    }
    
    return is_valid, report


def check_data_quality(df: pd.DataFrame, strict: bool = True) -> Dict:
    """
    Run all data quality checks
    Raises DataQualityError if strict=True and checks fail
    """
    results = {}
    
    # Check null values
    null_valid, null_report = check_null_values(df)
    results["null_check"] = null_report
    
    # Check schema
    schema_valid, schema_report = check_schema(df)
    results["schema_check"] = schema_report
    
    # Check value ranges
    range_valid, range_report = check_value_ranges(df)
    results["range_check"] = range_report
    
    # Overall validation
    overall_valid = null_valid and schema_valid and range_valid
    results["overall_valid"] = overall_valid
    
    if strict and not overall_valid:
        error_msg = "Data quality checks failed:\n"
        if not null_valid:
            error_msg += f"  - Null value violations: {null_report['violations']}\n"
        if not schema_valid:
            error_msg += f"  - Schema violations: Missing={schema_report['missing_columns']}, Types={schema_report['type_mismatches']}\n"
        if not range_valid:
            error_msg += f"  - Range violations: {range_report['violations']}\n"
        
        raise DataQualityError(error_msg)
    
    return results


if __name__ == "__main__":
    # Test the quality check module
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quality_check.py <data_file_path>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    df = pd.read_parquet(filepath)
    
    try:
        results = check_data_quality(df, strict=True)
        print("✓ All data quality checks passed!")
        print(f"  Total rows: {results['null_check']['total_rows']}")
    except DataQualityError as e:
        print(f"✗ Data quality check failed:\n{e}")
        sys.exit(1)

