# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 02:49:17 2024

@author: Kayalvili
"""

def validate_data(df, required_columns):
    """
    Validates the given DataFrame for required columns and checks for null or invalid values.

    :param df: Spark DataFrame to validate
    :param required_columns: List of required column names
    :raises ValueError: If required columns are missing or contain null/invalid values
    """
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check for null or invalid values in required columns
    invalid_columns = []
    for column in required_columns:
        null_count = df.filter(df[column].isNull() | df[column].isNaN()).count()
        if null_count > 0:
            invalid_columns.append((column, null_count))

    if invalid_columns:
        raise ValueError(f"Columns with null or invalid values: {invalid_columns}")

    print("Data validation passed successfully.")
