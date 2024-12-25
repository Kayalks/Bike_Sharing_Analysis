# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 01:58:04 2024

@author: Kayalvili
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from utils.validation import validate_data
from utils.data_transformation import transform_raw_data

@pytest.fixture(scope="session")
def spark_session():
    return SparkSession.builder.master("local").appName("TestPipeline").getOrCreate()

# Mock data for raw data
@pytest.fixture
def mock_raw_data(spark_session):
    raw_data_path = "tests/mock_data/raw_data.csv"
    schema = StructType([
        StructField("Rental Id", StringType(), True),
        StructField("Start Date", StringType(), True),
        StructField("Duration", IntegerType(), True),
        StructField("StartStation Id", StringType(), True)
    ])
    return spark_session.read.csv(raw_data_path, header=True, schema=schema)

# Mock data for weather data
@pytest.fixture
def mock_weather_data(spark_session):
    weather_data_path = "tests/mock_data/weather_data.csv"
    schema = StructType([
        StructField("Year", IntegerType(), True),
        StructField("Month", IntegerType(), True),
        StructField("Day", IntegerType(), True),
        StructField("weather_code", IntegerType(), True),
        StructField("temperature(average)", DoubleType(), True),
        StructField("apparent_temperature(average)", DoubleType(), True),
        StructField("precipitation", DoubleType(), True),
        StructField("wind_speed", DoubleType(), True),
        StructField("humidity", DoubleType(), True),
        StructField("pressure", DoubleType(), True),
        StructField("cloud_cover", DoubleType(), True)
    ])
    return spark_session.read.csv(weather_data_path, header=True, schema=schema)

def test_validate_raw_data(mock_raw_data):
    required_columns = ["Rental Id", "Start Date", "Duration", "StartStation Id"]
    with pytest.raises(ValueError):
        validate_data(mock_raw_data, required_columns)

def test_validate_weather_data(mock_weather_data):
    required_columns = ["Year", "Month", "Day", "weather_code", "temperature(average)", "apparent_temperature(average)", "precipitation", "wind_speed", "humidity", "pressure", "cloud_cover"]
    validate_data(mock_weather_data, required_columns)  # Should pass without errors

def test_merge_weather_and_raw(spark_session, mock_raw_data, mock_weather_data):
    # Transform mock_raw_data for joining
    transformed_raw_data = transform_raw_data(mock_raw_data)

    # Perform the join
    joined_df = transformed_raw_data.join(mock_weather_data, on=["Year", "Month", "Day"], how="left")

    # Validate the join
    assert joined_df.count() == transformed_raw_data.count()
    assert joined_df.filter(joined_df["weather_code"].isNull()).count() == 0
