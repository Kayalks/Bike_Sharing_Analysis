# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 01:58:04 2024

@author: Kayalvili
"""
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import split, count, avg
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
        StructField("StartStation Id", StringType(), True),
        StructField("StartStation Name", StringType(), True)
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
    required_columns = ["Rental Id", "Start Date", "Duration", "StartStation Id", "StartStation Name"]
    validate_data(mock_raw_data, required_columns)

def test_validate_weather_data(mock_weather_data):
    required_columns = ["Year", "Month", "Day", "weather_code", "temperature(average)", "apparent_temperature(average)", "precipitation", "wind_speed", "humidity", "pressure", "cloud_cover"]
    validate_data(mock_weather_data, required_columns)

def test_borough_extraction(mock_raw_data):
    transformed = mock_raw_data.withColumn("Borough", split(mock_raw_data["StartStation Name"], ",").getItem(1))
    assert transformed.filter(transformed["Borough"].isNull()).count() == 0

def test_demand_aggregation(mock_raw_data, spark_session):
    transformed = transform_raw_data(mock_raw_data)
    demand_data = transformed.groupBy("Year", "Month", "Day", "Hour").agg(count("Rental Id").alias("Demand"))
    assert demand_data.count() > 0

def test_duration_aggregation_by_borough(mock_raw_data, spark_session):
    transformed = transform_raw_data(mock_raw_data)
    transformed = transformed.withColumn("Borough", split(transformed["StartStation Name"], ",").getItem(1))
    duration_data = transformed.groupBy("Year", "Month", "Day", "Borough").agg(avg("Duration").alias("AverageDuration"))
    assert duration_data.count() > 0

def test_weather_merge_demand(mock_weather_data, spark_session, mock_raw_data):
    transformed = transform_raw_data(mock_raw_data)
    demand_data = transformed.groupBy("Year", "Month", "Day", "Hour").agg(count("Rental Id").alias("Demand"))
    merged_data = demand_data.join(mock_weather_data, on=["Year", "Month", "Day"], how="left")
    assert merged_data.filter(merged_data["weather_code"].isNull()).count() == 0

def test_weather_merge_duration(mock_weather_data, spark_session, mock_raw_data):
    transformed = transform_raw_data(mock_raw_data)
    transformed = transformed.withColumn("Borough", split(transformed["StartStation Name"], ",").getItem(1))
    duration_data = transformed.groupBy("Year", "Month", "Day", "Borough").agg(avg("Duration").alias("AverageDuration"))
    merged_data = duration_data.join(mock_weather_data, on=["Year", "Month", "Day"], how="left")
    assert merged_data.filter(merged_data["weather_code"].isNull()).count() == 0
