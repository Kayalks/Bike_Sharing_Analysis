# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 01:02:15 2024

@author: Kayalvili
"""

import logging
from pyspark.sql import SparkSession
from utils.validation import validate_data
from utils.data_transformation import transform_raw_data
from pyspark.sql.functions import count, avg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataCleansingPipeline")

# Initialize Spark session
spark = SparkSession.builder.appName("DataCleansingPipeline").getOrCreate()
logger.info("Spark session initialized.")

# Paths
input_path = "data/raw/*/*.csv"
weather_path = "data/raw/weather/*.csv"
output_demand_path = "data/processed/demand/demand_data.csv"
output_duration_path = "data/processed/duration/duration_data.csv"

# Define required columns
required_raw_columns = ["Rental Id", "Start Date", "Duration", "StartStation Id"]
required_weather_columns = ["Year", "Month", "Day", "weather_code", "temperature(average)", "apparent_temperature(average)", "precipitation", "wind_speed", "humidity", "pressure", "cloud_cover"]

# Load raw data
logger.info("Loading raw data.")
raw_data = spark.read.csv(input_path, header=True, inferSchema=True)
logger.info("Raw data loaded successfully.")

# Validate raw data
logger.info("Validating raw data.")
validate_data(raw_data, required_raw_columns)

# Transform raw data
logger.info("Transforming raw data.")
transformed_raw_data = transform_raw_data(raw_data)

# Load weather data
logger.info("Loading weather data.")
weather_data = spark.read.csv(weather_path, header=True, inferSchema=True)
logger.info("Weather data loaded successfully.")

# Validate weather data
logger.info("Validating weather data.")
validate_data(weather_data, required_weather_columns)

# Merge raw data with weather data
logger.info("Merging raw data with weather data.")
merged_data = transformed_raw_data.join(weather_data, on=["Year", "Month", "Day"], how="left")
logger.info("Data merged successfully.")

# Generate demand data
logger.info("Generating demand data.")
demand_data = merged_data.groupBy("Year", "Month", "Day", "Hour", "Weekday", "WorkingDay", "PublicHoliday", "weather_code", "temperature(average)", "apparent_temperature(average)", "precipitation", "wind_speed", "humidity", "pressure", "cloud_cover") \
    .agg(count("Rental Id").alias("Demand"))
logger.info("Demand data generated successfully.")

# Save demand data
logger.info("Saving demand data.")
demand_data.coalesce(1).write.csv(output_demand_path, header=True, mode="overwrite")
logger.info(f"Demand data saved to {output_demand_path}")

# Generate duration data
logger.info("Generating duration data.")
duration_data = merged_data.groupBy("Year", "Month", "Day", "Weekday", "WorkingDay", "PublicHoliday", "StartStation Id", "weather_code", "temperature(average)", "apparent_temperature(average)", "precipitation", "wind_speed", "humidity", "pressure", "cloud_cover") \
    .agg(avg("Duration").alias("AverageDuration"))
logger.info("Duration data generated successfully.")

# Save duration data
logger.info("Saving duration data.")
duration_data.coalesce(1).write.csv(output_duration_path, header=True, mode="overwrite")
logger.info(f"Duration data saved to {output_duration_path}")
