# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 01:02:15 2024

@author: Kayalvili
"""

import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, hour, date_format, col, count, when, avg, isnan, isnull

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BikeRentalDataPipeline")

# Initialize Spark session
spark = SparkSession.builder.appName("BikeRentalDataPipeline").getOrCreate()
logger.info("Spark session initialized.")

# Paths
input_path = "data/raw/*/*.csv"  # Adjusted to match the folder structure for raw data
weather_path = "data/raw/weather/*.csv"  # Path for weather data
output_demand_path = "data/processed/demand/demand_data.csv"  # Output path for demand data
output_duration_path = "data/processed/duration/duration_data.csv"  # Output path for duration data

# Public Holiday List (2022, 2023, 2024)
public_holidays = [
    "2022-01-01", "2022-01-03", "2022-04-15", "2022-05-02", "2022-06-02", "2022-06-03", "2022-09-19", "2022-12-25", "2022-12-26", "2022-12-27",
    "2023-01-01", "2023-01-02", "2023-04-07", "2023-05-01", "2023-05-08", "2023-05-29", "2023-12-25", "2023-12-26",
    "2024-01-01", "2024-03-29", "2024-05-06", "2024-05-27", "2024-12-25", "2024-12-26"
]
logger.info("Public holiday list configured.")

# Load all CSV files into a DataFrame
logger.info("Loading raw data from input path.")
df = spark.read.csv(input_path, header=True, inferSchema=True)
logger.info("Raw data loaded successfully.")

# Load Weather Data
logger.info("Loading weather data.")
weather_df = spark.read.csv(weather_path, header=True, inferSchema=True)
logger.info("Weather data loaded successfully.")

# Automated Validation Function
def validate_data(df, required_columns):
    logger.info("Starting data validation.")
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns in the raw data: {missing_columns}")
        raise ValueError(f"Missing required columns in the raw data: {missing_columns}")

    validation_results = {}
    for column in required_columns:
        null_count = df.filter(isnull(col(column)) | isnan(col(column))).count()
        validation_results[column] = null_count

    invalid_columns = [col for col, null_count in validation_results.items() if null_count > 0]
    if invalid_columns:
        logger.error(f"Columns with null or invalid values: {validation_results}")
        raise ValueError(f"Columns with null or invalid values: {validation_results}")

    logger.info("Data validation passed successfully.")

# List of required columns for raw data
required_columns = ["Rental Id", "Start Date", "Duration", "StartStation Id"]
validate_data(df, required_columns)

# List of required columns for weather data
required_weather_columns = ["Year", "Month", "Day", "weather_code", "temperature(average)", "apparent_temperature(average)", "precipitation", "wind_speed", "humidity", "pressure", "cloud_cover"]
validate_data(weather_df, required_weather_columns)

# Transform Start Date to extract Year, Month, Day, Weekday, and Hour
logger.info("Transforming raw data.")
df = df.withColumn("Year", year(col("Start Date"))) \
       .withColumn("Month", month(col("Start Date"))) \
       .withColumn("Day", dayofmonth(col("Start Date"))) \
       .withColumn("Hour", hour(col("Start Date"))) \
       .withColumn("Weekday", date_format(col("Start Date"), 'E')) \
       .withColumn("WorkingDay", when(date_format(col("Start Date"), 'E').isin(['Sat', 'Sun']), 0).otherwise(1)) \
       .withColumn("PublicHoliday", when(col("Start Date").substr(1, 10).isin(public_holidays), 1).otherwise(0))
logger.info("Data transformation completed.")

# Optimize Weather Data for Join
logger.info("Optimizing weather data for join.")
weather_df = weather_df.select("Year", "Month", "Day", "weather_code", "temperature(average)", "apparent_temperature(average)", "precipitation", "wind_speed", "humidity", "pressure", "cloud_cover").drop_duplicates()
logger.info("Weather data optimized for join.")

# Validation for Join Integrity
logger.info("Validating join integrity.")
missing_weather = df.join(weather_df, on=["Year", "Month", "Day"], how="left_anti").count()
if missing_weather > 0:
    logger.warning(f"{missing_weather} records in raw data have no corresponding weather data.")
else:
    logger.info("All records in raw data have corresponding weather data.")

# Merge Weather Data
logger.info("Merging weather data with raw data.")
df = df.join(weather_df, on=["Year", "Month", "Day"], how="left")
logger.info("Weather data merged successfully.")

# Generate Demand Data (Hourly Grouping)
logger.info("Generating demand data.")
demand_df = df.groupBy("Year", "Month", "Day", "Hour", "Weekday", "WorkingDay", "PublicHoliday", "weather_code", "temperature(average)", "apparent_temperature(average)", "precipitation", "wind_speed", "humidity", "pressure", "cloud_cover") \
              .agg(count("Rental Id").alias("Demand"))

# Save Demand Data
logger.info("Saving demand data.")
demand_df.coalesce(1).write.csv(output_demand_path, header=True, mode="overwrite")
logger.info(f"Demand data has been saved to {output_demand_path}")

# Generate Duration Data (Daily Grouping by Start Station)
logger.info("Generating duration data.")
duration_df = df.groupBy("Year", "Month", "Day", "Weekday", "WorkingDay", "PublicHoliday", "StartStation Id", "weather_code", "temperature(average)", "apparent_temperature(average)", "precipitation", "wind_speed", "humidity", "pressure", "cloud_cover") \
                .agg(avg("Duration").alias("AverageDuration"))

# Save Duration Data
logger.info("Saving duration data.")
duration_df.coalesce(1).write.csv(output_duration_path, header=True, mode="overwrite")
logger.info(f"Duration data has been saved to {output_duration_path}")
