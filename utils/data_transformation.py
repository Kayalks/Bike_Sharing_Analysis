# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 02:49:30 2024

@author: Kayalvili
"""

from pyspark.sql.functions import year, month, dayofmonth

def transform_raw_data(df):
    """
    Transforms the raw data DataFrame by extracting Year, Month, and Day from the Start Date column.

    :param df: Spark DataFrame with raw data
    :return: Transformed Spark DataFrame
    """
    transformed_df = df.withColumn("Year", year(df["Start Date"])) \
                      .withColumn("Month", month(df["Start Date"])) \
                      .withColumn("Day", dayofmonth(df["Start Date"]))
    return transformed_df
