# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 19:25:49 2024

@author: Kayalvili
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load duration data (update the path as needed)
duration_data = pd.read_csv("data/processed/duration/duration_data.csv")

# Convert date-related columns to datetime for easier manipulation
duration_data['Date'] = pd.to_datetime(duration_data[['Year', 'Month', 'Day']])

# Set styles for plots
sns.set(style="whitegrid")

# 1. Time-Based Trends
# a. Daily Average Duration
plt.figure(figsize=(12, 6))
daily_duration = duration_data.groupby('Date')['AverageDuration'].mean().reset_index()
plt.plot(daily_duration['Date'], daily_duration['AverageDuration'], marker='o', color='blue')
plt.title('Daily Average Duration', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Duration (minutes)', fontsize=12)
plt.show()

# b. Hourly Patterns
plt.figure(figsize=(12, 6))
hourly_duration = duration_data.groupby('Hour')['AverageDuration'].mean().reset_index()
sns.barplot(x='Hour', y='AverageDuration', data=hourly_duration, palette='coolwarm')
plt.title('Hourly Average Duration Patterns', fontsize=14)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Average Duration (minutes)', fontsize=12)
plt.show()

# 2. Spatial Analysis
# Borough-Wise Average Duration (if Borough exists)
if 'Borough' in duration_data.columns:
    borough_duration = duration_data.groupby('Borough')['AverageDuration'].mean().reset_index()
    borough_duration = borough_duration.sort_values(by='AverageDuration', ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='AverageDuration', y='Borough', data=borough_duration, palette='viridis')
    plt.title('Borough-Wise Average Duration', fontsize=14)
    plt.xlabel('Average Duration (minutes)', fontsize=12)
    plt.ylabel('Borough', fontsize=12)
    plt.show()

# 3. Weather Impact
# a. Temperature vs. Average Duration
plt.figure(figsize=(12, 6))
sns.scatterplot(x='temperature(average)', y='AverageDuration', data=duration_data, alpha=0.7, hue='WorkingDay', palette='coolwarm')
plt.title('Temperature vs. Average Duration', fontsize=14)
plt.xlabel('Temperature (Average)', fontsize=12)
plt.ylabel('Average Duration (minutes)', fontsize=12)
plt.show()

# b. Precipitation Impact
plt.figure(figsize=(12, 6))
sns.boxplot(x='PublicHoliday', y='AverageDuration', data=duration_data, palette='Set2')
plt.title('Precipitation Impact on Average Duration', fontsize=14)
plt.xlabel('Public Holiday (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Average Duration (minutes)', fontsize=12)
plt.show()

# 4. Statistical Summary
print("Descriptive Statistics for Average Duration:")
print(duration_data['AverageDuration'].describe())

# 5. Correlation Analysis
# Correlation Heatmap
plt.figure(figsize=(12, 6))
correlation_matrix = duration_data[['AverageDuration', 'temperature(average)', 'precipitation', 'wind_speed', 'humidity', 'pressure', 'cloud_cover']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap', fontsize=14)
plt.show()
