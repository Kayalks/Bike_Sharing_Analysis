# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
data = pd.read_csv("data/processed/duration_data.csv")
data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
data.set_index('Date', inplace=True)

# Borough-Wise Summary
borough_summary = data.groupby('Borough')['Average Duration'].agg(['mean', 'median', 'min', 'max']).reset_index()
print("Borough-Wise Duration Summary:")
print(borough_summary)

# Time-Based Trends
hourly_trends = data.groupby(['Borough', 'Hour'])['Average Duration'].mean().reset_index()
daily_trends = data.groupby(['Borough', 'Weekday'])['Average Duration'].mean().reset_index()

# Plot Hourly Trends for Each Borough
plt.figure(figsize=(10, 6))
for borough in data['Borough'].unique():
    borough_data = hourly_trends[hourly_trends['Borough'] == borough]
    plt.plot(borough_data['Hour'], borough_data['Average Duration'], label=borough)
plt.xlabel("Hour")
plt.ylabel("Average Duration")
plt.title("Hourly Duration Trends by Borough")
plt.legend()
plt.grid(True)
plt.show()

# Plot Daily Trends for Each Borough
plt.figure(figsize=(10, 6))
for borough in data['Borough'].unique():
    borough_data = daily_trends[daily_trends['Borough'] == borough]
    plt.plot(borough_data['Weekday'], borough_data['Average Duration'], label=borough)
plt.xlabel("Day of the Week")
plt.ylabel("Average Duration")
plt.title("Daily Duration Trends by Borough")
plt.legend()
plt.grid(True)
plt.show()

# Public Holiday Impact
holiday_impact = data.groupby(['Borough', 'Public Holiday'])['Average Duration'].mean().reset_index()
print("Impact of Public Holidays on Durations:")
print(holiday_impact)

# Visualize Public Holiday Impact
plt.figure(figsize=(10, 6))
for borough in data['Borough'].unique():
    borough_data = holiday_impact[holiday_impact['Borough'] == borough]
    plt.bar(borough_data['Public Holiday'], borough_data['Average Duration'], label=borough)
plt.xlabel("Public Holiday (0 = No, 1 = Yes)")
plt.ylabel("Average Duration")
plt.title("Public Holiday Impact on Average Duration by Borough")
plt.legend()
plt.grid(True)
plt.show()

# Weather Impact Analysis
weather_impact = data.groupby(['Borough', 'Weather Condition'])['Average Duration'].mean().reset_index()
print("Impact of Weather on Durations:")
print(weather_impact)

# Visualize Weather Impact
plt.figure(figsize=(10, 6))
for borough in data['Borough'].unique():
    borough_data = weather_impact[weather_impact['Borough'] == borough]
    plt.bar(borough_data['Weather Condition'], borough_data['Average Duration'], label=borough)
plt.xlabel("Weather Condition")
plt.ylabel("Average Duration")
plt.title("Weather Impact on Average Duration by Borough")
plt.legend()
plt.grid(True)
plt.show()
