# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 19:23:06 2024

@author: Kayalvili
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load demand data (update the path as needed)
demand_data = pd.read_csv("data/processed/demand/demand_data.csv")

# Convert date-related columns to datetime for easier manipulation
demand_data['Date'] = pd.to_datetime(demand_data[['Year', 'Month', 'Day']])

# Set styles for plots
sns.set(style="whitegrid")

# 1. Daily Demand Trends
plt.figure(figsize=(12, 6))
daily_demand = demand_data.groupby('Date')['Demand'].sum().reset_index()
plt.plot(daily_demand['Date'], daily_demand['Demand'], marker='o')
plt.title('Daily Demand Trends', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Demand', fontsize=12)
plt.show()

# 2. Hourly Demand Patterns
plt.figure(figsize=(12, 6))
hourly_demand = demand_data.groupby('Hour')['Demand'].mean().reset_index()
sns.barplot(x='Hour', y='Demand', data=hourly_demand, palette='viridis')
plt.title('Hourly Demand Patterns', fontsize=14)
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Average Demand', fontsize=12)
plt.show()

# 3. Day-of-Week Trends
plt.figure(figsize=(12, 6))
day_of_week_demand = demand_data.groupby('Weekday')['Demand'].mean().reset_index()
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
sns.barplot(x='Weekday', y='Demand', data=day_of_week_demand, order=day_order, palette='muted')
plt.title('Day of Week Trends', fontsize=14)
plt.xlabel('Weekday', fontsize=12)
plt.ylabel('Average Demand', fontsize=12)
plt.show()

# 4. Temperature vs. Demand
plt.figure(figsize=(12, 6))
sns.scatterplot(x='temperature(average)', y='Demand', data=demand_data, alpha=0.7, hue='WorkingDay', palette='cool')
plt.title('Temperature vs. Demand', fontsize=14)
plt.xlabel('Temperature (Average)', fontsize=12)
plt.ylabel('Demand', fontsize=12)
plt.show()

# 5. Precipitation Impact
plt.figure(figsize=(12, 6))
sns.boxplot(x='WorkingDay', y='Demand', hue='PublicHoliday', data=demand_data, palette='Set2')
plt.title('Precipitation Impact on Demand', fontsize=14)
plt.xlabel('Working Day', fontsize=12)
plt.ylabel('Demand', fontsize=12)
plt.legend(title="Public Holiday")
plt.show()

# 6. Statistical Summary
print("Descriptive Statistics:")
print(demand_data['Demand'].describe())

# 7. Correlation Analysis
plt.figure(figsize=(12, 6))
correlation_matrix = demand_data[['Demand', 'temperature(average)', 'precipitation', 'wind_speed', 'humidity', 'pressure', 'cloud_cover']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap', fontsize=14)
plt.show()
