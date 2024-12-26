# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 00:27:58 2024

@author: Kayalvili
"""

import pandas as pd
from models import ARIMAModel, SARIMAModel, LSTMModel
from utils.model_utils import evaluate_model, normalize_data

# Load data
data = pd.read_csv("data/processed/demand/demand_data.csv")
data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
data = data.set_index('Date')
data = data.loc['2022':'2023']

# Target column
target = 'Demand'
splits = {'70:30': 0.7, '60:40': 0.6, '80:20': 0.8}
performance_results = []

for normalization in ["none", "minmax", "standard"]:
    for split_name, split_ratio in splits.items():
        train_size = int(len(data) * split_ratio)
        train, test = data.iloc[:train_size], data.iloc[train_size:]

        if normalization == "none":
            train_scaled, test_scaled = train, test
            scaler = None
        else:
            train_scaled, scaler = normalize_data(train[target], method=normalization)
            train[target] = train_scaled

        # ARIMA
        arima = ARIMAModel(train, test, target)
        arima.fit()
        arima_forecast = arima.predict()
        mse, r2 = evaluate_model(test[target], arima_forecast)
        performance_results.append({
            'Model': 'ARIMA', 'Split': split_name, 'Normalization': normalization, 'MSE': mse, 'R-squared': r2
        })
        arima.save_predictions(arima_forecast, 'ARIMA', split_name, normalization)

        # SARIMA
        sarima = SARIMAModel(train, test, target)
        sarima.fit()
        sarima_forecast = sarima.predict()
        mse, r2 = evaluate_model(test[target], sarima_forecast)
        performance_results.append({
            'Model': 'SARIMA', 'Split': split_name, 'Normalization': normalization, 'MSE': mse, 'R-squared': r2
        })
        sarima.save_predictions(sarima_forecast, 'SARIMA', split_name, normalization)

        # LSTM
        if normalization != "none":
            lstm = LSTMModel(train, test, target, scaler)
            lstm.fit()
            lstm_forecast = lstm.predict()
            mse, r2 = evaluate_model(test[target].iloc[5:], lstm_forecast)
            performance_results.append({
                'Model': 'LSTM', 'Split': split_name, 'Normalization': normalization, 'MSE': mse, 'R-squared': r2
            })
            lstm.save_predictions(lstm_forecast, 'LSTM', split_name, normalization)

# Save performance results
performance_df = pd.DataFrame(performance_results)
performance_df.to_csv('model_performance_results.csv', index=False)
print("Model evaluation and predictions saved.")
