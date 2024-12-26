# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 00:27:58 2024

@author: Kayalvili
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data/processed/demand/demand_data.csv")
data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
data = data.set_index('Date')

# Filter for training and testing data (2022 and 2023 only)
data = data.loc['2022':'2023']

# Prepare results storage
performance_results = []

# Function to evaluate model performance
def evaluate_model(y_true, y_pred, model_name, split_ratio, normalization):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    performance_results.append({
        'Model': model_name,
        'Split': split_ratio,
        'Normalization': normalization,
        'MSE': mse,
        'R-squared': r2
    })
    return mse, r2

# Function to save predictions
def save_predictions(data, predictions, model_name, split_ratio, normalization):
    output = data.copy()
    output[f'{model_name}_predictions'] = predictions
    output.to_csv(f'{model_name}_{split_ratio}_{normalization}_predictions.csv', index=True)

# Define splits
splits = {'70:30': 0.7, '60:40': 0.6, '80:20': 0.8}

# Target column
target = 'Demand'

# Normalization functions
def normalize_data(series, method="minmax"):
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()
    series_scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    return series_scaled, scaler

# Loop through each split and normalization method
for normalization in ["none", "minmax", "standard"]:
    for split_name, split_ratio in splits.items():
        train_size = int(len(data) * split_ratio)
        train, test = data.iloc[:train_size], data.iloc[train_size:]

        # Normalize train and test data if needed
        if normalization == "none":
            train_scaled, test_scaled = train[target], test[target]
            train_scaler, test_scaler = None, None
        else:
            train_scaled, train_scaler = normalize_data(train[target], method=normalization)
            test_scaled, test_scaler = normalize_data(test[target], method=normalization)

        # 1. ARIMA
        arima_model = ARIMA(train[target], order=(5, 1, 0))
        arima_result = arima_model.fit()
        arima_forecast = arima_result.forecast(steps=len(test))
        mse, r2 = evaluate_model(test[target], arima_forecast, 'ARIMA', split_name, normalization)
        save_predictions(test, arima_forecast, 'ARIMA', split_name, normalization)

        # 2. SARIMA
        sarima_model = SARIMAX(train[target], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarima_result = sarima_model.fit()
        sarima_forecast = sarima_result.forecast(steps=len(test))
        mse, r2 = evaluate_model(test[target], sarima_forecast, 'SARIMA', split_name, normalization)
        save_predictions(test, sarima_forecast, 'SARIMA', split_name, normalization)

        # 3. LSTM
        def prepare_lstm_data(series, n_lags=1):
            X, y = [], []
            for i in range(n_lags, len(series)):
                X.append(series[i - n_lags:i])
                y.append(series[i])
            return np.array(X), np.array(y)

        if normalization != "none":
            X_train, y_train = prepare_lstm_data(train_scaled, n_lags=5)
            X_test, y_test = prepare_lstm_data(test_scaled, n_lags=5)

            lstm_model = Sequential([
                LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')
            lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

            lstm_forecast = lstm_model.predict(X_test).flatten()
            lstm_forecast_rescaled = test_scaler.inverse_transform(lstm_forecast.reshape(-1, 1)).flatten()
            mse, r2 = evaluate_model(test[target].values[-len(lstm_forecast):], lstm_forecast_rescaled, 'LSTM', split_name, normalization)
            save_predictions(test.iloc[5:], lstm_forecast_rescaled, 'LSTM', split_name, normalization)

        # 4. Ensemble (Average of ARIMA, SARIMA, and LSTM)
        ensemble_forecast = (
            arima_forecast.values + 
            sarima_forecast.values + 
            (lstm_forecast_rescaled[:len(test)] if normalization != "none" else 0)
        ) / (3 if normalization != "none" else 2)
        mse, r2 = evaluate_model(test[target], ensemble_forecast, 'Ensemble', split_name, normalization)
        save_predictions(test, ensemble_forecast, 'Ensemble', split_name, normalization)

# Save performance results
performance_df = pd.DataFrame(performance_results)
performance_df.to_csv('model_performance_results.csv', index=False)

print("Model evaluation and predictions saved.")
