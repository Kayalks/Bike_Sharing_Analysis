# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 00:27:58 2024

@author: Kayalvili
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from pmdarima import auto_arima
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from abc import ABC, abstractmethod

# Base Model Class
class TimeSeriesModel(ABC):
    def __init__(self, train, test, target):
        self.train = train
        self.test = test
        self.target = target

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def save_predictions(self, predictions, model_name, split_ratio, normalization):
        pass

# ARIMA Model Class
class ARIMAModel(TimeSeriesModel):
    def fit(self):
        self.model = auto_arima(
            self.train[self.target], seasonal=False, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True
        )

    def predict(self):
        return self.model.predict(n_periods=len(self.test))

    def save_predictions(self, predictions, model_name, split_ratio, normalization):
        output = self.test.copy()
        output[f'{model_name}_predictions'] = predictions
        output.to_csv(f'{model_name}_{split_ratio}_{normalization}_predictions.csv', index=True)

# SARIMA Model Class
class SARIMAModel(TimeSeriesModel):
    def fit(self):
        self.model = auto_arima(
            self.train[self.target], seasonal=True, m=12, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True
        )

    def predict(self):
        return self.model.predict(n_periods=len(self.test))

    def save_predictions(self, predictions, model_name, split_ratio, normalization):
        output = self.test.copy()
        output[f'{model_name}_predictions'] = predictions
        output.to_csv(f'{model_name}_{split_ratio}_{normalization}_predictions.csv', index=True)

# LSTM Model Class
class LSTMModel(TimeSeriesModel):
    def __init__(self, train, test, target, scaler):
        super().__init__(train, test, target)
        self.scaler = scaler

    def prepare_lstm_data(self, series, n_lags=1):
        X, y = [], []
        for i in range(n_lags, len(series)):
            X.append(series[i - n_lags:i])
            y.append(series[i])
        return np.array(X), np.array(y)

    def fit(self):
        train_scaled = self.scaler.fit_transform(self.train[self.target].values.reshape(-1, 1))
        self.X_train, self.y_train = self.prepare_lstm_data(train_scaled, n_lags=5)

        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.X_train.shape[1], 1)),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=32, verbose=0)

    def predict(self):
        test_scaled = self.scaler.transform(self.test[self.target].values.reshape(-1, 1))
        X_test, _ = self.prepare_lstm_data(test_scaled, n_lags=5)
        predictions = self.model.predict(X_test).flatten()
        return self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    def save_predictions(self, predictions, model_name, split_ratio, normalization):
        output = self.test.iloc[5:].copy()
        output[f'{model_name}_predictions'] = predictions
        output.to_csv(f'{model_name}_{split_ratio}_{normalization}_predictions.csv', index=True)

# Utility Functions
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

def normalize_data(series, method="minmax"):
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    return scaler.fit_transform(series.values.reshape(-1, 1)), scaler

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
