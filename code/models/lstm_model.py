# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 01:58:30 2024

@author: Kayalvili
"""

from .base_model import TimeSeriesModel
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

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
