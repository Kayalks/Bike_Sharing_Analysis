# -*- coding: utf-8 -*-

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from .base_model import TimeSeriesModel

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

    def fit(self, n_lags=5, dropout_rate=0.2, learning_rate=0.001, epochs=50, batch_size=32):
        train_scaled = self.scaler.fit_transform(self.train[self.target].values.reshape(-1, 1))
        self.X_train, self.y_train = self.prepare_lstm_data(train_scaled, n_lags=n_lags)

        self.model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(self.X_train.shape[1], 1)),
            Dropout(dropout_rate),
            LSTM(50, activation='relu'),
            Dropout(dropout_rate),
            Dense(1)
        ])

        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self):
        test_scaled = self.scaler.transform(self.test[self.target].values.reshape(-1, 1))
        X_test, _ = self.prepare_lstm_data(test_scaled, n_lags=5)
        predictions = self.model.predict(X_test).flatten()
        return self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    def save_predictions(self, predictions, model_name, split_ratio, normalization):
        output = self.test.iloc[5:].copy()
        output[f'{model_name}_predictions'] = predictions
        output.to_csv(f'{model_name}_{split_ratio}_{normalization}_predictions.csv', index=True)
