# -*- coding: utf-8 -*-

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from .base_model import TimeSeriesModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LSTMModel(TimeSeriesModel):
    def __init__(self, train, test, target, scaler):
        """
        Initialize the LSTM model with train, test datasets, and a scaler.
        """
        try:
            super().__init__(train, test, target)
            self.scaler = scaler
            logging.info("LSTMModel initialized successfully.")
        except Exception as e:
            logging.error(f"Error during LSTMModel initialization: {e}")
            raise

    def prepare_lstm_data(self, series, n_lags=1):
        """
        Prepare data sequences for LSTM input.
        """
        try:
            logging.info(f"Preparing LSTM data with n_lags={n_lags}...")
            X, y = [], []
            for i in range(n_lags, len(series)):
                X.append(series[i - n_lags:i])
                y.append(series[i])
            logging.info("LSTM data prepared successfully.")
            return np.array(X), np.array(y)
        except Exception as e:
            logging.error(f"Error preparing LSTM data: {e}")
            raise

    def fit(self, n_lags=5, dropout_rate=0.2, learning_rate=0.001, epochs=50, batch_size=32):
        """
        Train the LSTM model with the specified parameters.
        """
        try:
            logging.info("Scaling training data for LSTM...")
            train_scaled = self.scaler.fit_transform(self.train[self.target].values.reshape(-1, 1))
            self.X_train, self.y_train = self.prepare_lstm_data(train_scaled, n_lags=n_lags)

            logging.info("Building the LSTM model...")
            self.model = Sequential([
                LSTM(50, activation='relu', return_sequences=True, input_shape=(self.X_train.shape[1], 1)),
                Dropout(dropout_rate),
                LSTM(50, activation='relu'),
                Dropout(dropout_rate),
                Dense(1)
            ])

            optimizer = Adam(learning_rate=learning_rate)
            self.model.compile(optimizer=optimizer, loss='mse')

            logging.info("Training the LSTM model...")
            self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            logging.info("LSTM model trained successfully.")
        except Exception as e:
            logging.error(f"Error during LSTM model training: {e}")
            raise

    def predict(self):
        """
        Generate predictions using the trained LSTM model.
        """
        try:
            logging.info("Scaling test data for LSTM prediction...")
            test_scaled = self.scaler.transform(self.test[self.target].values.reshape(-1, 1))
            X_test, _ = self.prepare_lstm_data(test_scaled, n_lags=5)

            logging.info("Generating LSTM predictions...")
            predictions = self.model.predict(X_test).flatten()
            return self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        except Exception as e:
            logging.error(f"Error during LSTM prediction: {e}")
            raise

    def save_predictions(self, predictions, model_name, split_ratio, normalization):
        """
        Save the LSTM predictions to a CSV file.
        """
        try:
            if predictions is None or len(predictions) == 0:
                raise ValueError("Predictions must not be empty.")
            logging.info(f"Saving LSTM predictions for model {model_name}, split ratio {split_ratio}, and normalization {normalization}...")
            output = self.test.iloc[5:].copy()
            output[f'{model_name}_predictions'] = predictions
            output.to_csv(f'{model_name}_{split_ratio}_{normalization}_predictions.csv', index=True)
            logging.info("LSTM predictions saved successfully.")
        except ValueError as ve:
            logging.error(f"Validation error in save_predictions: {ve}")
            raise
        except Exception as e:
            logging.error(f"Error saving LSTM predictions: {e}")
            raise
