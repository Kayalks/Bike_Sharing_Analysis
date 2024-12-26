# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from models.arima_model import ARIMAModel
from models.sarima_model import SARIMAModel
from models.lstm_model import LSTMModel
from utils.model_utils import evaluate_model, normalize_data
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Load the processed demand data
    logging.info("Loading processed demand data...")
    data = pd.read_csv("data/processed/demand/demand_data.csv")
    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
    data.set_index('Date', inplace=True)
except FileNotFoundError:
    logging.error("Data file not found. Ensure the file path is correct.")
    raise
except Exception as e:
    logging.error(f"An error occurred while loading the data: {e}")
    raise

# Configuration
target = 'Demand'
splits = {'70:30': 0.7, '60:40': 0.6, '80:20': 0.8}
normalizations = ["none", "minmax", "standard"]
performance_results = []

# Hyperparameter grids
arima_params = ParameterGrid({
    "p": [1, 2, 3],
    "d": [0, 1],
    "q": [1, 2, 3]
})

sarima_params = ParameterGrid({
    "p": [1, 2],
    "d": [0, 1],
    "q": [1, 2],
    "P": [0, 1],
    "D": [0, 1],
    "Q": [0, 1],
    "m": [24]
})

lstm_params = ParameterGrid({
    "n_lags": [24],
    "dropout_rate": [0.2, 0.3],
    "learning_rate": [0.001, 0.0005],
    "epochs": [50, 100],
    "batch_size": [16, 32]
})

# Main pipeline for training and evaluating models
for normalization in normalizations:
    for split_name, split_ratio in splits.items():
        logging.info(f"Processing split: {split_name}, Normalization: {normalization}...")
        try:
            # Train-test split
            train_size = int(len(data) * split_ratio)
            if train_size == 0 or train_size == len(data):
                raise ValueError("Split ratio results in empty training or testing set.")

            train, test = data.iloc[:train_size], data.iloc[train_size:]

            # Normalize data if required
            if normalization == "none":
                train_scaled, test_scaled = train, test
                scaler = None
            else:
                train_scaled, scaler = normalize_data(train[target], method=normalization)
                train[target] = train_scaled

            # ARIMA model with hyperparameter tuning
            best_arima_mse, best_arima_order = float("inf"), None
            for params in arima_params:
                try:
                    arima = ARIMAModel(train, test, target)
                    arima.fit(order=(params["p"], params["d"], params["q"]))
                    arima_forecast = arima.predict()
                    mse, _ = evaluate_model(test[target], arima_forecast)
                    if mse < best_arima_mse:
                        best_arima_mse, best_arima_order = mse, (params["p"], params["d"], params["q"])
                except Exception as e:
                    logging.warning(f"ARIMA tuning failed for params {params}: {e}")

            performance_results.append({
                'Model': 'ARIMA', 'Split': split_name, 'Normalization': normalization, 'MSE': best_arima_mse, 'Best Params': best_arima_order
            })

            # SARIMA model with hyperparameter tuning
            best_sarima_mse, best_sarima_order = float("inf"), None
            for params in sarima_params:
                try:
                    sarima = SARIMAModel(train, test, target)
                    sarima.fit(order=(params["p"], params["d"], params["q"]), seasonal_order=(params["P"], params["D"], params["Q"], params["m"]))
                    sarima_forecast = sarima.predict()
                    mse, _ = evaluate_model(test[target], sarima_forecast)
                    if mse < best_sarima_mse:
                        best_sarima_mse, best_sarima_order = mse, ((params["p"], params["d"], params["q"]), (params["P"], params["D"], params["Q"], params["m"]))
                except Exception as e:
                    logging.warning(f"SARIMA tuning failed for params {params}: {e}")

            performance_results.append({
                'Model': 'SARIMA', 'Split': split_name, 'Normalization': normalization, 'MSE': best_sarima_mse, 'Best Params': best_sarima_order
            })

            # LSTM model with hyperparameter tuning
            best_lstm_mse, best_lstm_params = float("inf"), None
            for params in lstm_params:
                try:
                    lstm = LSTMModel(train, test, target, scaler)
                    lstm.fit(
                        n_lags=params["n_lags"],
                        dropout_rate=params["dropout_rate"],
                        learning_rate=params["learning_rate"],
                        epochs=params["epochs"],
                        batch_size=params["batch_size"]
                    )
                    lstm_forecast = lstm.predict()
                    mse, _ = evaluate_model(test[target].iloc[params["n_lags"]:], lstm_forecast)
                    if mse < best_lstm_mse:
                        best_lstm_mse, best_lstm_params = mse, params
                except Exception as e:
                    logging.warning(f"LSTM tuning failed for params {params}: {e}")

            performance_results.append({
                'Model': 'LSTM', 'Split': split_name, 'Normalization': normalization, 'MSE': best_lstm_mse, 'Best Params': best_lstm_params
            })

        except Exception as e:
            logging.error(f"Error processing split {split_name} with normalization {normalization}: {e}")

# Save performance results
try:
    performance_df = pd.DataFrame(performance_results)
    performance_df.to_csv('model_performance_results.csv', index=False)
    logging.info("Performance results saved successfully.")
except Exception as e:
    logging.error(f"Failed to save performance results: {e}")

logging.info("Model evaluation and hyperparameter tuning complete.")
