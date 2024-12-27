# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 18:22:13 2024

@author: Kayalvili
"""

import pandas as pd
from datetime import datetime
from models.arima_model import ARIMAModel
from models.sarima_model import SARIMAModel
from models.lstm_model import LSTMModel
from utils.model_utils import normalize_data
import logging
import pickle
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Load and preprocess data
    logging.info("Loading demand data...")
    data = pd.read_csv("data/processed/demand_data.csv")
    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
    data.set_index('Date', inplace=True)
    target = 'Demand'
except FileNotFoundError:
    logging.error("Data file not found. Ensure 'data/processed/demand_data.csv' exists.")
    raise
except Exception as e:
    logging.error(f"An error occurred while loading data: {e}")
    raise

# Configuration
forecast_start = datetime(2024, 1, 1, 0, 0)
forecast_horizon = 31 * 24  # 31 days * 24 hours

# Load the best ARIMA model
try:
    logging.info("Loading best ARIMA model...")
    with open("ARIMA_best_model.pkl", 'rb') as f:
        arima_model = pickle.load(f)
except FileNotFoundError:
    logging.error("ARIMA model file not found. Ensure 'ARIMA_best_model.pkl' exists.")
    raise
except Exception as e:
    logging.error(f"An error occurred while loading the ARIMA model: {e}")
    raise

# Load the best SARIMA model
try:
    logging.info("Loading best SARIMA model...")
    with open("SARIMA_best_model.pkl", 'rb') as f:
        sarima_model = pickle.load(f)
except FileNotFoundError:
    logging.error("SARIMA model file not found. Ensure 'SARIMA_best_model.pkl' exists.")
    raise
except Exception as e:
    logging.error(f"An error occurred while loading the SARIMA model: {e}")
    raise

# Load the best LSTM model
try:
    logging.info("Loading best LSTM model...")
    from keras.models import load_model
    lstm_model = load_model("LSTM_best_model.h5")
except FileNotFoundError:
    logging.error("LSTM model file not found. Ensure 'LSTM_best_model.h5' exists.")
    raise
except Exception as e:
    logging.error(f"An error occurred while loading the LSTM model: {e}")
    raise

# Normalize data for LSTM
try:
    logging.info("Normalizing data for LSTM...")
    data_scaled, scaler = normalize_data(data[target], method="minmax")
except Exception as e:
    logging.error(f"Error during normalization: {e}")
    raise

# Forecast with ARIMA
try:
    logging.info("Forecasting with ARIMA...")
    arima_forecast = arima_model.predict(n_periods=forecast_horizon)
    arima_dates = pd.date_range(start=forecast_start, periods=forecast_horizon, freq='H')
    arima_result = pd.DataFrame({'Date': arima_dates, 'ARIMA_Forecast': arima_forecast}).set_index('Date')
except Exception as e:
    logging.error(f"Error during ARIMA forecasting: {e}")
    raise

# Forecast with SARIMA
try:
    logging.info("Forecasting with SARIMA...")
    sarima_forecast = sarima_model.predict(n_periods=forecast_horizon)
    sarima_dates = pd.date_range(start=forecast_start, periods=forecast_horizon, freq='H')
    sarima_result = pd.DataFrame({'Date': sarima_dates, 'SARIMA_Forecast': sarima_forecast}).set_index('Date')
except Exception as e:
    logging.error(f"Error during SARIMA forecasting: {e}")
    raise

# Forecast with LSTM
try:
    logging.info("Forecasting with LSTM...")
    lstm_forecast = []
    last_sequence = data_scaled[-24:].reshape(1, 24, 1)
    for _ in range(forecast_horizon):
        next_prediction = lstm_model.predict(last_sequence).flatten()[0]
        lstm_forecast.append(next_prediction)
        last_sequence = np.append(last_sequence[:, 1:, :], [[next_prediction]], axis=1)
    lstm_forecast_rescaled = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()
    lstm_dates = pd.date_range(start=forecast_start, periods=forecast_horizon, freq='H')
    lstm_result = pd.DataFrame({'Date': lstm_dates, 'LSTM_Forecast': lstm_forecast_rescaled}).set_index('Date')
except Exception as e:
    logging.error(f"Error during LSTM forecasting: {e}")
    raise

# Combine results and save
try:
    logging.info("Combining forecast results...")
    forecast_result = pd.concat([arima_result, sarima_result, lstm_result], axis=1)
    forecast_result.to_csv('demand_forecast_2024_01.csv')
    logging.info("Forecast for January 2024 saved to 'demand_forecast_2024_01.csv'.")
except Exception as e:
    logging.error(f"Error saving forecast results: {e}")
    raise
