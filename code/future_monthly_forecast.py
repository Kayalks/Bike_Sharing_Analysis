# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from models.arima_model import ARIMAModel
from models.sarima_model import SARIMAModel
from models.lstm_model import LSTMModel
from utils.model_utils import normalize_data
import logging
import pickle
from keras.models import load_model
import numpy as np
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper to fetch latest model path
def get_latest_model_path(model_name, extension="pkl"):
    """Fetch the latest version of the model file."""
    files = glob.glob(f"{model_name}_best_model_*.{extension}")
    if not files:
        raise FileNotFoundError(f"No saved models found for {model_name}.")
    latest_file = max(files, key=lambda x: datetime.strptime(x.split('_')[-1].replace(f'.{extension}', ''), '%Y_%m'))
    return latest_file

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

# Dynamic Configuration
forecast_start = datetime.now().replace(day=1) + relativedelta(months=1)
forecast_horizon = (forecast_start + relativedelta(months=1) - forecast_start).days * 24
output_filename = f"demand_forecast_{forecast_start.strftime('%Y_%m')}.csv"

# Load the best ARIMA model
try:
    logging.info("Loading best ARIMA model...")
    arima_path = get_latest_model_path("ARIMA")
    with open(arima_path, 'rb') as f:
        arima_model = pickle.load(f)
    logging.info(f"Loaded ARIMA model from {arima_path}.")
except Exception as e:
    logging.error(f"Error loading ARIMA model: {e}")
    raise

# Load the best SARIMA model
try:
    logging.info("Loading best SARIMA model...")
    sarima_path = get_latest_model_path("SARIMA")
    with open(sarima_path, 'rb') as f:
        sarima_model = pickle.load(f)
    logging.info(f"Loaded SARIMA model from {sarima_path}.")
except Exception as e:
    logging.error(f"Error loading SARIMA model: {e}")
    raise

# Load the best LSTM model
try:
    logging.info("Loading best LSTM model...")
    lstm_model = load_model("LSTM_best_model.h5")
    logging.info("Loaded LSTM model from LSTM_best_model.h5.")
except Exception as e:
    logging.error(f"Error loading LSTM model: {e}")
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
    forecast_result.to_csv(output_filename)
    logging.info(f"Forecast saved to {output_filename}.")
except Exception as e:
    logging.error(f"Error saving forecast results: {e}")
    raise
