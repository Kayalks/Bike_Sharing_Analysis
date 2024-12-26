# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import ParameterGrid
from models.arima_model import ARIMAModel
from models.sarima_model import SARIMAModel
from models.lstm_model import LSTMModel
from utils.model_utils import evaluate_model, normalize_data

# Load the processed demand data
data = pd.read_csv("data/processed/demand/demand_data.csv")
data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
data = data.set_index('Date')
data = data.loc['2022':'2023']  # Filter data for 2022 and 2023

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
    "m": [12]
})

lstm_params = ParameterGrid({
    "n_lags": [3, 5, 7],
    "dropout_rate": [0.2, 0.3],
    "learning_rate": [0.001, 0.0005],
    "epochs": [50, 100],
    "batch_size": [16, 32]
})

# Main pipeline for training and evaluating models
for normalization in normalizations:
    for split_name, split_ratio in splits.items():
        # Train-test split
        train_size = int(len(data) * split_ratio)
        train, test = data.iloc[:train_size], data.iloc[train_size:]

        # Normalize data if required
        if normalization == "none":
            train_scaled, test_scaled = train, test
            s
