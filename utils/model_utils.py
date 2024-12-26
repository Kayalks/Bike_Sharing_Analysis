# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 01:23:13 2024

@author: Kayalvili
"""

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def evaluate_model(y_true, y_pred):
    """Calculate MSE and R-squared for model evaluation."""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

def normalize_data(series, method="minmax"):
    """Normalize data using MinMaxScaler or StandardScaler."""
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    return scaler.fit_transform(series.values.reshape(-1, 1)), scaler
