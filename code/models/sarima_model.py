# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 01:58:18 2024

@author: Kayalvili
"""

from pmdarima import auto_arima
from .base_model import TimeSeriesModel

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
