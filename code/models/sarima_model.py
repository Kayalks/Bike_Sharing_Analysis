# -*- coding: utf-8 -*-

from pmdarima import auto_arima
from .base_model import TimeSeriesModel

class SARIMAModel(TimeSeriesModel):
    def fit(self, order=None, seasonal_order=None):
        self.model = auto_arima(
            self.train[self.target],
            seasonal=True,
            m=12,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            start_p=order[0] if order else None,
            start_d=order[1] if order else None,
            start_q=order[2] if order else None,
            start_P=seasonal_order[0] if seasonal_order else None,
            start_D=seasonal_order[1] if seasonal_order else None,
            start_Q=seasonal_order[2] if seasonal_order else None,
            seasonal_periods=seasonal_order[3] if seasonal_order else 12
        )

    def predict(self):
        return self.model.predict(n_periods=len(self.test))

    def save_predictions(self, predictions, model_name, split_ratio, normalization):
        output = self.test.copy()
        output[f'{model_name}_predictions'] = predictions
        output.to_csv(f'{model_name}_{split_ratio}_{normalization}_predictions.csv', index=True)
