# -*- coding: utf-8 -*-

from pmdarima import auto_arima
from .base_model import TimeSeriesModel

class ARIMAModel(TimeSeriesModel):
    def fit(self, order=None):
        if order is None:
            self.model = auto_arima(
                self.train[self.target],
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
        else:
            self.model = auto_arima(
                self.train[self.target],
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=False,
                start_p=order[0],
                start_d=order[1],
                start_q=order[2]
            )

    def predict(self):
        return self.model.predict(n_periods=len(self.test))

    def save_predictions(self, predictions, model_name, split_ratio, normalization):
        output = self.test.copy()
        output[f'{model_name}_predictions'] = predictions
        output.to_csv(f'{model_name}_{split_ratio}_{normalization}_predictions.csv', index=True)
