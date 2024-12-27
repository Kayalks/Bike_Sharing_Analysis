# -*- coding: utf-8 -*-

from pmdarima import auto_arima
from .base_model import TimeSeriesModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ARIMAModel(TimeSeriesModel):
    def fit(self, order=None):
        """Fit ARIMA model with specified order or dynamically tune parameters."""
        try:
            if order is None:
                logging.info("Fitting ARIMA model with dynamic parameter tuning...")
                self.model = auto_arima(
                    self.train[self.target],
                    seasonal=False,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
            else:
                logging.info(f"Fitting ARIMA model with order: {order}...")
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
            logging.info("ARIMA model fitted successfully.")
        except Exception as e:
            logging.error(f"Error fitting ARIMA model: {e}")
            raise

    def predict(self):
        """Generate predictions for the test set."""
        try:
            logging.info("Generating ARIMA predictions...")
            return self.model.predict(n_periods=len(self.test))
        except Exception as e:
            logging.error(f"Error generating ARIMA predictions: {e}")
            raise

    def save_predictions(self, predictions, model_name, split_ratio, normalization):
        """Save ARIMA predictions to a CSV file."""
        try:
            logging.info(f"Saving ARIMA predictions to CSV with model: {model_name}, split ratio: {split_ratio}, normalization: {normalization}...")
            output = self.test.copy()
            output[f'{model_name}_predictions'] = predictions
            output.to_csv(f'{model_name}_{split_ratio}_{normalization}_predictions.csv', index=True)
            logging.info("ARIMA predictions saved successfully.")
        except Exception as e:
            logging.error(f"Error saving ARIMA predictions: {e}")
            raise
