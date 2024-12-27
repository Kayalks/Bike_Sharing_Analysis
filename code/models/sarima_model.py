# -*- coding: utf-8 -*-

from pmdarima import auto_arima
from .base_model import TimeSeriesModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SARIMAModel(TimeSeriesModel):
    def fit(self, order=None, seasonal_order=None):
        """Fit SARIMA model with specified or dynamically tuned parameters."""
        try:
            logging.info("Fitting SARIMA model...")
            self.model = auto_arima(
                self.train[self.target],
                seasonal=True,
                m=seasonal_order[3] if seasonal_order else 12,
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
            logging.info("SARIMA model fitted successfully.")
        except Exception as e:
            logging.error(f"Error fitting SARIMA model: {e}")
            raise

    def predict(self):
        """Generate predictions for the test set."""
        try:
            logging.info("Generating SARIMA predictions...")
            return self.model.predict(n_periods=len(self.test))
        except Exception as e:
            logging.error(f"Error generating SARIMA predictions: {e}")
            raise

    def save_predictions(self, predictions, model_name, split_ratio, normalization):
        """Save SARIMA predictions to a CSV file."""
        try:
            if predictions is None or len(predictions) == 0:
                raise ValueError("Predictions must not be empty.")
            logging.info(f"Saving SARIMA predictions to CSV with model: {model_name}, split ratio: {split_ratio}, normalization: {normalization}...")
            output = self.test.copy()
            output[f'{model_name}_predictions'] = predictions
            output.to_csv(f'{model_name}_{split_ratio}_{normalization}_predictions.csv', index=True)
            logging.info("SARIMA predictions saved successfully.")
        except ValueError as ve:
            logging.error(f"Validation error in save_predictions: {ve}")
            raise
        except Exception as e:
            logging.error(f"Error saving SARIMA predictions: {e}")
            raise
