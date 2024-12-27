# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeSeriesModel(ABC):
    def __init__(self, train, test, target):
        """
        Initialize the base TimeSeriesModel with training and testing data.
        """
        try:
            if train is None or test is None or target not in train.columns:
                raise ValueError("Train/test data must not be None, and target column must exist in train dataset.")
            self.train = train
            self.test = test
            self.target = target
            logging.info("Base TimeSeriesModel initialized successfully.")
        except ValueError as ve:
            logging.error(f"Initialization error: {ve}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during initialization: {e}")
            raise

    @abstractmethod
    def fit(self):
        """
        Fit the time series model.
        """
        try:
            logging.info("Fitting the model...")
        except Exception as e:
            logging.error(f"Error in fitting the model: {e}")
            raise

    @abstractmethod
    def predict(self):
        """
        Predict using the fitted time series model.
        """
        try:
            logging.info("Making predictions...")
        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            raise

    @abstractmethod
    def save_predictions(self, predictions, model_name, split_ratio, normalization):
        """
        Save the predictions to a CSV file.
        """
        try:
            if predictions is None or len(predictions) == 0:
                raise ValueError("Predictions must not be empty.")
            logging.info(f"Saving predictions for model {model_name} with split ratio {split_ratio} and normalization {normalization}...")
        except ValueError as ve:
            logging.error(f"Validation error in save_predictions: {ve}")
            raise
        except Exception as e:
            logging.error(f"Error in saving predictions: {e}")
            raise
