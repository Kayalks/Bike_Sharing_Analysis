# -*- coding: utf-8 -*-
import unittest
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestDurationFeatureImportance(unittest.TestCase):

    def setUp(self):
        # Load mock dataset from CSV
        try:
            logging.info("Loading mock dataset...")
            self.data = pd.read_csv('mock_test_duration_data.csv')
            self.predictors = ['temperature', 'humidity', 'precipitation', 'apparent temperature', 'pressure', 'cloud_cover', 'wind_speed', 'Public Holiday', 'Working Day', 'Weekday']
            self.X = self.data[self.predictors]
            self.y = self.data['Average Duration']
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(self.X)
            logging.info("Mock dataset loaded and scaled successfully.")
        except FileNotFoundError:
            logging.error("Mock dataset file not found. Ensure 'mock_test_duration_data.csv' exists.")
            raise
        except Exception as e:
            logging.error(f"An error occurred during setup: {e}")
            raise

    def test_random_forest(self):
        try:
            logging.info("Testing Random Forest Regressor...")
            model = RandomForestRegressor(random_state=42, n_estimators=10)
            model.fit(self.X_scaled, self.y)
            predictions = model.predict(self.X_scaled)
            mae = mean_absolute_error(self.y, predictions)
            rmse = mean_squared_error(self.y, predictions, squared=False)
            r2 = r2_score(self.y, predictions)

            self.assertGreater(r2, 0.9, "R2 score for Random Forest is too low!")
            logging.info(f"Random Forest -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
        except Exception as e:
            logging.error(f"Random Forest test failed: {e}")
            raise

    def test_gradient_boosting(self):
        try:
            logging.info("Testing Gradient Boosting Regressor...")
            model = GradientBoostingRegressor(random_state=42, n_estimators=10, learning_rate=0.1)
            model.fit(self.X_scaled, self.y)
            predictions = model.predict(self.X_scaled)
            mae = mean_absolute_error(self.y, predictions)
            rmse = mean_squared_error(self.y, predictions, squared=False)
            r2 = r2_score(self.y, predictions)

            self.assertGreater(r2, 0.9, "R2 score for Gradient Boosting is too low!")
            logging.info(f"Gradient Boosting -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
        except Exception as e:
            logging.error(f"Gradient Boosting test failed: {e}")
            raise

    def test_neural_network(self):
        try:
            logging.info("Testing Neural Network...")
            model = Sequential([
                Dense(64, activation='relu', input_dim=self.X_scaled.shape[1]),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(self.X_scaled, self.y, epochs=10, batch_size=2, verbose=0)
            predictions = model.predict(self.X_scaled).flatten()
            mae = mean_absolute_error(self.y, predictions)
            rmse = mean_squared_error(self.y, predictions, squared=False)
            r2 = r2_score(self.y, predictions)

            self.assertGreater(r2, 0.9, "R2 score for Neural Network is too low!")
            logging.info(f"Neural Network -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
        except Exception as e:
            logging.error(f"Neural Network test failed: {e}")
            raise

if __name__ == '__main__':
    unittest.main()
