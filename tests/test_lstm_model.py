# -*- coding: utf-8 -*-


import unittest
from models.lstm_model import LSTMModel
from utils.model_utils import normalize_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TestLSTMModel(unittest.TestCase):
    def setUp(self):
        data = {'Date': pd.date_range(start='2022-01-01', periods=50, freq='H'), 'Demand': np.random.randint(100, 200, 50)}
        self.data = pd.DataFrame(data)
        self.data.set_index('Date', inplace=True)
        self.train = self.data.iloc[:40]
        self.test = self.data.iloc[40:]
        self.scaler = MinMaxScaler()
        self.train_scaled, self.scaler = normalize_data(self.train['Demand'], method="minmax")

    def test_model_fit(self):
        model = LSTMModel(self.train, self.test, 'Demand', self.scaler)
        model.fit(n_lags=24, epochs=5, batch_size=8)
        self.assertIsNotNone(model.model)

    def test_model_predict(self):
        model = LSTMModel(self.train, self.test, 'Demand', self.scaler)
        model.fit(n_lags=24, epochs=5, batch_size=8)
        predictions = model.predict(n_lags=24, forecast_horizon=10)
        self.assertEqual(len(predictions), 10)

if __name__ == '__main__':
    unittest.main()
