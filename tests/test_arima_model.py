# -*- coding: utf-8 -*-
import unittest
from models.arima_model import ARIMAModel
import pandas as pd
import numpy as np

class TestARIMAModel(unittest.TestCase):
    def setUp(self):
        # Create a small synthetic dataset
        data = {'Date': pd.date_range(start='2022-01-01', periods=10, freq='D'), 'Demand': np.random.randint(100, 200, 10)}
        self.data = pd.DataFrame(data)
        self.data.set_index('Date', inplace=True)
        self.train = self.data.iloc[:7]
        self.test = self.data.iloc[7:]

    def test_model_initialization(self):
        model = ARIMAModel(self.train, self.test, 'Demand')
        self.assertIsNotNone(model)

    def test_model_fit(self):
        model = ARIMAModel(self.train, self.test, 'Demand')
        model.fit(order=(1, 1, 1))
        self.assertIsNotNone(model.model)

    def test_model_predict(self):
        model = ARIMAModel(self.train, self.test, 'Demand')
        model.fit(order=(1, 1, 1))
        predictions = model.predict()
        self.assertEqual(len(predictions), len(self.test))

    def test_save_predictions(self):
        model = ARIMAModel(self.train, self.test, 'Demand')
        model.fit(order=(1, 1, 1))
        predictions = model.predict()
        model.save_predictions(predictions, 'ARIMA', '70:30', 'none')
        self.assertTrue(pd.read_csv('ARIMA_70:30_none_predictions.csv').shape[0] > 0)

if __name__ == '__main__':
    unittest.main()
