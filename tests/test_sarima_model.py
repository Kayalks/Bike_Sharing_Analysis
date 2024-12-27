# -*- coding: utf-8 -*-
import unittest
from models.sarima_model import SARIMAModel
import pandas as pd
import numpy as np

class TestSARIMAModel(unittest.TestCase):
    def setUp(self):
        data = {'Date': pd.date_range(start='2022-01-01', periods=10, freq='D'), 'Demand': np.random.randint(100, 200, 10)}
        self.data = pd.DataFrame(data)
        self.data.set_index('Date', inplace=True)
        self.train = self.data.iloc[:7]
        self.test = self.data.iloc[7:]

    def test_model_fit(self):
        model = SARIMAModel(self.train, self.test, 'Demand')
        model.fit(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        self.assertIsNotNone(model.model)

    def test_model_predict(self):
        model = SARIMAModel(self.train, self.test, 'Demand')
        model.fit(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        predictions = model.predict()
        self.assertEqual(len(predictions), len(self.test))

if __name__ == '__main__':
    unittest.main()

