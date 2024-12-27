# -*- coding: utf-8 -*-
import unittest
from future_monthly_forecast import get_latest_model_path
import os

class TestForecasting(unittest.TestCase):
    def test_get_latest_model_path(self):
        # Mock files for testing
        open("ARIMA_best_model_2024_01.pkl", 'a').close()
        latest_path = get_latest_model_path("ARIMA")
        self.assertTrue(latest_path.endswith("ARIMA_best_model_2024_01.pkl"))
        os.remove("ARIMA_best_model_2024_01.pkl")

if __name__ == '__main__':
    unittest.main()
