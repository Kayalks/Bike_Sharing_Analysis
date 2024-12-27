# -*- coding: utf-8 -*-
import unittest
from time_series_pipeline import save_best_model, save_performance_metrics
import pickle
import os

class TestPipeline(unittest.TestCase):
    def test_save_best_model(self):
        dummy_model = {'mock_key': 'mock_value'}
        save_best_model(dummy_model, 'ARIMA')
        self.assertTrue(os.path.exists('ARIMA_best_model_2024_01.pkl'))

    def test_save_performance_metrics(self):
        metrics = [{'Model': 'ARIMA', 'MSE': 10}]
        save_performance_metrics(metrics)
        self.assertTrue(os.path.exists('model_performance_metrics_2024_01.json'))

if __name__ == '__main__':
    unittest.main()


