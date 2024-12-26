# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

class TimeSeriesModel(ABC):
    def __init__(self, train, test, target):
        self.train = train
        self.test = test
        self.target = target

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def save_predictions(self, predictions, model_name, split_ratio, normalization):
        pass
