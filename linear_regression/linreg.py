import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class LinearRegression():
    def __init__(self, lr, n_epoch):
        self.lr = lr
        self.n_epoch = n_epoch

    def _epoch(self):
        y_pred = self.predict(self.x)
        dw = - (2 * (self.x.T).dot(self.y - y_pred)) / self.x.shape[0]
        db = - 2 * np.sum(self.y - y_pred) / self.x.shape[0]
        self.w = self.w - self.lr * dw
        self.b = self.b - self.lr * db
        return self

    def fit(self, x, y):
        x.shape[0], self.n = x.shape
        self.w = np.zeros(x.shape[1])
        self.b = 0
        self.x = x
        self.y = y
        for _ in range(self.n_epoch):
            self._epoch()
        return self

    def predict(self, x):
        return x.dot(self.w) + self.b
