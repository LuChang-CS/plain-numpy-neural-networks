from os import name
import numpy as np

from model.model import Model
from model.init import uniform
from model.activation import sigmoid
from model.loss import binary_crossentropy


class Logistic(Model):
    def __init__(self, dim_in, seed=None):
        super().__init__()
        self.w = uniform((dim_in, ), seed=seed)

    def forward(self, x):
        z = np.matmul(x, self.w)
        output = sigmoid(z)
        return output

    def loss(self, y_hat, y):
        return binary_crossentropy(y_hat, y, aggregation='mean')

    def backward(self, x, y_hat, y):
        x_shape_len, y_shape_len = len(x.shape), len(y.shape)
        if y_shape_len < x_shape_len:
            shape = (*y.shape, *[1] * (x_shape_len - y_shape_len))
            y = y.reshape(shape)
            y_hat = y_hat.reshape(shape)
        dw = np.mean(x * (y_hat - y), axis=0)
        return dw


def logistic_sgd(logistic_model, dw, learning_rate=0.01):
    logistic_model.w -= dw * learning_rate
