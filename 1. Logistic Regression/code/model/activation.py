import numpy as np


def sigmoid(x):
    e_x = np.exp(-x)
    x = 1 / (1 + e_x)
    return x


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    x = e_x / np.sum(e_x, axis=-1, keepdims=True)
    return x
