import numpy as np


def sigmoid(x):
    e_x = np.exp(-x)
    x = 1 / (1 + e_x)
    return x


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    x = e_x / np.sum(e_x, axis=axis, keepdims=True)
    return x


def relu(x):
    return np.maximum(x, 0)


def relu_bp(x):
    return (x > 0).astype(x.dtype)
