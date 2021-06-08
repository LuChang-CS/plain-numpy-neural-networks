import numpy as np


def log_sum_exp(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    lse = np.log(np.sum(e_x, axis=axis))
    return x_max.squeeze(axis=axis) + lse
