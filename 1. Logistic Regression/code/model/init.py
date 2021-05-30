import numpy as np


def set_seed(seed=None):
    if seed is not None:
        np.random.seed(seed)


def uniform(shape, low=-1.0, high=1.0, seed=None):
    set_seed(seed)
    return np.random.uniform(low, high, shape)


def normal(shape, mean=0.0, std=1.0, seed=None):
    set_seed(seed)
    return np.random.normal(mean, std, shape)


def constant(shape, c=0.0, dtype=np.float32):
    return c * np.ones(shape, dtype=dtype)


def glorot_uniform(shape, gain=1.0, seed=None):
    if not hasattr(shape, '__len__'):
        dims = shape + 1
    elif len(shape) == 1:
        dims = shape[0] + 1
    else:
        dims = shape[0] + shape[-1]
    a = gain * np.sqrt(6 / dims)
    return uniform(shape, -a, a, seed)
