import numpy as np


def aggregation_loss(loss, aggregation):
    if aggregation is None:
        return loss
    aggr_fns = {
        'sum': np.sum,
        'mean': np.mean,
    }
    if aggregation not in aggr_fns:
        raise KeyError('Unsupported aggregation function %s', aggregation)
    else:
        fn = aggr_fns[aggregation]
        return fn(loss, axis=0)


def binary_crossentropy(y_hat, y, aggregation='mean'):
    y_hat = np.clip(y_hat, a_min=1e-9, a_max=1 - 1e-9)
    loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    loss = aggregation_loss(loss, aggregation)
    return loss


def sigmoid_crossentropy(logits, y, aggregation='mean'):
    loss = np.log(1 + np.exp(-logits)) + logits * (1 - y)
    loss = aggregation_loss(loss, aggregation)
    return loss
