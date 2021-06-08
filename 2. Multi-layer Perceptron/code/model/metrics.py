import numpy as np


def _calculate_tp(pred, target):
    return np.sum(np.logical_and(pred == 1, target == 1))


def _to_vector(pred, target):
    if len(pred.shape) >= 2:
        pred = np.argmax(pred, axis=-1)
    if len(target.shape) >= 2:
        target = np.argmax(target, axis=-1)
    return pred, target


def accuracy(pred, target):
    pred, target = _to_vector(pred, target)
    a = np.sum(pred == target)
    n = len(target)
    return a / n


def precision(pred, target):
    tp = _calculate_tp(pred, target)
    n = np.sum(pred == 1)
    return tp / n


def recall(pred, target):
    tp = _calculate_tp(pred, target)
    n = np.sum(target == 1)
    return tp / n


def f1_score(pred, target):
    tp = _calculate_tp(pred, target)
    false = np.sum(pred != target)
    return tp / (tp + 0.5 * false)
