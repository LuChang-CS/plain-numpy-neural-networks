import numpy as np


def train_test_split(*arrays, shuffle=True, train_num=1, test_num=None, validation=False, valid_num=0):
    assert train_num >= 1
    size = len(arrays[0])

    idx = np.arange(size)
    if shuffle:
        np.random.shuffle(idx)
    test_end = size if test_num is None else train_num + test_num
    assert test_end <= size

    train_idx, test_idx = idx[:train_num], idx[train_num:test_end]
    train_data = tuple(x[train_idx] for x in arrays)
    test_data = tuple(x[test_idx] for x in arrays)

    if validation:
        assert 1 <= valid_num <= train_num
        train_num = train_num - valid_num
        train_idx, valid_idx = train_idx[:train_num], train_idx[train_num:]
        train_data = tuple(x[train_idx] for x in arrays)
        valid_data = tuple(x[valid_idx] for x in arrays)
    else:
        valid_data = None
    
    return train_data, test_data, valid_data


def str_label_to_int(y, n_class=None):
    labels = {}
    for label in y:
        if label not in labels:
            labels[label] = len(labels)
    if n_class is None:
        n_class = len(labels)
    else:
        assert n_class >= len(labels)
    result = np.zeros((len(y), ), dtype=int)
    for i, label in enumerate(y):
        result[i] = labels[label]
    return result, n_class


def to_onehot(y, n_class=None):
    if n_class is None:
        n_class = np.max(y) + 1
    else:
        assert n_class > np.max(y)
    n = len(y)
    result = np.zeros((n, n_class), dtype=int)
    result[np.arange(len(y)), y] = 1
    return result
