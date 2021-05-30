import os

import wget
import numpy as np

from data.dataset import Dataset
from data.utils import str_label_to_int, to_onehot


class Iris(Dataset):
    def __init__(self, base_path=None):
        super().__init__(base_path)
        self.url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        self.filename = 'iris.data'
        self.x, self.y = self.load()

    def _name(self):
        return 'iris'

    def _download(self):
        wget.download(self.url, self.path)

    def _check_dataset_exist(self):
        return os.path.exists(os.path.join(self.path, self.filename))

    def _load(self):
        filename = os.path.join(self.path, self.filename)
        x = np.loadtxt(filename, usecols=(0, 1, 2, 3), delimiter=',', dtype=np.float32)
        y = np.loadtxt(filename, usecols=4, delimiter=',', dtype=np.str)
        y_label, n_class = str_label_to_int(y)
        y_hot = to_onehot(y_label, n_class)
        return x, y_hot
