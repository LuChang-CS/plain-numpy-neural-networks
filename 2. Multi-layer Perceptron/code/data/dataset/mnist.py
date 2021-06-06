import os

import wget
import gzip
import numpy as np

from data.dataset import Dataset
from data.utils import to_onehot


class MNIST(Dataset):
    def __init__(self, base_path=None):
        super().__init__(base_path)
        self.base_url = 'http://yann.lecun.com/exdb/mnist/'
        self.filenames = [
            'train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz'
        ]
        self.pased_names = ['train_x.npy', 'train_y.npy', 'test_x.npy', 'test_y.npy']
        self.raw_path = os.path.join(self.path, 'raw')
        self.parsed_path = os.path.join(self.path, 'parsed')

        self.train_data, self.test_data = self.load()

    def _name(self):
        return 'mnist'

    def _reformat_image(self, f):
        f.read(4)  # header
        n_image = int.from_bytes(f.read(4), 'big')  # number of images
        n_row = int.from_bytes(f.read(4), 'big')  # image shape, row
        n_col = int.from_bytes(f.read(4), 'big')  # image shape, column
        u_image = f.read()
        images = np.frombuffer(u_image, dtype=np.uint8).reshape((n_image, n_row, n_col))
        return images

    def _reformat_label(self, f):
        f.read(8)  # header and label number
        u_label = f.read()
        labels = np.frombuffer(u_label, dtype=np.uint8)
        return labels


    def _reformat(self, raw_name, parsed_name, reformat_fn):
        image_path = os.path.join(self.raw_path, raw_name)
        with gzip.open(image_path, 'rb') as f:
            data = reformat_fn(f)
            np.save(os.path.join(self.parsed_path, parsed_name), data)

    def _download(self):
        if not os.path.exists(self.raw_path):
            os.mkdir(self.raw_path)
        for filename in self.filenames:
            url = self.base_url + filename
            print('downloading', url)
            wget.download(self.base_url + filename, self.raw_path)
            print('')

    def _after_download(self):
        if not os.path.exists(self.parsed_path):
            os.mkdir(self.parsed_path)
        self._reformat(self.filenames[0], self.pased_names[0], self._reformat_image)
        self._reformat(self.filenames[1], self.pased_names[1], self._reformat_label)
        self._reformat(self.filenames[2], self.pased_names[2], self._reformat_image)
        self._reformat(self.filenames[3], self.pased_names[3], self._reformat_label)

    def _check_dataset_exist(self):
        for filename in self.filenames:
            if not os.path.exists(os.path.join(self.raw_path, filename)):
                return False
        return True

    def _load(self):
        train_x = np.load(os.path.join(self.parsed_path, self.pased_names[0]))
        train_y = np.load(os.path.join(self.parsed_path, self.pased_names[1]))
        test_x = np.load(os.path.join(self.parsed_path, self.pased_names[2]))
        test_y = np.load(os.path.join(self.parsed_path, self.pased_names[3]))
        return (train_x, train_y), (test_x, test_y)
