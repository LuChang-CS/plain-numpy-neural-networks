import torch
import numpy as np
from torch._C import dtype


class DataLoader:
    def __init__(self, data, batch_size=32, shuffle=True, use_torch=False):
        self.data = data

        if hasattr(data, '__len__'):
            self._get_item_fn = self._get_item_multiple
            self._size = len(self.data[0])
        else:
            self._get_item_fn = self._get_item_single
            self._size = len(self.data)

        self.idx = np.arange(self._size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._use_torch = use_torch

        self.batch_num = self._calculate_batch_num()

        self.current = 0
        self.on_epoch_end()

    def use_torch(self):
        self._use_torch = True

    def _get_item_single(self, slices):
        result = self.data[slices]
        if self._use_torch:
            result = torch.from_numpy(result)
        return result

    def _get_item_multiple(self, slices):
        if self._use_torch:
            result = tuple(torch.from_numpy(x[slices]) for x in self.data)
        else:
            result = tuple(x[slices] for x in self.data)
        return result

    def size(self):
        return self._size

    def _calculate_batch_num(self):
        len_ = self._size // self.batch_size
        return len_ if self._size % self.batch_size == 0 else len_ + 1

    def __len__(self):
        return self.batch_num

    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        slices = self.idx[start:end]
        return self._get_item_fn(slices)

    def on_epoch_end(self):
        self.current = 0
        if self.shuffle:
            np.random.shuffle(self.idx)

    def __iter__(self):
        return self

    def __next__(self):
        elems = self.__getitem__(self.current)
        self.current += 1
        if self.current == self.batch_num:
            self.on_epoch_end()
            raise StopIteration
        return elems
