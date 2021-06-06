import os
from pathlib import Path


def get_default_base_path():
    home = Path.home()
    default_base_path = os.path.join(home, '.ml_cache')
    return default_base_path


class Dataset:
    def __init__(self, base_path=None):
        self.dataset_name = self._name()
        self.base_path = base_path if base_path is not None \
            else get_default_base_path()
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        self.path = os.path.join(self.base_path, self.dataset_name)

    def _name(self):
        raise NotImplementedError

    def _preload(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        if not self._check_dataset_exist():
            self._download()
            self._after_download()

    def _download(self):
        raise NotImplementedError

    def _after_download(self):
        pass

    def _check_dataset_exist(self):
        raise NotImplementedError

    def _load(self):
        raise NotImplementedError

    def load(self):
        self._preload()
        return self._load()
