from abc import ABC, abstractmethod
from torch.utils.data import ConcatDataset


class Dataset:

    @abstractmethod
    def __init__(self, data):
        self.data = data 


    def get_n_classes(self):
        return len(self.data.datasets[0].classes)

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index: int):
        pass