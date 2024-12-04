from abc import ABC, abstractmethod
from torch.utils.data import ConcatDataset
import torch

class Dataset:

    @abstractmethod
    def __init__(self, data):
        self.data = data 

    def get_n_classes(self):
        ##access data from Dataset
        ##access datasets[0] because Dataset contains a ConcatDataset and the first one is the Training set
        ##so we get the number of classes from the training set
        n_classes = len(self.data.datasets[0].classes)

        return n_classes

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index: int):
        pass