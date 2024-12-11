import torch
import torch.nn as nn

from torchvision.datasets import FashionMNIST

from erasure.core.factory_base import *
from erasure.data.data_sources.datasource import DataSource
from erasure.data.datasets.Dataset import Dataset
from erasure.data.datasets.DatasetManager import DatasetManager


class MIAAttack(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(n_classes, 3*n_classes)
        self.fc2 = nn.Linear(3*n_classes, int(1.5*n_classes))
        # self.fc3 = nn.Linear(50, 1)
        self.fc3 = nn.Linear(int(1.5*n_classes), 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.last_layer = self.fc3

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        intermediate_output = x
        x = self.fc3(x)
        #x = self.softmax(x)
        return intermediate_output, x

