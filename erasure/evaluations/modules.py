import torch
import torch.nn as nn

from erasure.data.data_sources.datasource import DataSource
from erasure.data.datasets.Dataset import Dataset

from torchvision.datasets import FashionMNIST


class MIAAttack(nn.Module):

    def __init__(self, n_classes):     # ToDo capire chi gli manda il numero di classi come parametro
        super().__init__()
        self.fc1 = nn.Linear(11, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.last_layer = self.fc3

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        intermediate_output = x
        x = self.fc3(x)
        return intermediate_output, x


class MIADataSource(DataSource):

    def __init__(self):
        pass

    def get_name(self):
        return "MIAAttackDataset"

    def create_data(self) -> Dataset:
        torch_dataset = torch.load("tmp/mia.pt")
        return BinaryDataset(torch_dataset)


class BinaryDataset(Dataset):
    def get_n_classes(self):
        return 2