from .datasource import DataSource
from erasure.data.datasets.Dataset import Dataset 
from ucimlrepo import fetch_ucirepo 
from torch.utils.data import TensorDataset, ConcatDataset
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class UCIRepositoryDataSource(DataSource):
    def __init__(self, id):
        self.id = id
        self.dataset = None  
    
    def get_name(self):
        if self.dataset is None:
            self.dataset = fetch_ucirepo(id=self.id)
        return self.dataset.name

    def create_data(self):
        if self.dataset is None:
            self.dataset = fetch_ucirepo(id=self.id)
        
        X = self.dataset.data.features.to_numpy()
        y = self.dataset.data.targets.to_numpy()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        tensor_dataset = TensorDataset(X_tensor, y_tensor)

        concat_dataset = ConcatDataset([tensor_dataset])

        concat_dataset.datasets[0].classes = torch.unique(y_tensor)

        dataset = Dataset(concat_dataset)


        return dataset
