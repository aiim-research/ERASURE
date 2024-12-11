from .datasource import DataSource
from erasure.utils.config.global_ctx import Global
from erasure.data.datasets.Dataset import DatasetWrapper 
from erasure.utils.config.local_ctx import Local
from ucimlrepo import fetch_ucirepo 
from torch.utils.data import ConcatDataset
import torch
import pandas as pd
import numpy as np
from datasets import Dataset

class UCIWrapper(DatasetWrapper):
    def __init__(self, data, preprocess,label):
        super().__init__(data,preprocess)
        self.label = label

    def __getitem__(self, index: int):
        sample = self.data[index]

        X = {key:value for key,value in sample.items() if key!=self.label}
        y = sample[self.label]
        
        X,y = self.apply_preprocessing(X,y)
        
        return X,y


class UCIRepositoryDataSource(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.id = self.local_config['parameters']['id']
        self.dataset = None
        self.label = self.local_config['parameters']['label']

    def get_name(self):
        return self.name

    def create_data(self):

        if self.dataset is None:
            self.dataset = fetch_ucirepo(id=self.id)
        
        self.name = self.dataset.metadata.name if 'name' in self.dataset.metadata else 'Name not found'

        pddataset = pd.DataFrame(self.dataset.data.original)

        hfdataset = Dataset.from_pandas(pddataset)
        
        self.dataset = ConcatDataset( [ hfdataset ] )

        self.dataset.classes = pddataset[self.label].unique()

        return UCIWrapper(self.dataset, self.preprocess, self.label)

    
    def get_wrapper(self, data):
        ##data is a Subset wrapping the data
        ##data.indices contains the indices of the subset
        ##slice the dataset with these indices and wrap it around
        return UCIWrapper(data, self.preprocess, self.label)
