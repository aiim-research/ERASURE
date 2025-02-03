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
    def __init__(self, data, preprocess,label, data_columns):
        super().__init__(data,preprocess)
        self.label = label
        self.data_columns = data_columns

    def __realgetitem__(self, index: int):
        sample = self.data[index]

        X = torch.Tensor([value for key,value in sample.items() if key in self.data_columns])

        y = sample[self.label]
     
        return X,y


class UCIRepositoryDataSource(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.id = self.local_config['parameters']['id']
        self.dataset = None
        self.label = self.local_config['parameters']['label']
        self.data_columns = self.local_config['parameters']['data_columns']
        self.to_encode = self.local_config['parameters']['to_encode']

    def get_name(self):
        return self.name

    def create_data(self):

        if self.dataset is None:
            self.dataset = fetch_ucirepo(id=self.id)

        pddataset = pd.DataFrame(self.dataset.data.original)

        if not self.data_columns:
            self.data_columns = [col for col in pddataset if col != self.label]
            
        
        self.name = self.dataset.metadata.name if 'name' in self.dataset.metadata else 'Name not found'


        hfdataset = Dataset.from_pandas(pddataset)
        
        self.dataset = ConcatDataset( [ hfdataset ] )

        self.dataset.classes = pddataset[self.label].unique()

        return self.get_simple_wrapper(self.dataset)

    
    def get_simple_wrapper(self, data):
        return UCIWrapper(data, self.preprocess, self.label, self.data_columns)
    

    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['label'] = self.local_config['parameters'].get('label','')
        self.local_config['parameters']['data_columns'] = self.local_config['parameters'].get('data_columns',[])
        self.local_config['parameters']['to_encode'] = self.local_config['parameters'].get('to_encode',[])

##Adult has a lot of errors in its data, so it's best to handle them in a different loader.
class UCI_Adult_DataSource(UCIRepositoryDataSource):
    
    def create_data(self):

        if self.dataset is None:
            self.dataset = fetch_ucirepo(id=self.id)

        pddataset = pd.DataFrame(self.dataset.data.original)

        if not self.data_columns:
            self.data_columns = [col for col in pddataset if col != self.label]

        self.label_mappings = {}
        for column_to_encode in self.to_encode:
            unique_labels = pddataset[column_to_encode].unique()
            self.label_mappings[column_to_encode] = {orig_label: new_label for new_label, orig_label in enumerate(unique_labels)}


            pddataset[column_to_encode] = pddataset[column_to_encode].map(lambda x: self.label_mappings[column_to_encode][x])



            
        pddataset[self.label] = pddataset[self.label].apply(lambda x: 0 if '<' in x else 1)
        
        self.name = self.dataset.metadata.name if 'name' in self.dataset.metadata else 'Name not found'


        hfdataset = Dataset.from_pandas(pddataset)
        
        self.dataset = ConcatDataset( [ hfdataset ] )

        self.dataset.classes = pddataset[self.label].unique()

        return self.get_simple_wrapper(self.dataset)