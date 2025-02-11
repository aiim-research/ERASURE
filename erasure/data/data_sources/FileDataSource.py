from .datasource import DataSource
from erasure.data.datasets.Dataset import DatasetWrapper 
from torch.utils.data import ConcatDataset
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
import inspect 
import torch
from torchvision.transforms import Compose
import pandas as pd

class CSVDataSource(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.path = self.local_config['parameters']['path']
        self.data_columns = self.local_config['parameters']['data_columns']
        self.label_columns  = self.local_config['parameters']['label_columns']

    
    def get_name(self):
        return self.path.split(".")[-1] 

    def create_data(self):
        self.data = pd.read_csv(self.path, index_col = 0)
        self.data_columns = [col for col in self.data.columns if col != self.label_column] if not self.data_columns else self.data_columns
        self.label_columns = [self.data.columns[-1]] if not self.label_columns else self.label_columns

        dataset = CSVDatasetWrapper(self.data, self.label_columns, self.data_columns, self.preprocess)
        return dataset
    

    def get_simple_wrapper(self, data):
        data_csv = self.data.loc[data.indices]
        return CSVDatasetWrapper(data_csv, self.label_columns, self.data_columns, self.preprocess)
    
    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['root_path'] = self.local_config.get('root_path','resources/data')
        self.local_config['parameters']['label_columns'] = self.local_config['parameters'].get('label_columns', 'targets')
        self.local_config['parameters']['data_columns'] = self.local_config['parameters'].get('data_columns', [])
    
  
class CSVDatasetWrapper(DatasetWrapper):
    def __init__(self, data, label_columns, data_columns, preprocess = []):
        self.data = data 
        self.preprocess = preprocess
        self.data_columns = data_columns
        self.label_columns = label_columns
        self.classes =  self.data[self.label_columns[0]].unique() 

    def __realgetitem__(self, index: int):
        row = self.data.iloc[index]  
        x = row[self.data_columns].values  
        y = row[self.label_columns].values
        x = x[0]

        return x, y

    def get_n_classes(self):
        return len(self.classes)
