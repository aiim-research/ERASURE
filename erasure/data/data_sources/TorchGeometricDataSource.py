import torch
from .datasource import DataSource
import torch_geometric
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from erasure.data.datasets.Dataset import DatasetWrapper 
import numpy as np
from erasure.core.factory_base import get_instance_kvargs
from torch_geometric.transforms import Pad
from torch_geometric.data import Data

class GeometricWrapper(DatasetWrapper):
    def __init__(self, data, preprocess):
        super().__init__(data,preprocess)

    def __realgetitem__(self, index: int):
        sample = self.data[index]

        X = Data(sample.x, sample.edge_index, sample.edge_attr)
        y = sample.y

        y = y.squeeze().long()
     
        return X,y
    
    def get_n_classes(self):
        all_y = torch.cat([data.y for data in self.data], dim=0)
        unique_y = torch.unique(all_y)
        return len(unique_y)
    



class TorchGeometricDataSource(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.dataset = None
    
        self.dataset = get_instance_kvargs(self.local_config['parameters']['datasource']['class'],
                        self.local_config['parameters']['datasource']['parameters'])
        
        self.name = self.local_config['parameters']['datasource']['parameters']['name']

        

    def get_name(self):
        return self.name


    def create_data(self):
        
        #Remove empty graphs
        filtered_data_list = [data for data in self.dataset if data.x is not None and data.x.shape[0] > 0]
        filtered_dataset = self.dataset.__class__(root=self.dataset.root, name=self.name)  
        filtered_dataset.data, filtered_dataset.slices = self.dataset.collate(filtered_data_list)  


        return GeometricWrapper(filtered_dataset, self.preprocess)

    def get_simple_wrapper(self, data):
        return GeometricWrapper(data, self.preprocess)
    


    def check_configuration(self):
        super().check_configuration()
