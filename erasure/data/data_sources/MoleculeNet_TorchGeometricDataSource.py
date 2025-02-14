import torch
from .datasource import DataSource
from torch_geometric.datasets import MoleculeNet
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from erasure.data.datasets.Dataset import DatasetWrapper 
import numpy as np
from torch_geometric.transforms import Pad

class MoleculeWrapper(DatasetWrapper):
    def __init__(self, data, preprocess):
        super().__init__(data,preprocess)

    def __realgetitem__(self, index: int):
        sample = self.data[index]

        sample = Pad(200)(sample)

        #Data(x=[20, 9], edge_index=[2, 40], edge_attr=[40, 3], smiles='[Cl].CC(C)NCC(O)COc1cccc2ccccc12', y=[1, 1])

        X = [sample.x, sample.edge_index, sample.edge_attr]
        y = sample.y

        #X= torch.tensor(X)
     
        return X,y
    
    def get_n_classes(self):
        all_y = torch.cat([data.y for data in self.data], dim=0)
        unique_y = torch.unique(all_y)
        return len(unique_y)
    



class MoleculeNetDataSource(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.dataset = None
        self.max_size = 0
        self.name = self.local_config['parameters']['name']

    
    def get_name(self):
        return self.name


    def create_data(self):
        
        self.dataset = MoleculeNet(root='data/MoleculeNet', name=self.name)


        return MoleculeWrapper(self.dataset, self.preprocess)

    def get_simple_wrapper(self, data):
        return MoleculeWrapper(data, self.preprocess)
    


    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['name'] = self.local_config['parameters']['name']
