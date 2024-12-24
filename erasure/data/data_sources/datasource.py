from abc import ABC, abstractmethod
from erasure.core.base import Configurable
from erasure.data.datasets.DataSplitter import DataSplitter
from erasure.data.datasets.Dataset import DatasetWrapper,DatasetExtendedWrapper
from torch.utils.data import DataLoader
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
import torch
import numpy as np
from torch.utils.data import TensorDataset, ConcatDataset
from erasure.core.factory_base import get_instance_kvargs
import inspect

class DataSource(Configurable):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.__init_preprocess__()

    @abstractmethod
    def create_data(self) -> DatasetWrapper:
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_simple_wrapper(self,data):
        pass

    def get_extended_wrapper(self,data):
        return DatasetExtendedWrapper(self.get_simple_wrapper(data))
    
    def get_wrapper(self, data):
        caller =  inspect.currentframe().f_back.f_locals['self']

        if isinstance(caller,DataSplitter):
            return self.get_extended_wrapper(data)
        else:
            return self.get_simple_wrapper(data)
    
    def get_extended_wrapper(self, data):
        return self.get_extended_wrapper(data)

    def check_integrity(self, dataset: DatasetWrapper) -> bool:
        """
        Checks that the dataset's data can be iterated over using a DataLoader.
        Returns True if successful, otherwise raises a ValueError.
        """
        try:
            dataloader = DataLoader(dataset, batch_size=1)

            for _, _ in zip(dataloader, range(5)): 
                pass

            return True  
        except Exception as e:
            raise ValueError(f"Dataset from {self.get_name()} failed integrity check: {e}. Dataset.data must be iterable by Pytorch's dataloader.")

    def create_and_validate_data(self) -> DatasetWrapper:
        """
        Validates the data integrity before creating the dataset.
        """
        data = self.create_data()

        

        if not self.check_integrity(data):
            raise ValueError(f"Integrity check failed for data source: {self.get_name()}")

        return data
    

    def __init_preprocess__(self):
        self.preprocess = []
        preprocesses =  self.params.get('preprocess',[])

        for preprocess in preprocesses:
            current = Local(preprocess)
            self.preprocess.append( self.global_ctx.factory.get_object( current ) )