from abc import ABC, abstractmethod
from erasure.core.base import Configurable
from erasure.data.datasets.Dataset import DatasetWrapper
from torch.utils.data import DataLoader
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
import torch
import numpy as np
from torch.utils.data import TensorDataset, ConcatDataset
from erasure.core.factory_base import get_instance_kvargs

class DataSource(Configurable):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)

    @abstractmethod
    def create_data(self) -> DatasetWrapper:
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_wrapper(self, data):
        pass

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
    
    def set_preprocess(self, preprocess):
        self.preprocess = preprocess
