from abc import ABC, abstractmethod
from erasure.core.base import Configurable
from erasure.data.datasets.Dataset import Dataset
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
        self.__init_preprocess__(global_ctx)

    def __init_preprocess__(self, global_ctx):
        self.preprocess = []
        preprocesses =  self.params.get('preprocess',[])

        for preprocess in preprocesses:
            current = Local(preprocess)
            self.preprocess.append( global_ctx.factory.get_object( current ) )


    @abstractmethod
    def fetch_raw_data(self):
        """
        Abstract method to fetch raw data (X, y). 
        """
        pass

    def create_data(self):
        """
        Fetch raw data, apply preprocessing, and return the final Dataset.
        """
        X, y = self.fetch_raw_data()


        X, y = self.apply_preprocessing(X, y)

        X = np.array(X, dtype=np.float32) 
        y = np.array(y, dtype=np.float32)  

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        tensor_dataset = TensorDataset(X_tensor, y_tensor)
        concat_dataset = ConcatDataset([tensor_dataset])

        concat_dataset.datasets[0].classes = torch.unique(y_tensor)

        return Dataset(concat_dataset)

    @abstractmethod
    def get_name(self):
        pass

    def check_integrity(self, dataset: Dataset) -> bool:
        """
        Checks that the dataset's data can be iterated over using a DataLoader.
        Returns True if successful, otherwise raises a ValueError.
        """
        try:
            dataloader = DataLoader(dataset.data, batch_size=1)
            
            
            for _, _ in zip(dataloader, range(5)):  
                pass

            return True  
        except Exception as e:
            raise ValueError(f"Dataset from {self.get_name()} failed integrity check: {e}. Dataset.data must be iterable by Pytorch's dataloader.")

    def validate_and_create_data(self) -> Dataset:
        """
        Validates the data integrity before creating the dataset.
        """
        data = self.create_data()

        if not self.check_integrity(data):
            raise ValueError(f"Integrity check failed for data source: {self.get_name()}")

        return data
    
    def apply_preprocessing(self, X, y):
        """
        Apply each preprocessing step to the data (X, y).
        """
        for preprocess in self.preprocess:
            X, y = preprocess.process(X, y)
        return X, y