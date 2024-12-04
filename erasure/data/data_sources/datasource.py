from abc import ABC, abstractmethod
from erasure.data.datasets.Dataset import Dataset
from torch.utils.data import DataLoader

class DataSource(ABC):
    @abstractmethod
    def create_data(self) -> Dataset:
        pass

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