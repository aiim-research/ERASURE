from abc import ABC, abstractmethod
from erasure.data.datasets.Dataset import Dataset


class DataSource(ABC):
    @abstractmethod
    def create_data(self) -> Dataset:
        pass

    @abstractmethod
    def get_name(self):
        pass