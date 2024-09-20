from .datasource import DataSource
from erasure.data.datasets.Dataset import Dataset 
from torch.utils.data import ConcatDataset

class TVDataSource(DataSource):
    def __init__(self, path):
        self.path = path

    
    def get_name(self):
        return self.path.split(".")[-1] 

    def create_data(self):

        parts = self.path.split('.')

        lib = __import__( parts[0] )
        m = lib
        for part in parts[1:-1]:
            m = getattr(m, part)
        
        dataset_class = getattr(m, parts[-1])
    
        try:
            train = dataset_class(train=True, root='resources/data', download=True, transform=lib.transforms.ToTensor())
            test = dataset_class(train=False, root='resources/data', download=True, transform=lib.transforms.ToTensor())
            concat =  ConcatDataset([train, test])
        except:
            raise ValueError(f"Dataset {self.dataset_name} is not available.")


        dataset = Dataset(concat)

        return dataset