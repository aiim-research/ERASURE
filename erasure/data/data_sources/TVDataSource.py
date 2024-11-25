from .datasource import DataSource
from erasure.data.datasets.Dataset import Dataset 
from torch.utils.data import ConcatDataset
import inspect 

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

        print(f"dataset_class is {dataset_class}")

        params = inspect.signature(dataset_class.__init__).parameters

    
        #try:
        if 'train' in params:
            train = dataset_class(train=True, root='resources/data', download=True, transform=lib.transforms.ToTensor())
            test = dataset_class(train=False, root='resources/data', download=True, transform=lib.transforms.ToTensor())
        elif 'split' in params:
            train = dataset_class(split='train', root='resources/data', download=True, transform=lib.transforms.ToTensor())
            test = dataset_class(split='test', root='resources/data', download=True, transform=lib.transforms.ToTensor())
        else:
            raise ValueError("Unknown dataset parameters.")
        #except:
            #raise ValueError(f"Dataset {dataset_class} is not available.")

        concat =  ConcatDataset([train, test])
        dataset = Dataset(concat)

        return dataset
