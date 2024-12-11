from .datasource import DataSource
from erasure.data.datasets.Dataset import DatasetWrapper 
from torch.utils.data import ConcatDataset
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
import inspect 
import torch
from torchvision.transforms import Compose
import ast 
import re

class TVDataSource(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.dataset = None
        self.path = self.local_config['parameters']['path']
        self.transform = self.local_config['parameters'].get('transform',[])
        self.root_path = self.local_config.get('root_path','resources/data')
    
    def get_name(self):
        return self.path.split(".")[-1] 

    def create_data(self):

        parts = self.path.split('.')

        lib = __import__( parts[0] )
        m = lib
        for part in parts[1:-1]:
            m = getattr(m, part)
        
        dataset_class = getattr(m, parts[-1])

        self.transform = [
            parse_transform(lib.transforms,t) if isinstance(t, str) else t
            for t in self.transform
        ]

        self.transform = Compose(self.transform)

        params = inspect.signature(dataset_class.__init__).parameters

        #try:
        if 'train' in params:
            train = dataset_class(train=True, root=self.root_path, download=True, transform=self.transform)
            test = dataset_class(train=False, root=self.root_path, download=True, transform=self.transform)
        elif 'split' in params:
            train = dataset_class(split='train', root=self.root_path, download=True, transform=self.transform)
            test = dataset_class(split='test', root=self.root_path, download=True, transform=self.transform)
        else:
            raise ValueError("Unknown dataset parameters.")


        concat =  ConcatDataset([train, test])
        concat.classes = torch.unique(torch.tensor(train.targets))
        dataset = self.get_wrapper(concat)

        return dataset
    

    def get_wrapper(self, data):
        return DatasetWrapper(data, self.preprocess)
    
def parse_transform(lib, transform_string):
    """
    Dynamically parses a transform string and instantiates it.

    Example:
        "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
        "RandomHorizontalFlip(p=0.5)"
        "ToTensor"
    """
    try:
        match = re.match(r"(\w+)\((.*)\)", transform_string)
        if match:
            class_name, args = match.groups()
            transform_class = getattr(lib, class_name, None)
            if not transform_class:
                raise ValueError(f"Transform '{class_name}' not found in the provided library")
            parsed_args = eval(f"dict({args})") if args else {}
            return transform_class(**parsed_args)
        else:
            # Handle transforms without arguments (e.g., "ToTensor")
            transform_class = getattr(lib, transform_string, None)
            if not transform_class:
                raise ValueError(f"Transform '{transform_string}' not found in the provided library")
            return transform_class()  # Instantiate without arguments
    except Exception as e:
        raise ValueError(f"Failed to parse transform: {transform_string}. Error: {e}")
