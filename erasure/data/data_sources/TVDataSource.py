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
from torchvision import transforms

class TVDataSource(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.dataset = None
        self.path = self.local_config['parameters']['path']
        self.transform = self.local_config['parameters']['transform']
        self.root_path = self.local_config.get('root_path','resources/data')
        self.label_column  = self.local_config['parameters']['label_column']
        self.classes = self.local_config['parameters']['classes']
    
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

        if self.classes is None:
            try:
                labels = torch.tensor(getattr(train, self.label_column))
            except:
                labels = torch.tensor([label for _, label in train])
            concat.classes = torch.unique(labels)  
            #TODO: manage if classes is a one-hot encoded vector multilabeled
        else:
            concat.classes = list(range(0,self.classes))

        dataset = self.get_wrapper(concat)

        return dataset
    

    def get_simple_wrapper(self, data):
        return DatasetWrapper(data, self.preprocess)
    
    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['transform'] = self.local_config['parameters'].get('transform',[])
        self.local_config['parameters']['root_path'] = self.local_config.get('root_path','resources/data')
        self.local_config['parameters']['label_column'] = self.local_config['parameters'].get('label_column', 'targets')
        self.local_config['parameters']['classes'] = self.local_config['parameters'].get('classes', None)
    


class TVDataSourceCelebA(TVDataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.target_type = self.local_config['parameters']['target_type']


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


        train = dataset_class(split='train', root=self.root_path, download=True, transform=self.transform, target_type=self.target_type)
        test = dataset_class(split='test', root=self.root_path, download=True, transform=self.transform, target_type=self.target_type)


        concat =  ConcatDataset([train, test])

        concat.classes = torch.unique(getattr(train, self.label_column).clone().detach())

        dataset = self.get_wrapper(concat)

        return dataset
    
    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['target_type'] = self.local_config['parameters'].get('target_type',[])
    

class TVDataSourceCifar100(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.dataset = None
        self.path = self.local_config['parameters']['path']
        self.transform = self.local_config['parameters']['transform']
        self.root_path = self.local_config.get('root_path','resources/data')
        self.label_column  = self.local_config['parameters']['label_column']
        self.classes = self.local_config['parameters']['classes']
    
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
            # resize the image to 224x224
            transforms.Resize((224, 224)),
            # convert the image to a tensor
            transforms.ToTensor(),
        ]

        # self.transform = [
        #     parse_transform(lib.transforms,t) if isinstance(t, str) else t
        #     for t in self.transform
        # ]

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

        if self.classes is None:
            labels = torch.tensor(getattr(train, self.label_column))
            concat.classes = torch.unique(labels)  
            #TODO: manage if classes is a one-hot encoded vector multilabeled
        else:
            concat.classes = list(range(0,self.classes))

        dataset = self.get_wrapper(concat)

        return dataset
    

    def get_simple_wrapper(self, data):
        return DatasetWrapper(data, self.preprocess)
    
    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['transform'] = self.local_config['parameters'].get('transform',[])
        self.local_config['parameters']['root_path'] = self.local_config.get('root_path','resources/data')
        self.local_config['parameters']['label_column'] = self.local_config['parameters'].get('label_column', 'targets')
        self.local_config['parameters']['classes'] = self.local_config['parameters'].get('classes', None)
        
    
import re
import ast

def parse_transform(lib, transform_string):
    """
    Parses a transform string and instantiates it dynamically.

    Example:
        "Resize((128, 128))"
        "ToTensor"
        "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
        "RandomHorizontalFlip(p=0.5)"
    """
    try:
        match = re.match(r"(\w+)\((.*)\)", transform_string)
        if match:
            class_name, args = match.groups()
            transform_class = getattr(lib, class_name, None)
            if not transform_class:
                raise ValueError(f"Transform '{class_name}' not found in the provided library")

            if args:
                # Special case: Tuples like Resize((128, 128))
                if args.startswith("(") and args.endswith(")"):
                    parsed_args = (ast.literal_eval(args),)  # Ensure it's treated as a tuple
                else:
                    parsed_args = ast.literal_eval(f"({args},)") if "," not in args else ast.literal_eval(f"({args})")

                return transform_class(*parsed_args) if isinstance(parsed_args, tuple) else transform_class(parsed_args)
            else:
                return transform_class()  # No arguments case

        else:
            # Handle transforms without parentheses (e.g., "ToTensor")
            transform_class = getattr(lib, transform_string, None)
            if not transform_class:
                raise ValueError(f"Transform '{transform_string}' not found in the provided library")
            return transform_class()  # Instantiate without arguments

    except Exception as e:
        raise ValueError(f"Failed to parse transform: {transform_string}. Error: {e}")
