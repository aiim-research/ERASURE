import os
import torch
from torch.utils.data import TensorDataset, ConcatDataset, Subset
from PIL import Image
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from erasure.data.data_sources.datasource import DataSource
from erasure.data.datasets.Dataset import DatasetWrapper
from torchvision import transforms
from torchvision.transforms import Compose
import torchvision
import pandas as pd
import re


class TorchVisionCustomSource(DataSource):
    """ Load dataset from a TorchVision - compatible local source. 
        The source needs to have a folder containing the images and 
        a csv file containing the image paths and any associate labels. """

    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.path = self.local_config['parameters']['path']
        self.transform = self.local_config['parameters']['transform']
        self.file_name = self.local_config['parameters']['file_name']
        self.is_RGB = self.local_config['parameters']['RGB']
        self.label = self.local_config['parameters']['label']


    def get_name(self):
        return "custom source"

    def create_data(self) -> DatasetWrapper:

        data_csv = pd.read_csv(os.path.join(self.path, self.file_name), index_col=0)

        data_csv['path'] = self.path + os.sep + data_csv['path']

        df = DataFrameWithClasses(data_csv, self.label)

        return TorchVisionDatasetWrapper(df)

    def get_wrapper(self, data):
        return DatasetWrapper(data, self.preprocess)

    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['path'] = self.local_config['parameters']['path']
        self.local_config['parameters']['file_name'] = self.local_config['parameters']['file_name']
        self.local_config['parameters']['RGB'] = self.local_config['parameters'].get("RGB",True)
        self.local_config['parameters']['transform'] = self.local_config['parameters'].get("transform","")
        self.local_config['parameters']['label'] = self.local_config['parameters']['label']


class DataFrameWithClasses():
    def __init__(self,data,label):
        self.data = data
        self.label = label
        self.classes =  self.data.loc[:,self.label].unique() 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.iloc[index]


class TorchVisionDatasetWrapper(DatasetWrapper):

    def __init__(self, data, preprocess = []):
        self.data = data 
        self.preprocess = preprocess
    

    def __realgetitem__(self, index: int):
        img_path = os.path.join(self.data.loc[index,'path'])
        y = [value for key, value in self.data.loc[index].items() if key != 'path']

        image = Image.open(img_path).convert('RGB') 
        transform = transforms.ToTensor() 

        image = transform(image)

        return image,y

    def __realgetitem__(self, index: int):
        """Retrieve the image and label for the given index."""
        row = self.data.data.iloc[index]  
        img_path = row['path']
        y = [value for key, value in row.items() if key != 'path']

        image = Image.open(img_path).convert('RGB')
        transform = transforms.ToTensor()

        image = transform(image)

        return image, y


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

    