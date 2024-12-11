from .datasource import DataSource
from erasure.data.datasets.Dataset import DatasetWrapper 
from torch.utils.data import ConcatDataset
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from datasets import load_dataset, DatasetDict, concatenate_datasets


class HFDatasetWrapper(DatasetWrapper):
    def __init__(self, data, preprocess,label):
        super().__init__(data,preprocess)
        self.label = label

    def __getitem__(self, index: int):
        sample = self.data[index]

        X = {key:value for key,value in sample.items() if key != self.label}
        y = sample[self.label]

        X,y = self.apply_preprocessing(X,y)
        return X,y

class HFDataSource(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.dataset = None
        self.path = self.local_config['parameters']['path']
        self.configuration = self.local_config.get("configuration","")
        self.label = self.local_config['parameters']['label']

    def get_name(self):
        return self.path.split("/")[-1] 

    def create_data(self):
        ds = load_dataset(self.path,self.configuration)

        if isinstance(ds, DatasetDict):
            splits = [ds[split] for split in ds.keys()]
        else:
            splits = [ds]

        concat = ConcatDataset(splits)
        concat.classes = splits[0].unique('label') if 'label' in splits[0] else [-1]
        dataset = self.get_wrapper(concat)

        return dataset
    
    def get_wrapper(self, data):
        return HFDatasetWrapper(data, self.preprocess, self.label)