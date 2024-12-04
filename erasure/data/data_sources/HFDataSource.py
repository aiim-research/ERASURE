from .datasource import DataSource
from erasure.data.datasets.Dataset import Dataset 
from torch.utils.data import ConcatDataset
import inspect 
from datasets import load_dataset, DatasetDict, concatenate_datasets


class HFDataSource(DataSource):
    def __init__(self, path):
        self.path = path

    def get_name(self):
        return self.path.split("/")[-1] 

    def create_data(self):
        ds = load_dataset("/".join(self.path.split("/")[:-1]), self.path.split("/")[-1])

        if isinstance(ds, DatasetDict):
            print("Splits available:", ds.keys())
            splits = [ds[split] for split in ds.keys()]
        else:
            print("Single Dataset with no splits.")
            splits = [ds]

        concat = ConcatDataset(splits)
        concat.datasets[0].classes = splits[0].unique('label') if 'label' in splits[0] else [-1]
        dataset = Dataset(concat)

        return dataset