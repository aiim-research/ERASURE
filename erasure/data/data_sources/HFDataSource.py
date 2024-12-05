from .datasource import DataSource
from erasure.data.datasets.Dataset import Dataset 
from torch.utils.data import ConcatDataset
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from datasets import load_dataset, DatasetDict, concatenate_datasets


class HFDataSource(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.dataset = None
        self.path = self.local_config['parameters']['path']

    def get_name(self):
        return self.path.split("/")[-1] 

    def create_data(self):
        ds = load_dataset("/".join(self.path.split("/")[:-1]), self.path.split("/")[-1])

        if isinstance(ds, DatasetDict):
            splits = [ds[split] for split in ds.keys()]
        else:
            splits = [ds]

        concat = ConcatDataset(splits)
        concat.datasets[0].classes = splits[0].unique('label') if 'label' in splits[0] else [-1]
        dataset = Dataset(concat)

        return dataset