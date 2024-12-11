import os

import torch
from torch.utils.data import TensorDataset, ConcatDataset, Subset

from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from erasure.data.data_sources.datasource import DataSource
from erasure.data.datasets.Dataset import DatasetWrapper


# torch.serialization.add_safe_globals([TensorDataset, ConcatDataset, Subset])


class TorchFileDataSource(DataSource):
    """ Load Dataset from a Torch file """

    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.path = self.params['path']

    def get_name(self):
        return os.path.basename(self.path)

    def create_data(self) -> DatasetWrapper:
        torch_dataset = torch.load(self.path, weights_only=False)
        return TorchFileDataset(torch_dataset)


class TorchFileDataset(DatasetWrapper):
    def get_n_classes(self):
        return self.data.n_classes
