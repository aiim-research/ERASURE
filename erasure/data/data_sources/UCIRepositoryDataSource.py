from .datasource import DataSource
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from ucimlrepo import fetch_ucirepo 
import numpy as np

class UCIRepositoryDataSource(DataSource):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.id = self.local_config['parameters']['id']
        self.dataset = None

    def get_name(self):
        if self.dataset is None:
            self.dataset = fetch_ucirepo(id=self.id)
        return self.dataset.name

    def fetch_raw_data(self):
        if self.dataset is None:
            self.dataset = fetch_ucirepo(id=self.id)

    
        X = self.dataset.data.features.to_numpy()
        y = self.dataset.data.targets.to_numpy()

        return X,y
