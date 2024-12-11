from abc import ABC, abstractmethod
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from erasure.core.factory_base import get_instance_kvargs
from erasure.core.base import Configurable
import numpy as np
import copy
import torch
import re
from erasure.data.preprocessing.preprocess import Preprocess

class IRISpreprocess(Preprocess):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.string_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        self.label_mapping = {label: idx for idx, label in enumerate(set(self.string_labels))}


    def process(self, X, y):
        
        X = torch.tensor(list(X.values()), dtype=torch.float32).unsqueeze(0)
        
        y = torch.tensor(self.label_mapping[y], dtype=torch.long)

        return X, y