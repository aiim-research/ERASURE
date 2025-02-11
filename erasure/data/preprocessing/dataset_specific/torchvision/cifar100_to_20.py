from abc import ABC, abstractmethod
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from erasure.core.factory_base import get_instance_kvargs
from erasure.core.base import Configurable
import numpy as np
import pickle
from erasure.data.preprocessing.preprocess import Preprocess

class CIFAR100preprocess(Preprocess):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)

        meta_file = './resources/data/cifar-100-python/meta'
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f, encoding='latin1')

        train_file = './resources/data/cifar-100-python/train'
        with open(train_file, 'rb') as f:
            train_data = pickle.load(f, encoding='latin1')

        fine_labels_numeric = train_data['fine_labels'] 
        coarse_labels_numeric = train_data['coarse_labels']  

        self.fine_to_coarse_dict = {}
        for fine_idx, coarse_idx in zip(fine_labels_numeric, coarse_labels_numeric):
            self.fine_to_coarse_dict[fine_idx] = coarse_idx


    def process(self, X, y, Z):
        
        Z = y
        y = self.fine_to_coarse_dict[Z]

        return X, y , Z