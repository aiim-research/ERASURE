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

class StringContain(Preprocess):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.contains = self.local_config['parameters']['contains']


    def process(self, X, y, z):

        z = 0
        for string in self.contains:
            if string in X:
                z = 1
                break

        return X,y,z
    
    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['columns'] = self.local_config['parameters']['contains']
