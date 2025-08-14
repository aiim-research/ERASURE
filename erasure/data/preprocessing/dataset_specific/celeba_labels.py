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

class Celeba_labels(Preprocess):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.label_columns =  self.local_config['parameters']['label_columns']


    def process(self, X, y, Z):

        attr, identity = y
        y_bin = (attr > 0).to(torch.long)


        if y_bin.dim() == 1:         
            y_sel = y_bin[self.label_columns]      
            k = y_sel.shape[0]
            weights = (2 ** torch.arange(k - 1, -1, -1, device=y_sel.device)).to(y_sel.dtype)
            y_mc = (y_sel * weights).sum().to(torch.long)
        else:                        
            y_sel = y_bin[:, self.label_columns]   
            k = y_sel.shape[1]
            weights = (2 ** torch.arange(k - 1, -1, -1, device=y_sel.device)).to(y_sel.dtype)
            y_mc = (y_sel * weights).sum(dim=1).to(torch.long)

        y_mc = [y_mc, identity]

        return X, y_mc, Z

