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

class reshape_y_z(Preprocess):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.keep_as_y = self.local_config['parameters']['keep_as_y']
        self.move_to_z = self.local_config['parameters']['move_to_z']


    def process(self, X, y, z):
        
        ##  input tensor is of shape
        ##  ( [img], [downloaded_attributes], None)
        ##  if keep_as_y is (0,0), we keep the first column of the first downloaded_attribute

        true_y = y
        for idx in self.keep_as_y:
            true_y = true_y[idx]
         
        z = y
        for idx in self.move_to_z:
            z = z[idx]

        return X, true_y , z
    
    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['keep_as_y'] = self.local_config['parameters'].get('keep_as_y',0)
        self.local_config['parameters']['move_to_z'] = self.local_config['parameters'].get('move_to_z',1)