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


class reshape_x_z(Preprocess):
    def __init__(self, global_ctx, local_ctx):
        super().__init__(global_ctx, local_ctx)
        self.keep_as_x = self.local_config['parameters'].get('keep_as_x', [])
        self.move_to_z = self.local_config['parameters'].get('move_to_z', [])


    def process(self, X, y, z):
        if not isinstance(self.keep_as_x, list):
            self.keep_as_x = [self.keep_as_x]
        if not isinstance(self.move_to_z, list):
            self.move_to_z = [self.move_to_z]

        true_x = [X[idx] for idx in self.keep_as_x]

        # If only one element, unpack
        if len(true_x) == 1:
            true_x = true_x[0]

        z = [X[idx] for idx in self.move_to_z]

        # If only one element, unpack
        if len(z) == 1:
            z = z[0]

        return true_x, y, z
        
    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['keep_as_x'] = self.local_config['parameters'].get('keep_as_x', [])
        self.local_config['parameters']['move_to_z'] = self.local_config['parameters'].get('move_to_z', [])
