from abc import ABC, abstractmethod
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from erasure.core.factory_base import get_instance_kvargs
from erasure.core.base import Configurable
import numpy as np
import copy
import re

class Preprocess(Configurable):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.process_X = self.local_config['parameters'].get('process_X',True)
        self.process_y = self.local_config['parameters'].get('process_y',True)


class CategoricalEncode(Preprocess):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.encoder = get_instance_kvargs(self.local_config['parameters']['encoder']['class'],
                                           self.local_config['parameters']['encoder']['parameters'])

    def process(self, X, y):
        X_encoded = X.copy()

        if self.process_X:
            encoder = copy.deepcopy(self.encoder)
            if np.issubdtype(X.dtype, np.object_) or isinstance(X.iloc[0], str):
                X = X.apply(lambda item: str(item).strip())
            X_encoded = encoder.fit_transform(X).astype(int)

        if self.process_y:
            if np.issubdtype(y.dtype, np.object_) or isinstance(y[0], str):
                y = np.array([str(item).strip() for item in y], dtype=object)
            y = self.encoder.fit_transform(y).astype(int)

        return X_encoded, y

class RemoveCharacter(Preprocess):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.character_to_remove = self.local_config['parameters']['character']

    def process(self, X, y):
        
        def clean_string(value):
            return str(value).strip().replace(self.character_to_remove, "")
        
        if self.process_X: 
            X = clean_string(X)

        if self.process_y:
            y = clean_string(y)
        
        return X, y
    
class Add(Preprocess):    
    
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.to_add = self.local_config['parameters']['add']

    def process(self, X, y):
        
        if self.process_X: 
            X = X + self.to_add

        if self.process_y:
            y = y + self.to_add
        
        return X, y
