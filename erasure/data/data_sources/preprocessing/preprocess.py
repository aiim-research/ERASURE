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
        self.process_X = self.local_config['parameters']['process_X']
        self.process_y = self.local_config['parameters']['process_y']


class CategoricalEncode(Preprocess):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.encoder = get_instance_kvargs(self.local_config['parameters']['encoder']['class'],
                                           self.local_config['parameters']['encoder']['parameters'])

    def process(self, X, y):

        X_encoded = X.copy() 

        if self.process_X: 
            for col in range(X.shape[1]):  
                encoder = copy.deepcopy(self.encoder) 
                column = X[:, col] 
                
                if np.issubdtype(column.dtype, np.object_) or isinstance(column[0], str):
                    column = np.array([str(item).strip() for item in column], dtype=object)  # Convert all to strings and strip whitespaces
                
                X_encoded[:, col] = encoder.fit_transform(column).astype(int)

        if self.process_y:
            y = y.ravel()
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
            return str(value).strip().rstrip(self.character_to_remove)
        
        if self.process_X: 
            for col in range(X.shape[1]):  
                column = X[:, col]  

                if np.issubdtype(column.dtype, np.object_) or isinstance(column[0], str):
                    column = np.array([clean_string(item) for item in column], dtype=object)  
                
                X[:, col] = column 

        if self.process_y:
            y = y.ravel()
            if np.issubdtype(y.dtype, np.object_) or isinstance(y[0], str):
                y = np.array([clean_string(item) for item in y], dtype=object) 
        
        return X, y