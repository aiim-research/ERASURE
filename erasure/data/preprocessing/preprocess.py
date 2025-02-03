from abc import ABC, abstractmethod
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from erasure.core.factory_base import get_instance_kvargs
from erasure.core.base import Configurable
import numpy as np
import torch
import numbers
import ast

class Preprocess(Configurable):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.process_X = self.local_config['parameters'].get('process_X',False)
        self.process_y = self.local_config['parameters'].get('process_y',False)
        self.process_z = self.local_config['parameters'].get('process_z',False)

    @abstractmethod
    def process(self, X, y, Z):
        pass 

class Encode(Preprocess):
    def process(self, X, y, Z):

        X = self.encode(X) if self.process_X else X
        y = self.encode(y) if self.process_y else y
        Z = self.encode(Z) if self.process_z else Z

        return X,y,Z
        
    def encode(self, tensor):
        return torch.unique(tensor, sorted=True, return_inverse=True)


class RemoveCharacter(Preprocess):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.character_to_remove = self.local_config['parameters']['character']

    def process(self, X, y, Z):
        
        def clean_string(value):
            return str(value).strip().replace(self.character_to_remove, "")
        
        if self.process_X: 
            X = clean_string(X)

        if self.process_y:
            y = clean_string(y)
        
        return X, y, Z
    
class Add(Preprocess):    
    
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.to_add = self.local_config['parameters']['add']

    def process(self, X, y, Z):
        
        if self.process_X: 
            X = X + self.to_add

        if self.process_y:
            y = y + self.to_add
        
        return X, y, Z

class StringToList(Preprocess):
    def __init__(self, global_ctx: Global, local_ctx: Local):
        super().__init__(global_ctx, local_ctx)
        self.x = self.local_config['parameters']['x']
        self.y = self.local_config['parameters']['y']
        self.z = self.local_config['parameters']['z']
        self.max_length = self.local_config['parameters']['max_length']

    def convert_to_list(self, value, flag):
        if flag and isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                raise ValueError(f"Invalid format for value: {value}. Expected a string representation of a list.")
            
        if isinstance(value, numbers.Number):  
            value = [value]
        
        if len(value) < self.max_length:
            value.extend([-1] * (self.max_length - len(value)))

        value = value[:self.max_length]
        return value

    def process(self, X, y, Z):
        X = self.convert_to_list(X, self.x) if self.x else X
        y = self.convert_to_list(y, self.y) if self.y else y
        Z = self.convert_to_list(Z, self.z) if self.z else Z
        return X, y, Z

    def check_configuration(self):
        self.local_config['parameters']['x'] = self.local_config['parameters'].get('x',False) 
        self.local_config['parameters']['y'] = self.local_config['parameters'].get('y',False) 
        self.local_config['parameters']['z'] = self.local_config['parameters'].get('z',False) 

        return super().check_configuration()