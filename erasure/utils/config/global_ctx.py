from copy import deepcopy
import os

from erasure.utils.config.file_parser import Config
from erasure.utils.logger import GLogger
import numpy as np
import torch
import random

class Global:

    def __init__(self, config_file):
        # Check that the path to the config file exists
        self.logger = GLogger.getLogger()
        self.logger.info("Creating Global Context for: "+config_file)
        if not os.path.exists(config_file):
            raise ValueError(f'''The provided config file does not exist. PATH: {config_file}''')
        
        self.config = Config.from_json(config_file)
        

    def __setitem__(self, key, value):
        self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]
    
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != 'logger':
                setattr(result, k, deepcopy(v, memo))
            else:
                setattr(result, k, self.logger)
        return result
    
def clean_cfg(cfg):
    if isinstance(cfg,dict):
        new_cfg = {}
        for k in cfg.keys():
            if hasattr(cfg[k],"local_config"):#k == 'oracle' or k == 'dataset':
                new_cfg[k] = clean_cfg(cfg[k].local_config)
            elif isinstance(cfg[k], (list,dict, np.ndarray)):
                new_cfg[k] = clean_cfg(cfg[k])
            else:
                new_cfg[k] = cfg[k]
    elif isinstance(cfg, (list, np.ndarray)):
        new_cfg = []
        for k in cfg:
            new_cfg.append(clean_cfg(k))
    else:
        new_cfg = cfg

    return new_cfg

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # For more deterministic behavior, you can set the following
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False