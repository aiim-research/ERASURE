from copy import deepcopy
import json
import os

from erasure.utils.config.file_parser import Config
from erasure.utils.logger import GLogger
import numpy as np
import torch
import random

class Global:

    logger = GLogger.getLogger()
    info = logger.info

    def __init__(self, config_file):

        self.info("Creating Global Context for: " + config_file)
        if not os.path.exists(config_file):
            raise ValueError(f'''The provided config file does not exist. PATH: {config_file}''')
        
        self.config = Config.from_json(config_file)
        self.__setglobals__()

    def __setglobals__(self):
        if not hasattr(self.config, 'globals'):
            self.config.globals={}

        if 'seed' in self.config.globals:
            self.info(f'''Setting seeds to: {self.config.globals['seed']}''' )
            self.set_seed(self.config.globals['seed'])
        else:
            gen_seed = random.SystemRandom().randint(0 , 2**32 - 1)
            self.config.globals['seed'] = gen_seed
            self.info(f'''{bcolors.FAIL}WARNING - SEEDS ARE RANDOMLY GENERATED AS {self.config.globals['seed']} - Add globals[\'seed\'] to the main Cfg to fix them.{bcolors.ENDC}''' )
            self.set_seed(gen_seed)

        if 'cached' not in self.config.globals:
            self.config.globals['cached'] = self.cached = False
        else:
            self.config.globals['cached'] = self.cached = strtobool(self.config.globals['cached'])
            
        self.info(f'''{bcolors.FAIL}Caching System: {self.cached}.{bcolors.ENDC}''' )



    def set_seed(self,seed=0):
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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def strtobool (val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    if not isinstance(val,bool):
        val = val.lower()
        if val in ('y', 'yes', 't', 'true', 'on', '1'):
            return True
        elif val in ('n', 'no', 'f', 'false', 'off', '0'):
            return False
        else:
            raise ValueError("invalid truth value %r" % (val,))
    else:
        return val