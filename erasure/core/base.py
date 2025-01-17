
import hashlib
import os
from pathlib import Path
import pickle
from erasure.utils.config.global_ctx import Global, bcolors
from abc import abstractmethod

from erasure.utils.logger import GLogger


class Base:
    info = GLogger.getLogger().info
    
    def __init__(self, global_ctx: Global):
        self.global_ctx = global_ctx
        
class Configurable(Base):   

    def __init__(self, global_ctx, local_ctx):
        super().__init__(global_ctx)
        self.local = local_ctx
        self.local_config = self.local.config

        self.local.config['parameters'] = self.local.config.get('parameters',{})
        self.params = self.local.config['parameters']

        self.check_configuration()
        
        if not self.__pre_init__():
            self.init()
            self.__post_init__()   
              
        
    def check_configuration(self):
        pass #self.local.config['parameters'] = self.local.config.get('parameters',{})

    def __pre_init__(self):
        return False

    def __post_init__(self):
        pass


    @abstractmethod
    def init(self):
        pass

class Saveable(Configurable):

    CACHE_DIR = 'resources/cached'
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    def check_configuration(self):
        super().check_configuration()
        self.local.config['parameters']['cached'] = self.local.config['parameters'].get('cached',self.global_ctx.cached)

    def __pre_init__(self):
        if not self.local.config['parameters']['cached']:
            return False

        file_name = Saveable.CACHE_DIR + '/'+self.__cfg_hashing(self.local_config['parameters']['alias']) if 'alias' in self.local_config['parameters'] else self.__cfg_hashing()
        
        if Path(file_name).exists(): #TODO: Disable Cache for unlearners
            try :
                with open (file_name, "rb") as file_handle :
                    self.__dict__ = pickle.load (file_handle)
                self.info(f'''{bcolors.FAIL}Loaded Instance from: {bcolors.UNDERLINE}{file_name}{bcolors.ENDC}''')
                return True
            except EOFError as eof_error :
                self.info(eof_error)
                pass
        return False

    def __post_init__(self):

        if not self.local.config['parameters']['cached']:
            return False

        file_name = Saveable.CACHE_DIR + '/'+self.__cfg_hashing(self.local_config['parameters']['alias']) if 'alias' in self.local_config['parameters'] else self.__cfg_hashing()
        self.info(f'''{bcolors.FAIL}Dumped Instance to: {bcolors.UNDERLINE}{file_name}{bcolors.ENDC}''')

        with open (file_name, "wb") as file_handle :
            pickle.dump (self.__dict__, file_handle)

    def __cfg_hashing(self, alias=None):
        dictionary  = __resolve_cfg_with_context__(self)

        cls = self.__class__.__name__ if not alias else alias   #TODO add modules in classname
        md5_hash = hashlib.md5() 

        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f'{parent_key}{sep}{k}' if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        if dictionary is not None:
            payload = f'{cls}_' + '_'.join([f'{key}={value}' for key, value in flatten_dict(dictionary).items()])
            md5_hash.update(payload.encode('utf-8'))           
            return cls+'-'+md5_hash.hexdigest()
        else:
            return cls

def __resolve_cfg_with_context__(inst):
    dictionary = {'globals':dict(sorted(inst.global_ctx.config.globals.items()))}
    

    for k, v in sorted(inst.local.__dict__.items()):
        if isinstance (v, Configurable):
            dictionary[k]=__resolve_cfg_with_context__(v)
        elif(isinstance (v, dict)):
            __nested_sort(v)
            dictionary[k]=dict(sorted(v.items())) #TODO: Ordinare ricorsivamente anche i sotto-dizionari
    
    return dictionary

def __nested_sort(node):
    for k, v in sorted(node.items()):
        if(isinstance (v, dict)):
            __nested_sort(v)
            node[k]=dict(sorted(v.items()))

    


