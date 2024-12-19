import pickle
import time
from abc import ABCMeta, abstractmethod
from erasure.core.base import Configurable, Saveable
from erasure.data.datasets.Dataset import DatasetWrapper
from erasure.utils.config.global_ctx import Global

class Trainable(Saveable, metaclass=ABCMeta):

    def __init__(self, global_ctx: Global, local_ctx):
        self.dataset = local_ctx.dataset
        super().__init__(global_ctx, local_ctx)

    def load_or_create(self, condition=False):
        super().load_or_create(self._to_retrain() or condition)        

    def _to_retrain(self):
        retrain = self.local_config['parameters'].get('retrain', False)
        self.local_config['parameters']['retrain']= False
        return retrain

    def retrain(self):
        self.fit()
        self.write()
        self.global_ctx.logger.info(str(self)+" re-saved.")

    def fit(self):
        stime = time.time()
        self.real_fit()
        self.training_time = time.time() - stime
        if hasattr(self, 'device') and self.device is not None:
            self.global_ctx.logger.info(f'{self.__class__.__name__} trained on {self.device} in: {self.training_time} secs')   
        else:
            self.global_ctx.logger.info(f'{self.__class__.__name__} trained in: {self.training_time} secs')   
        
    def create(self):
        self.fit()

    def write(self):
        filepath = self.global_ctx.get_path(self)
        dump = {
            "model" : self.model,
            "config": clean_cfg(self.local_config)
        }
        with open(filepath, 'wb') as f:
          pickle.dump(dump, f)
      
    def read(self):
        dump_file = self.global_ctx.get_path(self)        
        if self.saved:
            with open(dump_file, 'rb') as f:
                dump = pickle.load(f)
                self.model = dump['model']
                #self.local_config = dump['config']

    @abstractmethod
    def real_fit(self):
        pass

    def check_configuration(self):
        super().check_configuration()
        self.local_config['parameters']['fold_id'] =  self.local_config['parameters'].get('fold_id', -1)
        self.fold_id = self.local_config['parameters']['fold_id'] 
    

    