import pickle
import time
from abc import ABCMeta, abstractmethod
from erasure.data.datasets.Dataset import Dataset
from erasure.utils.cfg_utils import retake_dataset
from erasure.core.base import Configurable
from erasure.utils.config.global_ctx import Global, clean_cfg

# TODO: REMOVE
class GGRTL_Trainable(Configurable, metaclass=ABCMeta):

    def __init__(self, global_ctx, local_ctx):
        super().__init__(global_ctx, local_ctx)
        self.dataset = self.local.dataset

    def load_or_create(self, condition=False):
        super().load_or_create(self._to_retrain() or condition)        

    def _to_retrain(self):
        retrain = self.local_config['parameters'].get('retrain', False)
        self.local_config['parameters']['retrain']= False
        return retrain

    def retrain(self):
        self.fit()
        self.write()
        self.context.logger.info(str(self)+" re-saved.")

    def fit(self):
        stime = time.time()
        self.real_fit()
        self.training_time = time.time() - stime
        if hasattr(self, 'device') and self.device is not None:
            self.context.logger.info(f'{self.__class__.__name__} trained on {self.device} in: {self.training_time} secs')   
        else:
            self.context.logger.info(f'{self.__class__.__name__} trained in: {self.training_time} secs')   
        
    def create(self):
        self.fit()

    def write(self):
        filepath = self.context.get_path(self)
        dump = {
            "model" : self.model,
            "config": clean_cfg(self.local_config)
        }
        with open(filepath, 'wb') as f:
          pickle.dump(dump, f)
      
    def read(self):
        dump_file = self.context.get_path(self)        
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
