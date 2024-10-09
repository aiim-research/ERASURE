from abc import ABCMeta, abstractmethod
from erasure.core.base import Configurable
from erasure.data.datasets.Dataset import Dataset
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local

class Unlearner(Configurable, metaclass=ABCMeta):

    def __init__(self, global_ctx: Global, local_ctx):
        self.dataset = local_ctx.dataset
        self.predictor = local_ctx.predictor
        self.device = self.predictor.device

        super().__init__(global_ctx, local_ctx)


    def get_retrained(self, forget_set):
        if not hasattr(self, 'new_predictor'):
            print("No predictor found.")
            cfg_dataset = self.dataset.local_config 
            cfg_model = self.predictor.local_config
    
            #Create Dataset
            data_manager = self.global_ctx.factory.get_object( Local( cfg_dataset))
            data_manager.revise_split('train', forget_set)
        
            #Create Predictor
            current = Local(cfg_model)
            current.dataset = data_manager
            self.new_predictor = self.global_ctx.factory.get_object(current)
            

        return self.new_predictor
    

    def unlearn(self):
        self.__preprocess__()
        new_model = self.__unlearn__()
        self.__postprocess__()
        return new_model


    def __preprocess__(self):
        pass

    @abstractmethod
    def __unlearn__(self):
        pass

    def __postprocess__(self):
        pass
        
