from abc import ABCMeta, abstractmethod
import copy
from erasure.core.base import Configurable
from erasure.data.datasets.Dataset import DatasetWrapper
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local

class Unlearner(Configurable, metaclass=ABCMeta):

    def __init__(self, global_ctx: Global, local_ctx):
        self.dataset = local_ctx.dataset
        self.predictor = local_ctx.predictor
        self.device = self.predictor.device

        super().__init__(global_ctx, local_ctx)   

    def unlearn(self):
        self.__preprocess__()
        self.info('Unlearning on predictor: '+str(self.predictor))
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
        
