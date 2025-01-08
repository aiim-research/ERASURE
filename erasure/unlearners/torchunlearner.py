

from abc import ABCMeta, abstractmethod
from erasure.core.base import Configurable
from erasure.core.unlearner import Unlearner
from erasure.utils.config.global_ctx import Global

class TorchUnlearner(Unlearner, metaclass=ABCMeta):

    def __preprocess__(self):
        pass

    def __postprocess__(self):
        pass

    def check_configuration(self):
        pass 
