from abc import ABCMeta, abstractmethod

from erasure.core.base import Configurable
from erasure.evaluations.manager import Evaluation


class Measure(Configurable, metaclass=ABCMeta):

    @abstractmethod
    def process(self, e:Evaluation):
        pass
