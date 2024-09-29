from abc import abstractmethod, ABC

from erasure.evaluations.manager import Evaluation


class Measure(ABC):

    @abstractmethod
    def process(self, e:Evaluation):
        pass
