from erasure.core.base import Configurable
from erasure.core.factory_base import get_instance_kvargs
from erasure.utils.config.global_ctx import Global
from erasure.core.unlearner import Unlearner
import sys

class Evaluator(Configurable):

    def __init__(self, global_ctx: Global, local_ctx):
        super().__init__(global_ctx, local_ctx)

        self.__init_measures__()
        
    def evaluate(self, unlearner: Unlearner):
        e = Evaluation(unlearner)
        for measure in self.measures:
            e = measure.process(e)

        return e

    def __init_measures__(self):
        self.measures= [get_instance_kvargs(measure['class'],measure['parameters']) for measure in self.params['measures']]
        
class Evaluation():
    def __init__(self,unlearner: Unlearner):
        self.data_info = {}
        self.unlearned_model = None
        self.unlearner = unlearner
        self.forget_set = unlearner.dataset.partitions['forget set']
        self.data_info['unlearner'] = unlearner.__class__.__name__
        self.data_info['dataset'] = unlearner.dataset.name
        self.data_info['parameters'] = unlearner.params

    def add_value(self, key, value):
        self.data_info[key] = value