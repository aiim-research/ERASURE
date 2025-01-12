from copy import deepcopy

from erasure.core.unlearner import Unlearner


class Evaluation():
    def __init__(self,unlearner: Unlearner, predictor):
        self.data_info = {}
        self._unlearned_model = None
        self.unlearner = unlearner
        self.predictor = deepcopy(predictor)
        #self.forget_set = unlearner.dataset.partitions[default_forget]
        self.data_info['unlearner'] = unlearner.__class__.__name__
        self.data_info['dataset'] = unlearner.dataset.name
        self.data_info['parameters'] = unlearner.params

    def add_value(self, key, value):
        self.data_info[key] = value

    @property
    def unlearned_model(self):
        return deepcopy(self._unlearned_model)

    @unlearned_model.setter
    def unlearned_model(self, value):
        self._unlearned_model = value
