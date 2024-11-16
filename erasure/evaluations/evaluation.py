from erasure.core.unlearner import Unlearner


class Evaluation():
    def __init__(self,unlearner: Unlearner):
        self.data_info = {}
        self.unlearned_model = None
        self.unlearner = unlearner
        self.predictor = unlearner.predictor
        self.forget_set = unlearner.dataset.partitions['forget set']
        self.data_info['unlearner'] = unlearner.__class__.__name__
        self.data_info['dataset'] = unlearner.dataset.name
        self.data_info['parameters'] = unlearner.params

    def add_value(self, key, value):
        self.data_info[key] = value