from copy import deepcopy
from erasure.utils.config.local_ctx import Local
from erasure.core.unlearner import Unlearner


class Evaluation():
    def __init__(self,unlearner: Unlearner, predictor):
        self.data_info = {}
        self._unlearned_model = None
        self.unlearner = unlearner
        self.predictor = deepcopy_manual(predictor)
        #self.forget_set = unlearner.dataset.partitions[default_forget]
        self.data_info['unlearner'] = unlearner.__class__.__name__
        self.data_info['dataset'] = unlearner.dataset.name
        self.data_info['parameters'] = unlearner.params

    def add_value(self, key, value):
        self.data_info[key] = value

    @property
    def unlearned_model(self):
        return deepcopy_manual(self._unlearned_model)

    @unlearned_model.setter
    def unlearned_model(self, value):
        self._unlearned_model = value

def deepcopy_manual(predictor):
    if predictor is None:
        return None 
        
    new_current = Local(predictor.global_ctx.config.predictor)
    new_current.dataset = predictor.dataset
    new_current.skip_training = True
    new_pred = predictor.global_ctx.factory.get_object(new_current)
    new_pred.model.load_state_dict(predictor.model.state_dict())

    return new_pred

