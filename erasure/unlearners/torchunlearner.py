

from abc import ABCMeta, abstractmethod
from erasure.core.base import Configurable
from erasure.core.unlearner import Unlearner
from erasure.utils.config.global_ctx import Global

class TorchUnlearner(Unlearner, metaclass=ABCMeta):

    def __preprocess__(self):

        if self.local.config['parameters']['last_trainable_layers'] != -1:

            freezed_layers = self.local.config['parameters']['last_trainable_layers']

            for i, layer in enumerate(list(self.predictor.model.children())):
                if i >= len(list(self.predictor.model.children())) - freezed_layers:
                    for param in layer.parameters():
                        param.requires_grad = True
                    self.info(f'Layer {i} is trainable')
                else:
                    for param in layer.parameters():
                        param.requires_grad = False
                    self.info(f'Layer {i} is frozen')

    def __postprocess__(self):
        pass

    def check_configuration(self):
        self.local.config['parameters']['last_trainable_layers'] = self.local.config['parameters'].get('last_trainable_layers', -1)  # Default last_trainable_layers is -1 (all layers are trainable)
