

from abc import ABCMeta, abstractmethod
from erasure.core.base import Configurable
from erasure.core.unlearner import Unlearner
from erasure.utils.config.global_ctx import Global
import torch

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
        if self.local.config['parameters']['mask_path'] is not None:
            self.mask = torch.load(self.local.config['parameters']['mask_path'])
            self.info(f'Loaded mask from {self.local.config["parameters"]["mask_path"]}')

            # Register hooks for parameters
            for name, param in self.predictor.model.named_parameters():
                if name in self.mask:
                    mask_tensor = self.mask[name]
                    
                    # Define the hook
                    def apply_mask_to_grad(grad, mask=mask_tensor):
                        return grad * mask
                    
                    # Register the hook
                    param.register_hook(apply_mask_to_grad)

    def __postprocess__(self):
        pass

    def check_configuration(self):
        self.local.config['parameters']['last_trainable_layers'] = self.local.config['parameters'].get('last_trainable_layers', -1)  # Default last_trainable_layers is -1 (all layers are trainable)
        self.local.config['parameters']['mask_path'] = self.local.config['parameters'].get('model_mask_path', None)
