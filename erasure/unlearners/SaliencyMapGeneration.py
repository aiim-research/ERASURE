from erasure.unlearners.torchunlearner import TorchUnlearner
from fractions import Fraction
import torch

from erasure.core.factory_base import get_instance_kvargs

import os

class SaliencyMapGeneration(TorchUnlearner):
    def init(self):
        """
        Initializes the NegGrad class with global and local contexts.
        """

        super().init()

        self.ref_data = self.local.config['parameters']['ref_data'] 
        self.predictor.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.predictor.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})
        self.treshold = self.local.config['parameters']['treshold']

        self.save_dir = self.local.config['parameters']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        self.file_name = self.local.config['parameters']['file_name']

    def __unlearn__(self):
        """
        An implementation of the Saliency Mask Generation proposed in the following paper:
        "Fan, Chongyu, et al. "Salun: Empowering machine unlearning via gradient-based weight saliency in both image classification and generation." arXiv preprint arXiv:2310.12508 (2023)."
        
        Codebase taken from this implementation: https://github.com/OPTML-Group/Unlearn-Saliency/tree/master
        """

        self.info(f'Starting Saliency Mask Generation')
        
        forget_loader, _ = self.dataset.get_loader_for(self.ref_data, Fraction('0'))

        self.predictor.model.eval()

        gradients = {
                name: torch.zeros_like(param.data) for name, param in self.predictor.model.named_parameters()
            }

        for i, (image, target) in enumerate(forget_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            # compute output
            _, output_clean = self.predictor.model(image)
            loss = - self.predictor.loss_fn(output_clean, target)

            self.predictor.optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in self.predictor.model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data

        with torch.no_grad():
            for name in gradients:
                gradients[name] = torch.abs_(gradients[name])

        sorted_dict_positions = {}
        hard_dict = {}

        # Concatenate all tensors into a single tensor
        all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the treshold index for the top 10% elements
        treshold_index = int(len(all_elements) * self.treshold)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            treshold_tensor = torch.zeros_like(tensor_ranks)
            treshold_tensor[tensor_ranks < treshold_index] = 1
            treshold_tensor = treshold_tensor.reshape(tensor.shape)
            hard_dict[key] = treshold_tensor
            start_index += num_elements

            torch.save(hard_dict, os.path.join(self.save_dir, self.file_name))
        
        return self.predictor
    
    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['ref_data'] = self.local.config['parameters'].get("ref_data", 'forget')  # Default reference data is forget
        self.local.config['parameters']['optimizer'] = self.local.config['parameters'].get("optimizer", {'class':'torch.optim.Adam', 'parameters':{}})  # Default optimizer is Adam
        self.local.config['parameters']['treshold'] = self.local.config['parameters'].get('treshold', 0.5) # Default treshold is 0.5 (it represent how much of the model will be unlearned)
        self.local.config['parameters']['save_dir'] = self.local.config['parameters'].get("save_dir", '')  # Default save directory is the actual folder
        self.local.config['parameters']['file_name'] = self.local.config['parameters'].get("file_name", 'saliency_map') # Default file name is saliency_map