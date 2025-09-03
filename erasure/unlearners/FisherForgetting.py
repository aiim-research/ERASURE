from erasure.unlearners.torchunlearner import TorchUnlearner
from fractions import Fraction
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import Data
from tqdm import tqdm
from erasure.core.factory_base import get_instance_kvargs

class FisherForgetting(TorchUnlearner):
    def init(self):
        """
        Initializes the Fisher Forgetting class with global and local contexts.
        """
        super().init()

        self.ref_data_retain = self.local.config['parameters']['ref_data']
        self.alpha = self.local.config['parameters'].get('alpha', 1e-6)
        self.ff_epochs = self.local.config['parameters'].get('ff_epochs', 1000)
        self.num_classes = self.dataset.n_classes

        self.task = self.local.config['parameters']['task']

    def compute_fisher_information(self, dataset):
        """
        Computes the Fisher Information Matrix for each parameter using the retain set.
        Now generalized for any type of data without class-specific computation.
        """
        self.predictor.model.eval()
        
        if isinstance(dataset[0][0],Data):
            dataloader = GeometricDataLoader(dataset, batch_size=1, shuffle=False)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        # Initialize gradient accumulators
        for p in self.predictor.model.parameters():
            p.grad_acc = 0
            p.grad2_acc = 0

        processed_batches = 0
        
        for data, orig_target in tqdm(dataloader, total=self.ff_epochs):
            data, orig_target = data.to(self.device), orig_target.to(self.device)
            _, output = self.predictor.model(data)
            prob = F.softmax(output, dim=-1).data

            for y in range(output.shape[1]):
                target = torch.empty_like(orig_target).fill_(y)
                loss = self.predictor.loss_fn(output, target)
                self.predictor.model.zero_grad()
                loss.backward(retain_graph=True)
                for p in self.predictor.model.parameters():
                    if not p.requires_grad or p.grad is None:
                        continue
                    p.grad_acc += (orig_target == target).float() * p.grad.data
                    p.grad2_acc += prob[:, y] * p.grad.data.pow(2)
            
            processed_batches += 1
            if processed_batches >= self.ff_epochs:
                break

        for p in self.predictor.model.parameters():
            p.grad_acc /= processed_batches
            p.grad2_acc /= processed_batches

    def compute_fisher_information_multilabel(self, dataset):
        self.predictor.model.eval()

        if isinstance(dataset[0][0], Data):
            dataloader = GeometricDataLoader(dataset, batch_size=1, shuffle=False)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        for p in self.predictor.model.parameters():
            if p.requires_grad:
                p.grad_acc = torch.zeros_like(p, device=p.device)
                p.grad2_acc = torch.zeros_like(p, device=p.device)

        processed_batches = 0

        for data, orig_target in tqdm(dataloader, total=self.ff_epochs):
            data, orig_target = data.to(self.device), orig_target.to(self.device)
            
            _, outputs = self.predictor.model(data)
            probs = torch.sigmoid(outputs)

            # Loop over classes
            for y in range(outputs.shape[1]):
                self.predictor.model.zero_grad(set_to_none=True)
                outputs[:, y].sum().backward(retain_graph=True)

                pos_frac = orig_target[:, y].mean() 
                fisher_w = (probs[:, y] * (1.0 - probs[:, y])).mean().detach() 

                for p in self.predictor.model.parameters():
                    if not p.requires_grad or p.grad is None:
                        continue
                    g = p.grad.detach()
                    p.grad_acc += pos_frac * g
                    p.grad2_acc += fisher_w * g.pow(2)
            
            processed_batches += 1
            if processed_batches >= self.ff_epochs:
                break

        for p in self.predictor.model.parameters():
            p.grad_acc /= processed_batches
            p.grad2_acc /= processed_batches


    def get_mean_var(self, p, is_base_dist=False, alpha=3e-6):
        var = copy.deepcopy(1./(p.grad2_acc+1e-8))
        if isinstance(var, float):  
            var = torch.tensor(var) 
        var = var.clamp(max=1e3)
        if p.size(0) == self.num_classes:
            var = var.clamp(max=1e2)
        var = alpha * var
        
        if p.ndim > 1:
            var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
        if not is_base_dist:
            mu = copy.deepcopy(p.data0.clone())
        else:
            mu = copy.deepcopy(p.data0.clone())
        if p.size(0) == self.num_classes:
            # Last layer
            var *= 10
        elif p.ndim == 1:
            # BatchNorm
            var *= 10
    #         var*=1
        return mu, var

    def apply_fisher_noise(self):
        """
        Applies Fisher noise to model parameters for selective forgetting.
        """        
        for p in self.predictor.model.parameters():
            
            if not isinstance(p, torch.Tensor):
                continue

            if not hasattr(p, 'data0'):
                p.data0 = copy.deepcopy(p.data.clone())

            if not isinstance(p.grad2_acc, torch.Tensor):
                    continue
            
            mu, var = self.get_mean_var(p)
            p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()

    def __unlearn__(self):
        """
        An implementation of the Fisher Forgetting unlearning algorithm proposed in the following paper:
        "Golatkar, Aditya, Alessandro Achille, and Stefano Soatto. "Eternal sunshine of the spotless net: Selective forgetting in deep networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020."
        
        Codebase taken and rearranged from the original implementation: https://github.com/AdityaGolatkar/SelectiveForgetting/tree/master
        """
        self.info(f'Starting Fisher Forgetting')

        # Get data loaders
        retain_loader, _ = self.dataset.get_loader_for(self.ref_data_retain, Fraction('0'))

        # Compute Fisher Information using retain set
        self.info('Computing Fisher Information Matrix')
        if self.task == 'auto':
            self.compute_fisher_information(retain_loader.dataset)
        elif self.task == 'multilabel':
            self.compute_fisher_information_multilabel(retain_loader.dataset)

        # Apply Fisher noise for selective forgetting
        self.info('Applying Fisher noise for selective forgetting')
        self.apply_fisher_noise()

        return self.predictor

    def check_configuration(self):
        """
        Checks and sets default configuration parameters.
        """
        super().check_configuration()

        self.local.config['parameters']['ref_data'] = self.local.config['parameters'].get("ref_data", 'retain')
        self.local.config['parameters']['alpha'] = self.local.config['parameters'].get("alpha", 1e-6)
        self.local.config['parameters']['task'] = self.local.config['parameters'].get("task", 'auto')  # Default task is auto (single-label classification)
        self.local.config['parameters']['ff_epochs'] = self.local.config['parameters'].get("ff_epochs", 1000)
