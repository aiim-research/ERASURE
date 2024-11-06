from erasure.unlearners.torchunlearner import TorchUnlearner
from erasure.utils.config.global_ctx import Global
from fractions import Fraction

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

class Noise(nn.Module):
    """
    Noise class to add noise to the model weights. 
    """
    def __init__(self, batch_size, *dim):
        super().__init__()
        self.noise = nn.Parameter(torch.randn(batch_size, *dim), requires_grad=True)

    def forward(self):
        return self.noise

class UNSIR(TorchUnlearner):
    def __init__(self, global_ctx: Global, local_ctx):
        """
        Initializes the UNSIR class with global and local contexts.

        Args:
            global_ctx (Global): The global context containing configurations and shared resources.
            local_ctx (Local): The local context containing specific configurations for this instance.
        """
        super().__init__(global_ctx, local_ctx)
        self.epochs_1 = local_ctx.config['parameters'].get("epochs_1", 1)  # Default 1 epoch
        self.epochs_2 = local_ctx.config['parameters'].get("epochs_2", 1) # Default 1 epoch
        self.ref_data_retain = local_ctx.config['parameters'].get("ref_data_retain", 'retain set')  # Default reference data is retain
        self.ref_data_forget = local_ctx.config['parameters'].get("ref_data_forget", 'forget set')  # Default reference data is forget
        self.noise_lr = local_ctx.config['parameters'].get("noise_lr", 0.01)  # Default noise learning rate is 0.01
    def __unlearn__(self):
        """
        UNSIR unlearning algorithm for task agnostic setting proposed by https://arxiv.org/pdf/2311.02240, since the original method is thought for class-unlearning setting. 
        The method is divided in two phases:
        1. In the first phase (Impair), noise is added to perturb the weights of the model.
        2. In the second phase (Repair), the model is trained with the retain data to restore its performance.
        """

        self.global_ctx.logger.info(f'Starting UNSIR with {self.epochs_1} epochs for the impair phase and {self.epochs_2} epochs for the repair phase')

        retain_loader, _ = self.dataset.get_loader_for(self.ref_data_retain, Fraction('0'))

        forget_loader, _ = self.dataset.get_loader_for(self.ref_data_forget, Fraction('0'))

        for epoch in range(self.epochs_1):
            running_loss = 0
            self.predictor.optimizer.zero_grad()

            for batch_idx, ((x_retain, y_retain), (x_forget, y_forget)) in enumerate(zip(retain_loader, forget_loader)):
                y_retain = y_retain.to(self.device)
                batch_size_forget = y_forget.size(0)

                if x_retain.size(0) != retain_loader.batch_size or x_forget.size(0) != forget_loader.batch_size:
                    continue

                # Initialize the noise.
                noise_dim = x_retain.size(1), x_retain.size(2), x_retain.size(3)
                noise = Noise(batch_size_forget, *noise_dim).to(self.device)
                noise_optimizer = torch.optim.Adam(noise.parameters(), lr=self.noise_lr)
                noise_tensor = noise()[:batch_size_forget]

                # Update the noise for increasing the loss value.
                for _ in range(5):
                    _, outputs = self.predictor.model(noise_tensor)
                    with torch.no_grad():
                        _, target_logits = self.predictor.model(x_forget.to(self.device))
                    # Maximize the similarity between noise data and forget features.
                    loss_noise = -F.mse_loss(outputs, target_logits)

                    # Backpropagate to update the noise.
                    noise_optimizer.zero_grad()
                    loss_noise.backward(retain_graph=True)
                    noise_optimizer.step()

                # Train the model with noise and retain image
                noise_tensor = torch.clamp(noise_tensor, 0, 1).detach().to(self.device)
                _, outputs = self.predictor.model(noise_tensor.to(self.device))
                loss_1 = self.predictor.loss_fn(outputs, y_retain)

                outputs = self.predictor.model(x_retain.to(self.device))
                loss_2 = self.predictor.loss_fn(outputs, y_retain)

                joint_loss = loss_1 + loss_2

                joint_loss.backward()
                self.predictor.optimizer.step()
                running_loss += joint_loss.item() * x_retain.size(0)

            average_train_loss = running_loss / (len(retain_loader) * x_retain.size(0))

            for epoch in range(self.epochs_2):
                running_loss = 0
                self.predictor.optimizer.zero_grad()

                for batch_idx, (x_retain, y_retain) in enumerate(retain_loader):
                    y_retain = y_retain.to(self.device)

                    # Classification Loss
                    _, outputs_retain = self.predictor.model(x_retain.to(self.device))
                    classification_loss = self.predictor.loss_fn(outputs_retain, y_retain)

                    classification_loss.backward()
                    self.predictor.optimizer.step()

                    running_loss += classification_loss.item() * x_retain.size(0)
                average_epoch_loss = running_loss / (len(retain_loader) * x_retain.size(0))
        return self.predictor