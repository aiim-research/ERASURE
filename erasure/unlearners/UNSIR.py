from erasure.unlearners.torchunlearner import TorchUnlearner
from erasure.utils.config.global_ctx import Global
from fractions import Fraction

import torch
import torch.nn.functional as F
from torch import nn

class Noise(nn.Module):
    """
    Noise class to add noise to the model weights. 
    """
    def __init__(self, batch_size, *dim):
        super().__init__()
        self.noise = nn.Parameter(torch.randn(batch_size, *dim), requires_grad=True)

    def forward(self):
        return self.noise

#TODO: check

class UNSIR(TorchUnlearner):
    def __init__(self, global_ctx: Global, local_ctx):
        """
        Initializes the UNSIR class with global and local contexts.

        Args:
            global_ctx (Global): The global context containing configurations and shared resources.
            local_ctx (Local): The local context containing specific configurations for this instance.
        """
        super().__init__(global_ctx, local_ctx)

        self.epochs = self.local.config['parameters']['epochs'] 
        self.ref_data_retain = self.local.config['parameters']['ref_data_retain']
        self.ref_data_forget = self.local.config['parameters']['ref_data_forget']
        self.noise_lr = self.local.config['parameters']['noise_lr']
    
    def __unlearn__(self):
        """
        UNSIR unlearning algorithm for task agnostic setting proposed by https://arxiv.org/pdf/2111.08947, since the original method is thought for class-unlearning setting, we propose here the modified version proposed by https://arxiv.org/pdf/2311.02240. 
        The method is divided in two phases:
        1. In the first phase (Impair), noise is added to perturb the weights of the model.
        2. In the second phase (Repair), the model is trained with the retain data to restore its performance.

        Since the second phase of the model is a simple finetuning, we can use the Finetuning unlearner to implement the second phase and here only the first phase is implemented.
        """

        self.info(f'Starting UNSIR with {self.epochs} epochs for the impair phase')

        retain_loader, _ = self.dataset.get_loader_for(self.ref_data_retain, Fraction('0'))

        forget_loader, _ = self.dataset.get_loader_for(self.ref_data_forget, Fraction('0'))

        for epoch in range(self.epochs):
            running_loss = 0

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

                _, outputs = self.predictor.model(x_retain.to(self.device))
                loss_2 = self.predictor.loss_fn(outputs, y_retain)

                joint_loss = loss_1 + loss_2

                self.predictor.optimizer.zero_grad()
                joint_loss.backward()
                self.predictor.optimizer.step()
                running_loss += joint_loss.item() * x_retain.size(0)
            
            average_train_loss = running_loss / (len(retain_loader) * x_retain.size(0))
            
            self.info(f'UNSIR-1 - epoch = {epoch} ---> var_loss = {average_train_loss:.4f}')

        return self.predictor

    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 1)  # Default 1 epoch
        self.local.config['parameters']['ref_data_retain'] = self.local.config['parameters'].get("ref_data_retain", 'retain set')  # Default reference data is retain
        self.local.config['parameters']['ref_data_forget'] = self.local.config['parameters'].get("ref_data_forget", 'forget')  # Default reference data is forget
        self.local.config['parameters']['noise_lr'] = self.local.config['parameters'].get("noise_lr", 0.01)  # Default noise learning rate is 0.01