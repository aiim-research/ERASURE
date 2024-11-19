from erasure.unlearners.torchunlearner import TorchUnlearner
from erasure.utils.config.global_ctx import Global
from fractions import Fraction

import torch
import torch.nn.functional as F
from torch import nn

from copy import copy

class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class Scrub(TorchUnlearner):

    def __init__(self, global_ctx: Global, local_ctx):
        """
        Initializes the scrub class with global and local contexts.

        Args:
            global_ctx (Global): The global context containing configurations and shared resources.
            local_ctx (Local): The local context containing specific configurations for this instance.
        """
        super().__init__(global_ctx, local_ctx)
        self.epochs = local_ctx.config['parameters'].get("epochs", 1)  # Default 1 epoch
        self.ref_data_retain = local_ctx.config['parameters'].get("ref_data_retain", 'retain set')  # Default reference data is retain
        self.ref_data_forget = local_ctx.config['parameters'].get("ref_data_forget", 'forget set')  # Default reference data is forget
        self.T = local_ctx.config['parameters'].get("T", 4.0)  # Default temperature is 4.0

        self.criterion_div = DistillKL(self.T)

    def __unlearn__(self):
        """
        SCRUB unlearning algorithm for selectively removing specific data from a model as proposed by https://arxiv.org/pdf/2302.09880. The method operates in two main stages:
        
        1. Training on Retain Data: The model (student) is trained using the data to be retained, guided by a frozen version of the model (teacher) to minimize divergence and finetune on retain data.
        2. Divergence on Forget Data: The model then maximizes divergence between its outputs and the teacher's outputs on the data to be forgotten, aiming to remove learned information specific to these samples.

        During each epoch, loss is calculated for both retain and forget data.
        """


        self.global_ctx.logger.info(f'Starting scrub with {self.epochs} epochs')

        retain_loader, _ = self.dataset.get_loader_for(self.ref_data_retain, Fraction('0'))

        forget_loader, _ = self.dataset.get_loader_for(self.ref_data_forget, Fraction('0'))

        self.teacher = copy(self.predictor.model)

        total_loss_retain = 0
        total_loss_forget = 0

        for epoch in range(self.epochs):
            self.predictor.model.train()
            self.teacher.eval()

            # Training with retain data.
            for inputs_retain, labels_retain in retain_loader:

                self.predictor.optimizer.zero_grad()

                inputs_retain, labels_retain = inputs_retain.to(self.device), labels_retain.to(self.device)

                # Forward pass: Student
                _, outputs_retain_student = self.predictor.model(inputs_retain)

                # Forward pass: Teacher
                with torch.no_grad():
                    _, outputs_retain_teacher = self.teacher(inputs_retain)

                # Loss computation
                loss_cls = self.predictor.loss_fn(outputs_retain_student, labels_retain)
                loss_div_retain = self.criterion_div(outputs_retain_student, outputs_retain_teacher)

                loss = loss_cls + loss_div_retain

                # Update total loss and accuracy for retain data.
                total_loss_retain += loss.item()

                # Backward pass
                loss.backward()

                self.predictor.optimizer.step()

            # Training with forget data.
            for inputs_forget, labels_forget in forget_loader:
                inputs_forget, labels_forget = inputs_forget.to(self.device), labels_forget.to(self.device)

                self.predictor.optimizer.zero_grad()
                
                # Forward pass: Student
                _, outputs_forget_student = self.predictor.model(inputs_forget)

                # Forward pass: Teacher
                with torch.no_grad():
                    _, outputs_forget_teacher = self.teacher(inputs_forget)

                # We want to maximize the divergence for the forget data.
                loss_div_forget = -self.criterion_div(outputs_forget_student, outputs_forget_teacher)

                # Update total loss and accuracy for forget data.
                total_loss_forget += loss_div_forget.item()

                # Backward pass
                loss_div_forget.backward()
                self.predictor.optimizer.step()

            avg_loss_retain = total_loss_retain / len(retain_loader)

            avg_loss_forget = total_loss_forget / len(forget_loader)

            self.global_ctx.logger.info(f'scrub - epoch = {epoch} ---> loss_retain = {avg_loss_retain:.4f} - loss_forget = {avg_loss_forget:.4f}')

                
        return self.predictor