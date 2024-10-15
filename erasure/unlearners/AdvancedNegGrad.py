from erasure.core.unlearner import Unlearner
from erasure.unlearners.torchunlearner import TorchUnlearner
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from erasure.utils.dataset_utils import create_combined_dataloader
from fractions import Fraction

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from erasure.data.datasets.Dataset import Dataset


class AdvancedNegGrad(TorchUnlearner):
    def __init__(self, global_ctx: Global, local_ctx):
        """
        Initializes the AdvancedNegGrad class with global and local contexts.

        Args:
            global_ctx (Global): The global context containing configurations and shared resources.
            local_ctx (Local): The local context containing specific configurations for this instance.
        """
        super().__init__(global_ctx, local_ctx)
        self.epochs = local_ctx.config.get("epochs", 1)  # Default 5 epochs

    def __unlearn__(self):
        """
        Advanced NegGrad unlearning algorithm. The algorithm is based on the NegGrad algorithm, but it also includes the 
        loss of the retained data points in the loss function.
        """

        self.global_ctx.logger.info(f'Starting AdvancedNegGrad with {self.epochs} epochs')

        # TODO: The following code will be changed as train_set = self.dataset.partitions['retain set']
        forget_set = self.dataset.partitions['forget set']
        cfg_dataset = self.dataset.local_config 
        data_manager = self.global_ctx.factory.get_object(Local(cfg_dataset))
        data_manager.revise_split('train', forget_set)

        retain_loader, _ = self.dataset.get_loader_for('train', Fraction('0'))

        forget_loader, _ = self.dataset.get_loader_for('forget set', Fraction('0'))

        dataloader_iterator = iter(forget_loader)

        for epoch in range(self.epochs):
            losses = []
            self.predictor.model.train()

            for X_retain, labels_retain in retain_loader:
                X_retain, labels_retain = X_retain.to(self.device), labels_retain.to(self.device)
                self.predictor.optimizer.zero_grad() 

                try: 
                    (X_forget, labels_forget) = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(forget_loader)
                    (X_forget, labels_forget) = next(dataloader_iterator)
                
                if X_forget.size(0) != X_forget.size(0):
                    continue

                _, output_retain = self.predictor.model(X_retain.to(self.device))
                _, output_forget = self.predictor.model(X_forget.to(self.device))
                
                loss_ascent_forget = -self.predictor.loss_fn(output_forget, labels_forget.to(self.device))
                loss_retain = self.predictor.loss_fn(output_retain, labels_retain.to(self.device))

                # Overall loss
                joint_loss = loss_ascent_forget + loss_retain

                losses.append(joint_loss.to('cpu').detach().numpy())

                joint_loss.backward()
                self.predictor.optimizer.step()
            
            epoch_loss = sum(losses) / len(losses)
            self.global_ctx.logger.info(f'AdvancedNegGrad - epoch = {epoch} ---> var_loss = {epoch_loss:.4f}')

            self.predictor.lr_scheduler.step()
        
        return self.predictor