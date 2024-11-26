from erasure.unlearners.torchunlearner import TorchUnlearner
from erasure.utils.config.global_ctx import Global
from fractions import Fraction

class AdvancedNegGrad(TorchUnlearner):
    def __init__(self, global_ctx: Global, local_ctx):
        """
        Initializes the AdvancedNegGrad class with global and local contexts.

        Args:
            global_ctx (Global): The global context containing configurations and shared resources.
            local_ctx (Local): The local context containing specific configurations for this instance.
        """

        super().__init__(global_ctx, local_ctx)

    def __unlearn__(self):
        """
        Advanced NegGrad unlearning algorithm proposed by https://arxiv.org/pdf/2311.02240. 
        
        The algorithm is based on the NegGrad algorithm, but it also includes the loss of the retained data points in the loss function.
        """

        self.global_ctx.logger.info(f'Starting AdvancedNegGrad with {self.epochs} epochs')

        retain_loader, _ = self.dataset.get_loader_for(self.ref_data_retain, Fraction('0'))

        forget_loader, _ = self.dataset.get_loader_for(self.ref_data_forget, Fraction('0'))

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

    def check_configuration(self):
        super().check_configuration()

        self.epochs = self.local.config['parameters'].get("epochs", 5)  # Default 5 epoch
        self.ref_data_retain = self.local.config['parameters'].get("ref_data_retain", 'retain set')  # Default reference data is retain
        self.ref_data_forget = self.local.config['parameters'].get("ref_data_forget", 'forget set')  # Default reference data is forget