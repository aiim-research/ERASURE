from erasure.unlearners.torchunlearner import TorchUnlearner
from erasure.utils.config.global_ctx import Global
from fractions import Fraction


class NegGrad(TorchUnlearner):
    def __init__(self, global_ctx: Global, local_ctx):
        """
        Initializes the NegGrad class with global and local contexts.

        Args:
            global_ctx (Global): The global context containing configurations and shared resources.
            local_ctx (Local): The local context containing specific configurations for this instance.
        """

        super().__init__(global_ctx, local_ctx)

        self.epochs = self.local.config['parameters']['epochs']  
        self.ref_data = self.local.config['parameters']['ref_data'] 

    def __unlearn__(self):
        """
        An implementation of the Negative Gradient unlearning algorithm proposed in the following paper:
        "Golatkar, A., Achille, A. and Soatto, S., 2019. Eternal sunshine of the spotless net: Selective forgetting in deep networks. In 2020 IEEE. In CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 9301-9309)."
        
        Codebase taken from this implementation: https://github.com/ndb796/MachineUnlearning
        """

        self.global_ctx.logger.info(f'Starting NegGrad with {self.epochs} epochs')

        forget_loader, _ = self.dataset.get_loader_for(self.ref_data, Fraction('0'))

        for epoch in range(self.epochs):
            losses, preds, labels_list = [], [], []
            self.predictor.model.train()

            for X, labels in forget_loader:
                X, labels = X.to(self.device), labels.to(self.device)
                self.predictor.optimizer.zero_grad() 

                _, pred = self.predictor.model(X)

                loss = -self.predictor.loss_fn(pred, labels)
                losses.append(loss.to('cpu').detach().numpy())

                loss.backward()

                labels_list += list(labels.squeeze().long().detach().to('cpu').numpy())
                preds += list(pred.squeeze().detach().to('cpu').numpy())

                self.predictor.optimizer.step()
            
            epoch_loss = sum(losses) / len(losses)
            self.global_ctx.logger.info(f'NegGrad - epoch = {epoch} ---> var_loss = {epoch_loss:.4f}')

            self.predictor.lr_scheduler.step()
        
        return self.predictor
    
    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 5)  # Default 5 epoch
        self.local.config['parameters']['ref_data'] = self.local.config['parameters'].get("ref_data", 'forget')  # Default reference data is forget