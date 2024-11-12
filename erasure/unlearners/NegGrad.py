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
        self.epochs = local_ctx.config['parameters'].get("epochs", 5)  # Default 5 epoch
        self.ref_data = local_ctx.config['parameters'].get("ref_data", 'forget set')  # Default reference data is forget


    def __unlearn__(self):
        """
        Negative Gradient (NegGrad) unlearning algorithm for selective forgetting as proposed by https://arxiv.org/abs/1911.04933. This method fine-tunes the model to forget specific data points by reversing the gradient direction on the forget data, encouraging the model to minimize the influence of these samples.
        
        For each sample in the forget data (forget_set), the stochastic gradient ascent is applied, effectively pushing the model to unlearn the patterns associated with these specific samples.

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