from erasure.core.unlearner import Unlearner
from erasure.unlearners.torchunlearner import TorchUnlearner
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from fractions import Fraction


class Finetuning(TorchUnlearner):
    def __init__(self, global_ctx: Global, local_ctx):
        """
        Initializes the Finetuning class with global and local contexts.

        Args:
            global_ctx (Global): The global context containing configurations and shared resources.
            local_ctx (Local): The local context containing specific configurations for this instance.
        """
        super().__init__(global_ctx, local_ctx)
        self.epochs = local_ctx.config.get("epochs", 5)  # Default 5 epochs

    def __unlearn__(self):
        """
        Fine-tunes the model to forget specific data points in the forget_set.
        """

        self.global_ctx.logger.info(f'Starting Finetuning with {self.epochs} epochs')

        # TODO: The following code will be changed as train_set = self.dataset.partitions['retain set']
        forget_set = self.dataset.partitions['forget set']
        cfg_dataset = self.dataset.local_config 
        data_manager = self.global_ctx.factory.get_object(Local(cfg_dataset))
        data_manager.revise_split('train', forget_set)

        retain_loader, _ = self.dataset.get_loader_for('train', Fraction('0'))
        
        for epoch in range(self.epochs):
            losses = []
            self.predictor.model.train()

            for batch, (X, labels) in enumerate(retain_loader):
                X, labels = X.to(self.device), labels.to(self.device)
                self.predictor.optimizer.zero_grad() 

                _, pred = self.predictor.model(X)

                loss = self.predictor.loss_fn(pred, labels)
                losses.append(loss.to('cpu').detach().numpy())

                loss.backward()

                self.predictor.optimizer.step()
            
            epoch_loss = sum(losses) / len(losses)
            self.global_ctx.logger.info(f'Finetuning - epoch = {epoch} ---> var_loss = {epoch_loss:.4f}')

            self.predictor.lr_scheduler.step()
        
        return self.predictor