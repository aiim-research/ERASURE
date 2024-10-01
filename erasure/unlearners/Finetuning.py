from erasure.core.unlearner import Unlearner
from erasure.utils.config.global_ctx import Global
from erasure.utils.config.local_ctx import Local
from fractions import Fraction


class Finetuning(Unlearner):
    def __init__(self, global_ctx: Global, local_ctx):
        super().__init__(global_ctx, local_ctx)
        self.finetune_epochs = local_ctx.config.get("finetune_epochs", 5)  # Default 5 epochs
        # self.freeze_layers = local_ctx.config.get("freeze_layers", True)   # Default to freezing layers
        # self.n_layers = local_ctx.config.get("n_layers_unfreezed", 2)      # Default to freezing everything except the last 2 layers

    def unlearn(self):
        """
        Fine-tunes the model to forget specific data points in the forget_set.
        """
        print("Starting fine-tuning...")

        # TODO: The following code will be changes as train_set = self.dataset.partitions['retain set']
        forget_set = self.dataset.partitions['forget set']
        cfg_dataset = self.dataset.local_config 
        cfg_model = self.model.local_config
        data_manager = self.global_ctx.factory.get_object( Local( cfg_dataset))
        data_manager.revise_split('train', forget_set)

        train_loader, val_loader = self.dataset.get_loader_for('train', Fraction('0'))

        print(len(train_loader))
        print(type(self.model.model))

        print("Starting fine-tuning...")
        
        for epoch in range(self.finetune_epochs):
            losses, preds, labels_list = [], [], []
            self.model.model.train()

            num_images = 0

            for batch, (X, labels) in enumerate(train_loader):

                X, labels = X.to(self.device), labels.to(self.device)
                self.model.optimizer.zero_grad()


                _,pred = self.model.model(X)

                loss = self.model.loss_fn(pred, labels)
                
                losses.append(loss.to('cpu').detach().numpy())
                loss.backward()
                
                labels_list += list(labels.squeeze().long().detach().to('cpu').numpy())
                preds += list(pred.squeeze().detach().to('cpu').numpy())
                self.model.optimizer.step()
                num_images += X.size(0)
            
            epoch_loss = sum(losses) / len(losses)
            self.global_ctx.logger.info(f'Finetuning - epoch = {epoch} ---> var_loss = {epoch_loss:.4f}')
            print(num_images)

            self.model.lr_scheduler.step()
        
        return self.model




        