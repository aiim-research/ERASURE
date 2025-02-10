from erasure.unlearners.torchunlearner import TorchUnlearner
from fractions import Fraction

from erasure.core.factory_base import get_instance_kvargs

import time

class NegGrad(TorchUnlearner):
    def init(self):
        """
        Initializes the NegGrad class with global and local contexts.
        """

        super().init()

        self.epochs = self.local.config['parameters']['epochs']  
        self.ref_data = self.local.config['parameters']['ref_data'] 

        self.predictor.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.predictor.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})


    def __unlearn__(self):
        """
        An implementation of the Negative Gradient unlearning algorithm proposed in the following paper:
        "Golatkar, A., Achille, A. and Soatto, S., 2019. Eternal sunshine of the spotless net: Selective forgetting in deep networks. In 2020 IEEE. In CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 9301-9309)."
        
        Codebase taken from this implementation: https://github.com/ndb796/MachineUnlearning
        """

        self.info(f'Starting NegGrad with {self.epochs} epochs')

        start = time.time()

        forget_loader, _ = self.dataset.get_loader_for(self.ref_data, Fraction('0'))

        starting_time = time.time() - start

        for epoch in range(self.epochs):
            losses, preds, labels_list = [], [], []
            self.predictor.model.train()

            total_batches = len(forget_loader)

            for X, labels in forget_loader:
                start = time.time()
                X, labels = X.to(self.device), labels.to(self.device)
                self.predictor.optimizer.zero_grad() 

                _, pred = self.predictor.model(X)

                loss = -self.predictor.loss_fn(pred, labels)
                losses.append(loss.to('cpu').detach().numpy())

                loss.backward()

                labels_list += list(labels.squeeze().long().detach().to('cpu').numpy())
                preds += list(pred.squeeze().detach().to('cpu').numpy())

                self.predictor.optimizer.step()

                step_time = time.time() - start
                break 

            total_time = starting_time + step_time * total_batches

            with open('times.txt', 'a') as f:
                f.write(f"NegGrad: {total_time}\n")
            
            epoch_loss = sum(losses) / len(losses)
            self.info(f'NegGrad - epoch = {epoch} ---> var_loss = {epoch_loss:.4f}')

            self.predictor.lr_scheduler.step()
        
        return self.predictor
    
    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 5)  # Default 5 epoch
        self.local.config['parameters']['ref_data'] = self.local.config['parameters'].get("ref_data", 'forget')  # Default reference data is forget
        self.local.config['parameters']['optimizer'] = self.local.config['parameters'].get("optimizer", {'class':'torch.optim.Adam', 'parameters':{}})  # Default optimizer is Adam