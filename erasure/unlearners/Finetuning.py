from erasure.unlearners.torchunlearner import TorchUnlearner
from fractions import Fraction

from erasure.core.factory_base import get_instance_kvargs

import time


class Finetuning(TorchUnlearner):
    def init(self):
        """
        Initializes the Finetuning class with global and local contexts.
        """

        super().init()

        self.epochs = self.local.config['parameters']['epochs'] 
        self.ref_data = self.local.config['parameters']['ref_data'] 

        self.predictor.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.predictor.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})


    def __unlearn__(self):
        """
        Fine-tunes the model with a specific (sub)set of the full dataset (usually retain set)
        """

        self.info(f'Starting Finetuning with {self.epochs} epochs')

        start = time.time()
        retain_loader, _ = self.dataset.get_loader_for(self.ref_data, Fraction('0'))
        
        starting_time = time.time() - start

        for epoch in range(self.epochs):
            losses = []
            self.predictor.model.train()

            total_batches = len(retain_loader)

            for batch, (X, labels) in enumerate(retain_loader):
                start = time.time()
                X, labels = X.to(self.device), labels.to(self.device)
                self.predictor.optimizer.zero_grad() 

                _, pred = self.predictor.model(X)

                loss = self.predictor.loss_fn(pred, labels)
                losses.append(loss.to('cpu').detach().numpy())

                loss.backward()

                self.predictor.optimizer.step()
                step_time = time.time() - start
                break

            total_time = starting_time + step_time * total_batches
            
            with open('times.txt', 'a') as f:
                f.write(f"Finetuning: {total_time}\n")
            
            epoch_loss = sum(losses) / len(losses)
            self.info(f'Finetuning - epoch = {epoch} ---> var_loss = {epoch_loss:.4f}')

            self.predictor.lr_scheduler.step()
        
        return self.predictor
    
    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 5)  # Default 5 epoch
        self.local.config['parameters']['ref_data'] = self.local.config['parameters'].get("ref_data", 'retain')  # Default reference data is retain
        self.local.config['parameters']['optimizer'] = self.local.config['parameters'].get("optimizer", {'class':'torch.optim.Adam', 'parameters':{}})  # Default optimizer is Adam