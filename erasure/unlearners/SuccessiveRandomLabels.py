from erasure.unlearners.torchunlearner import TorchUnlearner

from erasure.core.factory_base import get_instance_kvargs

import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class SuccessiveRandomLabels(TorchUnlearner):
    def init(self):
        """
        Initializes the SuccessiveRandomLabels class with global and local contexts.
        """

        super().init()

        self.epochs = self.local.config['parameters']['epochs']
        self.ref_data_retain = self.local.config['parameters']['ref_data_retain']  
        self.ref_data_forget = self.local.config['parameters']['ref_data_forget'] 

        self.predictor.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.predictor.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})

        retain_set = self.dataset.get_dataset_from_partition(self.ref_data_retain)
        forget_set = self.dataset.get_dataset_from_partition(self.ref_data_forget)

        unlearning_data = UnLearningData(forget_set=forget_set, retain_set=retain_set, n_classes=self.dataset.n_classes)
        self.unlearning_loader = DataLoader(unlearning_data, batch_size = self.dataset.batch_size, shuffle=True)

    def __unlearn__(self):
        """
        Fine-tunes the model with both the retain set and forget set. The labels for the forget set are randomly assigned and different from the original ones.
        """

        self.info(f'Starting SRL with {self.epochs} epochs')

        for epoch in range(self.epochs):
            losses = []
            self.predictor.model.train()

            for X, labels in self.unlearning_loader:
                X, labels = X.to(self.device), labels.to(self.device)
                
                self.predictor.optimizer.zero_grad() 

                _, output = self.predictor.model(X.to(self.device))
                
                loss = self.predictor.loss_fn(output, labels.to(self.device))

                losses.append(loss.to('cpu').detach().numpy())

                loss.backward()
                self.predictor.optimizer.step()
            
            epoch_loss = sum(losses) / len(losses)
            self.info(f'SRL - epoch = {epoch} ---> var_loss = {epoch_loss:.4f}')

            self.predictor.lr_scheduler.step()
        
        return self.predictor

    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 5)  # Default 5 epoch
        self.local.config['parameters']['ref_data_retain'] = self.local.config['parameters'].get("ref_data_retain", 'retain')  # Default reference data is retain
        self.local.config['parameters']['ref_data_forget'] = self.local.config['parameters'].get("ref_data_forget", 'forget')  # Default reference data is forget
        self.local.config['parameters']['optimizer'] = self.local.config['parameters'].get("optimizer", {'class':'torch.optim.Adam', 'parameters':{}})  # Default optimizer is Adam

class UnLearningData(Dataset):
    def __init__(self, forget_set, retain_set, n_classes):
        super().__init__()
        self.forget_set = forget_set
        self.retain_set = retain_set
        self.forget_len = len(forget_set)
        self.retain_len = len(retain_set)
        self.n_classes = n_classes

    def __len__(self):
        return self.retain_len + self.forget_len
    
    def __getitem__(self, index):
        if(index < self.forget_len):
            x = self.forget_set[index][0]
            original_label = self.forget_set[index][1]
            y = np.random.randint(0, self.n_classes)
            while y == original_label:
                y = np.random.randint(0, self.n_classes)
            y = torch.tensor(y)
            return x,y
        else:
            x = self.retain_set[index - self.forget_len][0]
            y = torch.tensor(self.retain_set[index - self.forget_len][1])
            return x,y