import numpy as np
import random
import torch

from erasure.core.trainable_base import Trainable
from erasure.utils.cfg_utils import init_dflts_to_of
from erasure.core.factory_base import get_instance_kvargs, get_instance
from sklearn.metrics import accuracy_score
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Subset
#from torch_geometric.loader import DataLoader
import torch.nn as nn

from fractions import Fraction


class TorchModel(Trainable):
       
    def init(self):
        self.epochs = self.local_config['parameters']['epochs']

        self.model = get_instance_kvargs(self.local_config['parameters']['model']['class'],
                                   self.local_config['parameters']['model']['parameters'])
        
        self.model.apply(init_weights)
        
        self.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})
        
        self.loss_fn = get_instance_kvargs(self.local_config['parameters']['loss_fn']['class'],
                                           self.local_config['parameters']['loss_fn']['parameters'])
        
        self.early_stopping_threshold = self.local_config['parameters']['early_stopping_threshold']
        
        self.lr_scheduler =  lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=self.epochs)

        self.training_set = self.local_config['parameters'].get('training_set','train')



        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
                if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device) 
        self.model.device = self.device
        
        self.patience = 0     
        self.fit() 

    
    def real_fit(self):

        train_loader, val_loader = self.dataset.get_loader_for(self.training_set, Fraction('1/10'))
        
        best_loss = [0,0]
        
        for epoch in range(self.epochs):
            losses, preds, labels_list = [], [], []
            self.model.train()

            for batch, (X, labels) in enumerate(train_loader):
                print("x is ", X)
                print("y is" , labels)

                self.optimizer.zero_grad()

                X, labels = X.to(self.device), labels.to(self.device)
                

                _,pred = self.model(X)

                loss = self.loss_fn(pred, labels)
                
                # print(loss)
                losses.append(loss.to('cpu').detach().numpy())
                loss.backward()
                
                labels_list += list(labels.squeeze().long().detach().to('cpu').numpy())


                preds += list(pred.squeeze().detach().to('cpu').numpy())
               
                self.optimizer.step()

            accuracy = self.accuracy(labels_list, preds)
            self.global_ctx.logger.info(f'epoch = {epoch} ---> loss = {np.mean(losses):.4f}\t accuracy = {accuracy:.4f}')
            self.lr_scheduler.step()
            
            # check if we need to do early stopping
            if self.early_stopping_threshold and len(val_loader) > 0:
                self.model.eval()
                var_losses, var_labels, var_preds = [], [], []
                with torch.no_grad():
                    for batch, (X, labels) in enumerate(val_loader):

                        self.optimizer.zero_grad()

                        X, labels = X.to(self.device), labels.to(self.device)

                        _,pred = self.model(X)

                        loss = self.loss_fn(pred, labels)
                    
                            
                        var_labels += list(labels.squeeze().to('cpu').numpy())
                        var_preds += list(pred.squeeze().to('cpu').numpy())
                        
                        var_losses.append(loss.item())
                        
                    best_loss.pop(0)
                    var_loss = np.mean(var_losses)
                    best_loss.append(var_loss)
                            
                    accuracy = self.accuracy(var_labels, var_preds)
                    self.global_ctx.logger.info(f'epoch = {epoch} ---> var_loss = {var_loss:.4f}\t var_accuracy = {accuracy:.4f}')
                
                if abs(best_loss[0] - best_loss[1]) < self.early_stopping_threshold:
                    self.patience += 1
                    
                    if self.patience == 4:
                        self.global_ctx.logger.info(f"Early stopped training at epoch {epoch}")
                        break  
                
    def check_configuration(self):
        super().check_configuration()
        local_config=self.local_config
        # set defaults

        ###LOCAL_CONFIG MUST REFERENCE CONTEXT LOCAL
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 50)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 4)
        local_config['parameters']['early_stopping_threshold'] = local_config['parameters'].get('early_stopping_threshold', None)
        # populate the optimizer
        init_dflts_to_of(local_config, 'optimizer', 'torch.optim.Adam',lr=0.001)
        init_dflts_to_of(local_config, 'loss_fn', 'torch.nn.BCELoss')


        local_config['parameters']['model']['parameters']['n_classes'] = local_config['parameters']['model']['parameters'].get('n_classes',self.dataset.n_classes)

        local_config['parameters']['alias'] = local_config['parameters']['model']['class']
        local_config['parameters']['training_set'] = local_config['parameters'].get("training_set", "train")

        
    def accuracy(self, testy, probs):
        acc = accuracy_score(testy, np.argmax(probs, axis=1))
        return acc
'''
    def read(self):
        super().read()
        if isinstance(self.model, list):
            for mod in self.model:
                mod.to(self.device)
        else:
            self.model.to(self.device)
            
    def to(self, device):
        if isinstance(self.model, torch.nn.Module):
            self.model.to(device)
        elif isinstance(self.model, list):
            for model in self.model:
                if isinstance(model, torch.nn.Module):
                    model.to(self.device)
                    '''

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)