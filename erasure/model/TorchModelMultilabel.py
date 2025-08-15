import numpy as np
import random
import torch

from erasure.core.trainable_base import Trainable
from erasure.utils.cfg_utils import init_dflts_to_of
from erasure.core.factory_base import get_instance_kvargs, get_instance
from sklearn.metrics import accuracy_score  # kept, but not used in multi-label accuracy
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Subset
import torch.nn as nn
from fractions import Fraction


class TorchModelMultilabel(Trainable):

    def init(self):
        self.epochs = self.local_config['parameters']['epochs']

        self.model = get_instance_kvargs(
            self.local_config['parameters']['model']['class'],
            self.local_config['parameters']['model']['parameters']
        )

        self.model.apply(init_weights)

        self.optimizer = get_instance_kvargs(
            self.local_config['parameters']['optimizer']['class'],
            {'params': self.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']}
        )

        self.loss_fn = get_instance_kvargs(
            self.local_config['parameters']['loss_fn']['class'],
            self.local_config['parameters']['loss_fn']['parameters']
        )

        self.early_stopping_threshold = self.local_config['parameters']['early_stopping_threshold']

        self.lr_scheduler = lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=self.epochs
        )

        self.training_set = self.local_config['parameters'].get('training_set', 'train')

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)
        self.model.device = self.device

        self.patience = 0
        self.fit()

    def real_fit(self):

        train_loader, val_loader = self.dataset.get_loader_for(self.training_set, Fraction('1/10'))

        best_loss = [0, 0]

        for epoch in range(self.epochs):
            losses, preds, labels_list = [], [], []
            self.model.train()

            for batch, (X, labels) in enumerate(train_loader):

                self.optimizer.zero_grad()

                X = X.to(self.device)

                labels = labels.to(self.device).float()

                _, pred = self.model(X)            

                loss = self.loss_fn(pred, labels)   
                losses.append(loss.detach().to('cpu').numpy())

                loss.backward()
                self.optimizer.step()

                labels_list.append(labels.detach().to('cpu').numpy())  
                preds.append(pred.detach().to('cpu').numpy())          

            labels_mat = np.concatenate(labels_list, axis=0)
            preds_mat = np.concatenate(preds, axis=0)

            accuracy = self.accuracy(labels_mat, preds_mat, threshold=0.5, mode="per_attr")
            self.global_ctx.logger.info(f'epoch = {epoch} ---> loss = {np.mean(losses):.4f}\t accuracy = {accuracy:.4f}')
            self.lr_scheduler.step()

            if self.early_stopping_threshold and len(val_loader) > 0:
                self.model.eval()
                var_losses, var_labels, var_preds = [], [], []
                with torch.no_grad():
                    for batch, (X, labels) in enumerate(val_loader):
                        self.optimizer.zero_grad()
                        X = X.to(self.device)
                        labels = labels.to(self.device).float()

                        _, pred = self.model(X)

                        loss = self.loss_fn(pred, labels)
                        var_losses.append(loss.item())

                        var_labels.append(labels.detach().to('cpu').numpy())
                        var_preds.append(pred.detach().to('cpu').numpy())

                best_loss.pop(0)
                var_loss = np.mean(var_losses) if len(var_losses) > 0 else np.nan
                best_loss.append(var_loss)

                if len(var_labels) > 0:
                    var_labels_mat = np.concatenate(var_labels, axis=0)
                    var_preds_mat = np.concatenate(var_preds, axis=0)
                    val_acc = self.accuracy(var_labels_mat, var_preds_mat, threshold=0.5, mode="per_attr")
                else:
                    val_acc = float('nan')

                self.global_ctx.logger.info(f'epoch = {epoch} ---> var_loss = {var_loss:.4f}\t var_accuracy = {val_acc:.4f}')

                if abs(best_loss[0] - best_loss[1]) < self.early_stopping_threshold:
                    self.patience += 1
                    if self.patience == 4:
                        self.global_ctx.logger.info(f"Early stopped training at epoch {epoch}")
                        break

    def check_configuration(self):
        super().check_configuration()
        local_config = self.local_config

        # defaults
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 50)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 4)
        local_config['parameters']['early_stopping_threshold'] = local_config['parameters'].get('early_stopping_threshold', None)

        init_dflts_to_of(local_config, 'optimizer', 'torch.optim.Adam', lr=0.001)

        init_dflts_to_of(local_config, 'loss_fn', 'torch.nn.BCEWithLogitsLoss')

        local_config['parameters']['model']['parameters']['n_classes'] = \
            local_config['parameters']['model']['parameters'].get('n_classes', self.dataset.n_classes)

        local_config['parameters']['alias'] = local_config['parameters']['model']['class']
        local_config['parameters']['training_set'] = local_config['parameters'].get("training_set", "train")

    def accuracy(self, y_true, y_pred_logits, threshold=0.5, mode="per_attr"):
        """
        Multi-label accuracy:
          - Apply sigmoid to logits
          - Threshold to 0/1
          - mode="per_attr": mean over all entries (samples x attributes)
          - mode="subset": exact-match across all attributes per sample, then mean
        """
        y_true = np.asarray(y_true)
        y_pred_logits = np.asarray(y_pred_logits)

        y_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))
        y_pred = (y_prob >= threshold).astype(np.int32)

        if mode == "subset":
            return (y_pred == y_true).all(axis=1).mean()
        else:
            return (y_pred == y_true).mean()


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
