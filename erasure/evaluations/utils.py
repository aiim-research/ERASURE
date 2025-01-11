from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics import accuracy_score


def compute_accuracy(test_loader, model):
    var_labels, var_preds = [], [],
    with torch.no_grad():
        for batch, (X, labels) in enumerate(test_loader):
            _, pred = model(X.to(model.device))

            var_labels += list(labels.squeeze().to('cpu').numpy())
            var_preds += list(pred.squeeze().to('cpu').numpy())

        accuracy = accuracy_score(var_labels, np.argmax(var_preds, axis=1))

    return accuracy


def compute_relearn_time(model, data_loader, max_accuracy=0.8, max_epochs=100):

    # model = deepcopy(model)

    epochs = 0

    curr_accuracy = compute_accuracy(data_loader, model.model)

    while (curr_accuracy < max_accuracy  # reached the target accuracy
    and epochs < max_epochs):  # fine-tune for a maximum of epochs
        losses, preds, labels_list = [], [], []
        model.model.train()
        for batch, (X, labels) in enumerate(data_loader):
            X, labels = X.to(model.device), labels.to(model.device)
            model.optimizer.zero_grad()
            _, pred = model.model(X)
            loss = model.loss_fn(pred, labels)
            loss.backward()
            model.optimizer.step()

            losses.append(loss.to('cpu').detach().numpy())
            labels_list += list(labels.squeeze().long().detach().to('cpu').numpy())
            preds += list(pred.squeeze().detach().to('cpu').numpy())

        curr_accuracy = model.accuracy(labels_list, preds)
        model.lr_scheduler.step()

        epochs += 1

    return epochs