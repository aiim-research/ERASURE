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