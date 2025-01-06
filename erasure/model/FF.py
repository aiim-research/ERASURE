import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    # Non funziona.
    # Si rompe la fase di fitting. Da lavorarci se serve
    def __init__(self, n_classes):
        super().__init__()
        self.linear = nn.Linear(n_classes, 1)

    def forward(self, x):
        x = self.linear(x)
        intermediate_output = x
        return intermediate_output, x


class MIAAttack(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(n_classes, 3*n_classes)
        self.fc2 = nn.Linear(3*n_classes, int(1.5*n_classes))
        self.fc3 = nn.Linear(int(1.5*n_classes), 2)
        self.relu = nn.ReLU()
        self.last_layer = self.fc3

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        intermediate_output = x
        x = self.fc3(x)
        return intermediate_output, x

