import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.linear = nn.Linear(n_classes, 2)

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        intermediate_output = x
        return intermediate_output, x


class MIAAttack(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(n_classes, 3*n_classes)
        self.fc2 = nn.Linear(3*n_classes, int(1.5*n_classes))
        self.fc3 = nn.Linear(int(1.5*n_classes), int(0.5*n_classes))
        self.fc4 = nn.Linear(int(0.5*n_classes), 2)
        self.relu = nn.ReLU()
        self.last_layer = self.fc4

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        intermediate_output = x
        x = self.fc4(x)
        return intermediate_output, x


class FlexSeqBin(nn.Module):

    def __init__(self, n_classes, mult=[2,1,0.5]):
        super().__init__()
        self.layers = nn.Sequential()
        lIn = int(n_classes)
        nOut = 2

        for i in range(len(mult)):
            lOut = int(mult[i]*lIn) ; i < len(mult) ; nOut
            self.layers.append(nn.Linear(lIn, lOut))
            self.layers.append(nn.ReLU())
            lIn = lOut

        self.last = nn.Linear(lIn, nOut)

    def forward(self, x):
        intermediate_output = self.layers(x)
        return intermediate_output, self.last(intermediate_output)
