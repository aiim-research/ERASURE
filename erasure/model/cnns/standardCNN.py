import os
import torch.nn as nn
#from torch_geometric.nn.aggr import MeanAggregation,SoftmaxAggregation


class FMNIST_CNN(nn.Module):
   
    def __init__(self,
                 n_classes):
        
        super(FMNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(125, 200)   
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, n_classes)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.last_layer = self.fc3

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        intermediate_output = x
        x = self.fc3(x)
        return intermediate_output,x


class FashionCNN(nn.Module):
    
    def __init__(self,n_classes):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=n_classes)

        self.last_layer = self.fc3
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        #out = self.drop(out)
        out = self.fc2(out)
        intermediate_output = out
        out = self.fc3(out)       
        
        return intermediate_output,out