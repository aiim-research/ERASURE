import torch
import torch.nn as nn
from torchvision.models import resnet18

# class Cifar20ResNet18(nn.Module):
#     def __init__(self, n_classes=20):
#         super(Cifar20ResNet18, self).__init__()
        
#         resnet = resnet18(pretrained=False)
#         # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         # resnet.maxpool = nn.Identity()
        
#         self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  
        
        
#         self.fc1 = nn.Linear(resnet.fc.in_features, 512)  
#         self.fc2 = nn.Linear(512, n_classes)  
        
#         self.relu = nn.ReLU()
#         self.flatten = nn.Flatten()  
#         self.last_layer = self.fc2

#     def forward(self, x):
#         x = self.feature_extractor(x)
#         x = self.flatten(x)  
        
#         x = self.relu(self.fc1(x))
#         intermediate_output = x  
        
#         x = self.fc2(x)
        
#         return intermediate_output, x
    
import torch
import torch.nn as nn

class Cifar20ResNet18(nn.Module):
    def __init__(self, n_classes=20):
        super(Cifar20ResNet18, self).__init__()
        
        # First Conv Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Conv Block
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the flattened size
        # Input: 32x32x3 -> Conv1: 30x30x64 -> Pool1: 15x15x64 
        # -> Conv2: 13x13x64 -> Pool2: 6x6x64 = 2304
        self.flat_features = 6 * 6 * 64
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flat_features, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, n_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # First Conv Block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Second Conv Block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(-1, self.flat_features)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        intermediate_output = x
        x = self.softmax(x)
        
        return intermediate_output, x