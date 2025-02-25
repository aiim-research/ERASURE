import torch
import torch.nn as nn
import torchvision.models as models

class MNISTResNet18(nn.Module):
    def __init__(self, n_classes=10):
        super(MNISTResNet18, self).__init__()
        
        resnet = models.resnet18(pretrained=False)
        
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        resnet.maxpool = nn.Identity()
        
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]) 
        
        self.fc1 = nn.Linear(resnet.fc.in_features, 512)  
        self.fc2 = nn.Linear(512, n_classes)  
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()  

    def forward(self, x):
        x = self.feature_extractor(x)  
        x = self.flatten(x)  
        
        x = self.relu(self.fc1(x))  
        intermediate_output = x  
        
        x = self.fc2(x) 
        
        return intermediate_output, x


class FashionMNISTResNet18(nn.Module):
    def __init__(self, n_classes=10):
        super(FashionMNISTResNet18, self).__init__()
        
        resnet = models.resnet18(pretrained=False)
        
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        resnet.maxpool = nn.Identity()
        
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]) 
        
        self.fc1 = nn.Linear(resnet.fc.in_features, 512)  
        self.fc2 = nn.Linear(512, n_classes)  
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()  

    def forward(self, x):
        x = self.feature_extractor(x) 
        x = self.flatten(x)  
        
        x = self.relu(self.fc1(x))  
        intermediate_output = x  
        
        x = self.fc2(x) 
        
        return intermediate_output, x
