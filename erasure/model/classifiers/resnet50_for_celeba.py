import torch
import torch.nn as nn
from torchvision.models import resnet50  # Import ResNet50

class CelebAResNet50(nn.Module):
    def __init__(self, n_classes=2):
        super(CelebAResNet50, self).__init__()
        
        resnet = resnet50(pretrained=True)  
        
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        self.fc1 = nn.Linear(resnet.fc.in_features, 512) 
        self.fc2 = nn.Linear(512, n_classes)             
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.last_layer = self.fc2

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        
        x = self.relu(self.fc1(x))
        intermediate_output = x
        
        x = self.fc2(x)
        
        return intermediate_output, x
