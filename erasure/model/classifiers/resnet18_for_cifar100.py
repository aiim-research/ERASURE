import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, vgg16

class Cifar100ResNet18(nn.Module):
    def __init__(self, n_classes=100):
        super(Cifar100ResNet18, self).__init__()
        
        resnet = resnet18()
        
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


class Cifar100ResNet50(nn.Module):
    def __init__(self, n_classes=100):
        super(Cifar100ResNet50, self).__init__()
        
        resnet = resnet50()
        
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
    
class Cifar100VGG16(nn.Module):
    def __init__(self, n_classes=100):
        super().__init__()
        vgg = vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(
            vgg.features,
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, n_classes)
        self.relu = nn.ReLU()
        self.last_layer = self.fc2

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        intermediate_output = x
        x = self.fc2(x)
        return intermediate_output, x
