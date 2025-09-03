import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

class CelebAResNet18(nn.Module):
    def __init__(self, n_classes=2):
        super(CelebAResNet18, self).__init__()

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
    

class CelebAResNet18_logits(nn.Module):
    def __init__(self, n_attrs=None, n_classes=None):

        super(CelebAResNet18_logits, self).__init__()
        out_dim = n_attrs if n_attrs is not None else (n_classes if n_classes is not None else 40)

        resnet = resnet18()

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.fc1 = nn.Linear(resnet.fc.in_features, 512)
        self.fc2 = nn.Linear(512, out_dim)

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
    
class CelebAResNet50(nn.Module):
    def __init__(self, n_classes=2):
        super(CelebAResNet50, self).__init__()
        
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
    
class CelebAResNet50_logits(nn.Module):
    def __init__(self, n_attrs=None, n_classes=None):

        super(CelebAResNet50_logits, self).__init__()
        out_dim = n_attrs if n_attrs is not None else (n_classes if n_classes is not None else 40)

        resnet = resnet50()

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.fc1 = nn.Linear(resnet.fc.in_features, 512)
        self.fc2 = nn.Linear(512, out_dim)

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