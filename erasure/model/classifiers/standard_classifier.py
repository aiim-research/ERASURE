import torch
import torch.nn as nn

class IrisNN(nn.Module):
    def __init__(self, n_classes):
        super(IrisNN, self).__init__()
        
        self.fc1 = nn.Linear(4, 50)  
        self.fc2 = nn.Linear(50, 30)          
        self.fc3 = nn.Linear(30, n_classes) 
        
        self.relu = nn.ReLU() 
        self.flatten = nn.Flatten()
        self.last_layer = self.fc3

    def forward(self, x):
        x = self.relu(self.fc1(x))  
        x = self.relu(self.fc2(x))  
        intermediate_output = x   
        x = self.fc3(x)    
        x = x.squeeze(1)        
        return intermediate_output, x