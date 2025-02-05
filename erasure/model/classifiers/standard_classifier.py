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

class AdultNN(nn.Module):
    def __init__(self, n_classes, n_layers=3, inputsize=68, hidden_size=100):
        super(AdultNN, self).__init__()
        # pass n_layers, inputsize, hidden_size
        self.fc1 = nn.Linear(inputsize, hidden_size)  
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers-1)])
        self.fc3 = nn.Linear(hidden_size, n_classes) 
        
        self.relu = nn.ReLU() 

    def forward(self, x):
        x = self.relu(self.fc1(x))  
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        intermediate_output = x   
        x = self.fc3(x)    
        x = x.squeeze(1)        
        return intermediate_output, x

# class SpotifyNN(nn.Module):
#     def __init__(self, n_classes):
#         super(SpotifyNN, self).__init__()
        
#         self.fc1 = nn.Linear(15, 512)  
#         self.bn1 = nn.BatchNorm1d(512)
#         self.fc2 = nn.Linear(512, 256)     
#         self.bn2 = nn.BatchNorm1d(256)
#         self.fc3 = nn.Linear(256, 128)       
#         self.fc4 = nn.Linear(128, 64)
#         self.fc5 = nn.Linear(64, n_classes)
        
#         self.relu = nn.ReLU() 
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, x):
#         x = self.relu(self.bn1(self.fc1(x)))  
#         x = self.dropout(x)
#         x = self.relu(self.bn2(self.fc2(x)))  
#         x = self.dropout(x)
#         x = self.fc3(x)
#         x = self.fc4(x)
#         intermediate_output = x
#         x = self.fc5(x)
#         return intermediate_output, x
    
class SpotifyNN(nn.Module):
    def __init__(self, n_classes, n_layers=10, inputsize=15, hidden_size=100):
        super(SpotifyNN, self).__init__()
        print("The number of classes is: ", n_classes)
        self.fc1 = nn.Linear(inputsize, hidden_size)  
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers-1)])
        self.fc3 = nn.Linear(hidden_size, n_classes)
        
        self.relu = nn.ReLU() 

    def forward(self, x):
        x = self.relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        intermediate_output = x
        x = self.fc3(x)
        return intermediate_output, x