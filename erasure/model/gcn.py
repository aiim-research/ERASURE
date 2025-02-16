import torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
import torch
from erasure.core.factory_base import build_w_params_string

class GCN(nn.Module):
   
    def __init__(self, batch_size, node_features, num_conv_layers=2, conv_booster=1, pooling=MeanAggregation):
        super(GCN, self).__init__()
        
        self.in_channels = node_features
        self.out_channels = int(self.in_channels * conv_booster)
        self.batch_size = batch_size
        self.pooling =  build_w_params_string(pooling)
 
        
        if num_conv_layers>1:
            self.num_conv_layers = [(self.in_channels, self.out_channels)] + [(self.out_channels, self.out_channels)] * (num_conv_layers - 1)
        else:
            self.num_conv_layers = [(self.in_channels, self.out_channels)]
        self.graph_convs = self.__init__conv_layers()
        
    def forward(self, X):
        node_features = X.x.double()

        edge_index = X.edge_index.long()
        edge_weight = X.edge_attr[:, 0].double() if X.edge_attr is not None else None  # Use only one weight per edge

        for conv_layer in self.graph_convs[:-1]:
            node_features = conv_layer(node_features, edge_index, edge_weight)
            node_features = nn.functional.relu(node_features)

        intermediate_results = node_features

        # Handle last layer correctly
        if isinstance(self.graph_convs[-1], MeanAggregation):
            return intermediate_results, self.graph_convs[-1](node_features, X.batch)
        else:
            return intermediate_results, self.graph_convs[-1](node_features)

    
    def __init__conv_layers(self):
        ############################################
        # initialize the convolutional layers interleaved with pooling layers
        graph_convs = []
        for i in range(len(self.num_conv_layers)):#add len
            graph_convs.append(GCNConv(in_channels=self.num_conv_layers[i][0],
                                      out_channels=self.num_conv_layers[i][1]).double())
        graph_convs.append(self.pooling)
        return nn.Sequential(*graph_convs).double()


class DownstreamGCN(GCN):
   
    def __init__(self, batch_size, node_features,
                 n_classes=2,
                 num_conv_layers=2,
                 num_dense_layers=2,
                 conv_booster=2,
                 linear_decay=2,
                 pooling=MeanAggregation()):
        
        super().__init__(batch_size, node_features, num_conv_layers, conv_booster, pooling)
        
        self.num_dense_layers = num_dense_layers
        self.linear_decay = linear_decay
        self.n_classes = n_classes
        
        self.downstream_layers = self.__init__downstream_layers()
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, X):
        intermediate_results, node_features = super().forward(X)
        return intermediate_results, self.downstream_layers(node_features)
    
    def __init__downstream_layers(self):
        ############################################
        # initialize the linear layers interleaved with activation functions
        downstream_layers = []
        in_linear = self.out_channels
        for _ in range(self.num_dense_layers-1):
            downstream_layers.append(nn.Linear(in_linear, int(in_linear // self.linear_decay)))
            downstream_layers.append(nn.ReLU())
            in_linear = int(in_linear // self.linear_decay)
        # add the output layer
        downstream_layers.append(nn.Linear(in_linear, self.n_classes))
        #downstream_layers.append(nn.Sigmoid())
        #downstream_layers.append(nn.Softmax())
        # put the linear layers in sequential
        return nn.Sequential(*downstream_layers).double()