import torch
import torch.nn.functional as F
import dgl.function as fn
import sympy
import scipy
import dgl.nn.pytorch.conv as dglnn
import dgl
from torch import nn
from scipy.special import comb
import math
import copy
import numpy as np

### Common Submodules ###
class MLP(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, 
                 num_layers=2, dropout_rate=0, activation='ReLU', **kwargs):
        print('MLP:__init__ | ', in_feats, h_feats, num_classes, num_layers, dropout_rate, activation)

        super(MLP, self).__init__()

        # Linears
        self.layers = nn.ModuleList()       
        if num_layers == 0:
            return
        if num_layers == 1:
            self.layers.append(nn.Linear(in_feats, num_classes))
        else:
            self.layers.append(nn.Linear(in_feats, h_feats))
            for i in range(1, num_layers-1):
                self.layers.append(nn.Linear(h_feats, h_feats))
            self.layers.append(nn.Linear(h_feats, num_classes))

        # Other modules
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, h, is_graph=True):
        if is_graph:
            h = h.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
            if i != len(self.layers)-1:
                h = self.act(h)

        print('MLP output', h.shape)
        return h

### SIMPLE NN BENCHMARKS ###
## GCN
class GCN(nn.Module):
    def __init__(self, in_feats, num_classes, model_config, **kwargs):
        print("GCN:__init__ | ", in_feats, num_classes, model_config)
        
        super().__init__()
        h_feats = model_config['h_feats']
        num_layers = model_config['num_layers']
        mlp_h_feats = model_config['mlp_h_feats'] 
        mlp_num_layers = model_config['mlp_num_layers']
        dropout_rate = model_config['dropout_rate']
        act_name = model_config['act_name']
        
        # Other modules
        self.act = getattr(nn, act_name)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.train_mode = model_config['train_mode']
        
        # Layers
        self.layers = nn.ModuleList()
        if num_layers == 0:
            return
        else:
            self.layers.append(dglnn.GraphConv(in_feats, h_feats, activation=self.act))
            for i in range(num_layers-1):
                self.layers.append(dglnn.GraphConv(h_feats, h_feats, activation=self.act))
        self.mlp = MLP(h_feats, h_feats=mlp_h_feats, num_classes=num_classes, num_layers=mlp_num_layers, dropout_rate=dropout_rate)  

    def forward(self, blocks, x):
        print("GCN:forward | ", blocks, x)
        h = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks if self.train_mode != 'batch' else blocks[i], h)        
        h = self.mlp(h, False)

        print('GCN Output', h.shape)
        return h

## GCN V2
## GraphSAGE
## GIN
## GAT

### SPECTRAL
## BWGNN -> https://github.com/squareRoot3/GADBench
## SPLITGNN -> https://github.com/Split-GNN/SplitGNN/tree/master/src

## GHRN -> https://github.com/squareRoot3/GADBench
class GHRN(nn.Module):
    def __init__():
        super().__init__()
    
    def forward(self, graph):
        return 0

### HOMO/HETEROPHILY-BASED FRAUD BENCHMARKS ###
## GPRGNN -> https://github.com/jianhao2016/GPRGNN/tree/master
## GHRN -> https://github.com/squareRoot3/GADBench

## H2-FDetector -> https://github.com/squareRoot3/GADBench
class H2FD(nn.Module):
    def __init__():
        super().__init__()
    
    def forward(self, graph):
        return 0    

## SEC-GFD - NO LINK

### CAMOUFLAGE-BASED FRAUD BENCHMARKS ###    
## CARE-GNN - https://github.com/squareRoot3/GADBench
class CAREGNN(nn.Module):
    def __init__():
        super().__init__()
    
    def forward(self, graph):
        return 0        

## COFRAUD - NO LINK
## ACD - NO LINK
    
### NEIGHBORHOOD TREE-BASED BENCHMARKS ###
## RF-GRAPH
## XGB-GRAPH