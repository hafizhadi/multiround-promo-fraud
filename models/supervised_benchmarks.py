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

### CALL WRAPPER FUNCTION

def Benchmark(in_feats, bname='GCN',
              h_feats=32, num_classes=2, **kwargs):
    if bname == 'GCN':
        return GCN(in_feats, h_feats=h_feats, num_classes=num_classes)
    else:
        return # TODO: throw exception?

### SIMPLE NN BENCHMARKS ###

class MLP(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, 
                 num_layers=2, dropout_rate=0, activation='ReLU', **kwargs):

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
        return h

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, 
                 num_layers=2, mlp_layers=1, 
                 dropout_rate=0, activation='ReLU', 
                 **kwargs):
        
        super().__init__()
        
        # Layers
        self.h_feats = h_feats
        self.layers = nn.ModuleList()
        if num_layers == 0:
            return
        else:
            self.layers.append(dglnn.GraphConv(in_feats, h_feats, activation=self.act))
            for i in range(num_layers-1):
                self.layers.append(dglnn.GraphConv(h_feats, h_feats, activation=self.act))
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout_rate)
        
        # Other modules
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        h = self.mlp(h, False)
        return h
    
### GENERIC FRAUD BENCHMARKS ###
    
### HOMO/HETEROPHILY-BASED FRAUD BENCHMARKS ###
    
### CAMOUFLAGE-BASED FRAUD BENCHMARKS ###
    
### NEIGHBORHOOD TREE-BASED BENCHMARKS ###