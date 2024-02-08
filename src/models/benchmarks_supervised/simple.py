from utils_func import verPrint

import torch
import sympy
import scipy
import dgl.nn.pytorch.conv as dglnn
import dgl.function as fn

from torch import nn
from models.base_model import BaseModel

### Common Submodules ###
class MLP(nn.Module):
    def __init__(
        self, in_feats, h_feats=32, num_classes=2, num_layers=2, 
        dropout_rate=0, activation='ReLU', 
        verbose=0, **kwargs):
        """_summary_

        Args:
            in_feats (_type_): _description_
            h_feats (int, optional): _description_. Defaults to 32.
            num_classes (int, optional): _description_. Defaults to 2.
            num_layers (int, optional): _description_. Defaults to 2.
            dropout_rate (int, optional): _description_. Defaults to 0.
            activation (str, optional): _description_. Defaults to 'ReLU'.
            verbose (int, optional): _description_. Defaults to 0.
        """
        super(MLP, self).__init__()
        
        # Set verbosity
        self.verbose=verbose       
        verPrint(self.verbose, 3, f'MLP:__init__ | {in_feats} {num_classes} {h_feats} {num_layers} {dropout_rate} {activation} {kwargs}')

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
        """_summary_

        Args:
            h (_type_): _description_
            is_graph (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        verPrint(self.verbose, 3, f'MLP:forward | {h.shape} {is_graph}')
        if is_graph:
            h = h.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
            if i != len(self.layers)-1:
                h = self.act(h)
        return h


### GCN ###
class GCN(BaseModel):
    def __init__(
        self, in_feats, num_classes, h_feats, num_layers,
        mlp_h_feats, mlp_num_layers, 
        dropout_rate=0, act_name='ReLU', 
        train_mode='normal', verbose=0, **kwargs):
        """_summary_

        Args:
            in_feats (_type_): _description_
            num_classes (_type_): _description_
            h_feats (_type_): _description_
            num_layers (_type_): _description_
            mlp_h_feats (_type_): _description_
            mlp_num_layers (_type_): _description_
            dropout_rate (int, optional): _description_. Defaults to 0.
            act_name (str, optional): _description_. Defaults to 'ReLU'.
            train_mode (str, optional): _description_. Defaults to 'normal'.
            verbose (int, optional): _description_. Defaults to 0.
        """
        # Set verbosity
        self.verbose=verbose       
        verPrint(self.verbose, 3, f'GCN:__init__ | {in_feats} {num_classes} {h_feats} {num_layers} {mlp_h_feats} {mlp_num_layers} {dropout_rate} {act_name} {train_mode}')
        
        super().__init__()        
        # Other modules
        self.act = getattr(nn, act_name)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.train_mode = train_mode
        
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
        """_summary_

        Args:
            blocks (_type_): _description_
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        verPrint(self.verbose, 3, f'GCN:forward | {blocks} {x}')
        h = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks if self.train_mode != 'batch' else blocks[i], h)        
        h = self.mlp(h, False)

        return h, None # No loss returned
    
## GCN V2
## GraphSAGE
## GIN
## GAT