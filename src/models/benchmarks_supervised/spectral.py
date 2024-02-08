from utils_func import verPrint

import torch
import sympy
import scipy
import dgl.nn.pytorch.conv as dglnn
import dgl.function as fn

from torch import nn
from models.benchmarks_supervised.simple import MLP
from models.base_model import BaseModel

### BWGNN ###
## Polyconv Module
class PolyConv(nn.Module):
    def __init__(self, theta):
        """_summary_

        Args:
            theta (_type_): _description_
        """
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        return h
    
## Main Model
class BWGNN(BaseModel):
    def __init__(
        self, in_feats, num_classes, h_feats, num_layers, mlp_num_layers,
        dropout_rate=0, act_name='ReLU', verbose=0, **kwargs):
        """_summary_

        Args:
            in_feats (_type_): _description_
            num_classes (_type_): _description_
            h_feats (_type_): _description_
            num_layers (_type_): _description_
            mlp_num_layers (_type_): _description_
            dropout_rate (int, optional): _description_. Defaults to 0.
            act_name (str, optional): _description_. Defaults to 'ReLU'.
            verbose (int, optional): _description_. Defaults to 0.
        """

        # Set verbosity
        self.verbose=verbose       
        verPrint(self.verbose, 3, f'BWGNN:__init__ | {in_feats} {num_classes} {h_feats} {num_layers} {mlp_num_layers} {dropout_rate} {act_name} {kwargs}')
        super().__init__()

        # Misc modules
        self.act = getattr(nn, act_name)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # BW Filters
        self.thetas = self.calculate_theta(d=num_layers)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(self.thetas[i]))
        
        # Linear and MLP
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.mlp = MLP(h_feats*len(self.conv), h_feats, num_classes, mlp_num_layers, dropout_rate)


    def forward(self, blocks, x):
        verPrint(self.verbose, 3, f'BWGNN:forward | {blocks} {x}')

        # TODO: Batch
        graph = blocks

        in_feat = x
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0], device=h.device)

        for conv in self.conv:
            h0 = conv(graph, h)
            h_final = torch.cat([h_final, h0], -1)
        h_final = self.dropout(h_final)
        h = self.mlp(h_final, False)
        
        return h, None # No loss returned
    
    def calculate_theta(self, d):
        thetas = []
        x = sympy.symbols('x')
        for i in range(d+1):
            f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
            coeff = f.all_coeffs()
            inv_coeff = []
            for i in range(d+1):
                inv_coeff.append(float(coeff[d-i]))
            thetas.append(inv_coeff)
        return thetas

## GHRN -> https://github.com/squareRoot3/GADBench
class GHRN(BaseModel):
    def __init__():
        super().__init__(self)
    
    def forward(self, graph):
        return 0

## SPLITGNN -> https://github.com/Split-GNN/SplitGNN/tree/master/src