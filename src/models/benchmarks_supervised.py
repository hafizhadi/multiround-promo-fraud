from utils_func import verPrint

import torch
import sympy
import scipy
import dgl.nn.pytorch.conv as dglnn
import dgl.function as fn

from torch import nn

### Common Submodules ###
class MLP(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, 
                 num_layers=2, dropout_rate=0, activation='ReLU', 
                 verbose=0, **kwargs):
        super(MLP, self).__init__()
        
        # Set verbosity
        self.verbose=verbose       
        verPrint(self.verbose, 3, f'MLP:__init__ | {in_feats} {h_feats} {num_classes} {num_layers} {dropout_rate} {activation} {kwargs}')

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
    
class PolyConv(nn.Module):
    def __init__(self, theta):
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

### SIMPLE NN BENCHMARKS ###
## GCN
class GCN(nn.Module):
    def __init__(self, in_feats, num_classes, model_config, verbose=0, **kwargs):
        # Set verbosity
        self.verbose=verbose       
        verPrint(self.verbose, 3, f'GCN:__init__ | {in_feats} {num_classes} {model_config}')
        
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
        verPrint(self.verbose, 3, f'GCN:forward | {blocks} {x}')
        h = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks if self.train_mode != 'batch' else blocks[i], h)        
        h = self.mlp(h, False)

        return h

## GCN V2
## GraphSAGE
## GIN
## GAT

### SPECTRAL
## BWGNN
class BWGNN(nn.Module):
    def __init__(self, in_feats, num_classes, model_config, verbose=0, **kwargs):
        # Set verbosity
        self.verbose=verbose       
        verPrint(self.verbose, 3, f'BWGNN:__init__ | {in_feats} {num_classes} {model_config}')
        
        super().__init__()
        h_feats = model_config['h_feats']
        num_layers = model_config['num_layers']
        mlp_num_layers = model_config['mlp_num_layers']
        dropout_rate = model_config['dropout_rate']
        act_name = model_config['act_name']

        self.thetas = self.calculate_theta(d=num_layers)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(self.thetas[i]))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.mlp = MLP(h_feats*len(self.conv), h_feats, num_classes, mlp_num_layers, dropout_rate)
        self.act = getattr(nn, act_name)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

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
        return h
    
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