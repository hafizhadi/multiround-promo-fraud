from utils.utils_func import verPrint

import torch
import sympy
import scipy
import dgl.nn.pytorch.conv as dglnn
import dgl.function as fn
import dgl.utils as ut

from torch import nn
from torch.nn import init
import torch.nn.functional as F
from models.base_model import BaseModel
from utils.kmeans import KMeans

import time

EPS = 1e-10

class SplitMessagePass(nn.Module):
    def __init__(
            self,
            in_feats, out_feats, 
            act_name='ReLU', init_eps=0,
            verbose=0, **kwargs):
        super().__init__()
        self.verbose=verbose       
        verPrint(self.verbose, 3, f'SplitMessagePass:__init__ | {kwargs}')

        self.act = getattr(nn, act_name)()

        # Learnable adjustment param
        self.eps_pos = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        self.eps_neg = torch.nn.Parameter(torch.FloatTensor([init_eps]))

        # Weight
        self.weight_self = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.weight_pos = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.weight_neg = nn.Parameter(torch.Tensor(in_feats, out_feats))

        init.xavier_uniform_(self.weight_self)
        init.xavier_uniform_(self.weight_pos)
        init.xavier_uniform_(self.weight_neg)

    
    @staticmethod
    def _split(edges):
        start = time.time()

        res = torch.cat([
            edges.src['h'] * (edges.src['lbl'] - 1 * -1).unsqueeze(1), 
            edges.src['h'] * edges.src['lbl'].unsqueeze(1)], 
        dim=1)
        
        #print("Message passing time", time.time() - start)
        return { 
            "m": res,
        }

    def forward(self, blocks, x, **kwargs):
        with blocks.local_scope():
            feat_src, feat_dst = ut.expand_as_pair(x, blocks)

            blocks.srcdata['h'] = feat_src
            blocks.srcdata['lbl'] = blocks.ndata['label']
            blocks.update_all(SplitMessagePass._split, fn.sum('m', 'h_sum'))
            
            start = time.time()
            h_self = feat_dst 
            h_sum_pos = blocks.dstdata['h_sum'][:,:feat_dst.shape[1]]
            h_sum_neg = blocks.dstdata['h_sum'][:,feat_dst.shape[1]:]

            h_self_fin = h_self + (1 + self.eps_pos) * h_sum_pos + (1 + self.eps_neg) * h_sum_neg
            h_self_mult = torch.matmul(h_self_fin, self.weight_self)
            h_sum_pos_mult = torch.matmul(h_sum_pos, self.weight_pos)
            h_sum_neg_mult = torch.matmul(h_sum_neg, self.weight_neg)

            rst = torch.cat([h_self_mult, h_sum_pos_mult, h_sum_neg_mult], dim=1)
            #print("Postprocessing time", time.time() - start)

            if self.act is not None:
                rst = self.act(rst)
            
            return rst


class ProtoLayer(nn.Module):
    def __init__(
        self, num_proto=1,
        verbose=0, **kwargs):

        super().__init__()
        self.verbose=verbose       
        verPrint(self.verbose, 3, f'ProtoLayer:__init__ | {num_proto} {kwargs}')

        self.num_proto = num_proto

    def forward(self, blocks, x, **kwargs):
        # Get prototypes
        neg_idx = (blocks.ndata['label'] == 0).nonzero().flatten().detach().cpu().numpy()
        pos_idx = (blocks.ndata['label'] == 1).nonzero().flatten().detach().cpu().numpy()

        neg_feats = x[neg_idx].clone()
        pos_feats = x[pos_idx].clone()

        start = time.time()
        if self.num_proto > 1:
            neg_centers = KMeans(n_clusters=self.num_proto)(neg_feats.unsqueeze(0)).centers[0].unsqueeze(0)
            pos_centers = KMeans(n_clusters=self.num_proto)(pos_feats.unsqueeze(0)).centers[0].unsqueeze(0)
        else:
            neg_centers = neg_feats.mean(dim=0).unsqueeze(0)
            pos_centers = pos_feats.mean(dim=0).unsqueeze(0)
        #print("K-means time", time.time() - start)

        # Calculate Loss

        start = time.time()        
        diff_neg = (F.pairwise_distance(x.unsqueeze(1).expand(-1, neg_centers.shape[1], -1), neg_centers.expand(x.shape[0], -1, -1)).min(dim=1)[0] + EPS).pow(-1)
        diff_pos = (F.pairwise_distance(x.unsqueeze(1).expand(-1, pos_centers.shape[1], -1), pos_centers.expand(x.shape[0], -1, -1)).min(dim=1)[0] + EPS).pow(-1)
        diffs = torch.stack([diff_neg, diff_pos]).swapaxes(0, 1)
        probs = diffs / diffs.sum(dim=1).unsqueeze(1)
        #print("Distance calc time", time.time() - start)
        return probs

class ProtoFraud(BaseModel):
    def __init__(
        self, in_feats, num_classes, h_feats, num_layers,
        num_proto=1, dropout_rate=0, act_name='ReLU', 
        train_mode='normal', verbose=0, **kwargs):

        super().__init__()
        self.verbose=verbose       
        verPrint(self.verbose, 3, f'ProtoFraud:__init__ | {in_feats} {num_classes} {h_feats} {num_layers} {num_proto} {dropout_rate} {act_name} {train_mode}')
        
        # Other modules
        self.act = getattr(nn, act_name)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.train_mode = train_mode
        
        # Layers
        self.h_agg_layers = nn.ModuleList()
        self.h_agg_layers.append(dglnn.GraphConv(in_feats, h_feats, activation=self.act))
        for i in range(num_layers-1):
            self.h_agg_layers.append(dglnn.GraphConv(h_feats, h_feats, activation=self.act))
        
        self.proto_layer = ProtoLayer(num_proto=num_proto)

    def forward(self, blocks, x, **kwargs):
        verPrint(self.verbose, 3, f'ProtoFraud:forward | {blocks} {x}')
        h = x

        for i, layer in enumerate(self.h_agg_layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks if self.train_mode != 'batch' else blocks[i], h)        

        h = self.proto_layer(blocks, h)

        return h, None

class SplitProtoFraud(BaseModel):
    def __init__(
        self, in_feats, num_classes, h_feats, num_layers,
        num_proto=1, dropout_rate=0, act_name='ReLU', 
        train_mode='normal', verbose=0, **kwargs):

        super().__init__()
        self.verbose=verbose       
        verPrint(self.verbose, 3, f'ProtoFraud:__init__ | {in_feats} {num_classes} {h_feats} {num_layers} {num_proto} {dropout_rate} {act_name} {train_mode}')
        
        # Other modules
        self.act = getattr(nn, act_name)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.train_mode = train_mode
        
        # Layers
        self.h_agg_layers = nn.ModuleList()
        self.h_agg_layers.append(SplitMessagePass(in_feats, h_feats, activation=self.act))
        for i in range(num_layers-1):
            self.h_agg_layers.append(SplitMessagePass(h_feats * 3, h_feats, activation=self.act))
        
        self.proto_layer = ProtoLayer(num_proto=num_proto)

    def forward(self, blocks, x, **kwargs):
        verPrint(self.verbose, 3, f'ProtoFraud:forward | {blocks} {x}')
        h = x

        for i, layer in enumerate(self.h_agg_layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks if self.train_mode != 'batch' else blocks[i], h)        

        h = self.proto_layer(blocks, h)

        return h, None
    
    