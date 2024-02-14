from utils_func import verPrint, hinge_loss

import torch
import dgl
import copy
import numpy as np
import torch.nn.functional as F
import dgl.function as fn

from torch import nn
from models.base_model import BaseModel

### CAMOUFLAGE-BASED FRAUD BENCHMARKS ###    
## CARE-GNN - https://github.com/squareRoot3/GADBench
class CAREConv(nn.Module):
    def __init__(
            self, in_feats, num_classes, h_feats, 
            activation=None, step_size=0.02, **kwargs):
        super().__init__()
        self.activation = activation
        self.step_size = step_size
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.dist = {}
        self.linear = nn.Linear(self.in_feats, self.h_feats)
        self.MLP = nn.Linear(self.in_feats, self.num_classes)
        self.p = {}
        self.last_avg_dist = {}
        self.f = {}
        self.cvg = {}

    def _calc_distance(self, edges):
        # formula 2
        print("src", edges.src['h'].shape)
        print("dst", edges.dst['h'].shape)

        d = torch.norm(torch.tanh(self.MLP(edges.src["h"]))
            - torch.tanh(self.MLP(edges.dst["h"])), 1, 1,)
        return {"d": d}

    def _top_p_sampling(self, graph, p):
        # Compute the number of neighbors to keep for each node
        in_degrees = graph.in_degrees()
        num_neigh = torch.ceil(in_degrees.float() * p).int()

        # Fetch all edges and their distances
        all_edges = graph.edges(form="eid")
        dist = graph.edata["d"]

        # Create a prefix sum array for in-degrees to use for indexing
        prefix_sum = torch.cat([torch.tensor([0]), in_degrees.cumsum(0)[:-1]])

        # Get the edges for each node using advanced indexing
        selected_edges = []
        for i, node_deg in enumerate(num_neigh):
            start_idx = prefix_sum[i]
            end_idx = start_idx + node_deg
            sorted_indices = torch.argsort(dist[start_idx:end_idx])[:node_deg]
            selected_edges.append(all_edges[start_idx:end_idx][sorted_indices])
        return torch.cat(selected_edges)

    def forward(self, block, x, epoch=0):
        feat = x
        edges = block.canonical_etypes
        if epoch == 0:
            for etype in edges:
                self.p[etype] = 0.5
                self.last_avg_dist[etype] = 0
                self.f[etype] = []
                self.cvg[etype] = False

        with block.local_scope():
            block.ndata["h"] = feat

            hr = {}
            for i, etype in enumerate(edges):
                block.apply_edges(self._calc_distance, etype=etype)
                self.dist[etype] = block.edges[etype].data["d"]
                sampled_edges = self._top_p_sampling(block[etype], self.p[etype])

                # formula 8
                block.send_and_recv(
                    sampled_edges,
                    fn.copy_u("h", "m"),
                    fn.mean("m", "h_%s" % etype[1]),
                    etype=etype,
                )
                hr[etype] = block.ndata["h_%s" % etype[1]]
                if self.activation is not None:
                    hr[etype] = self.activation(hr[etype])

            # formula 9 using mean as inter-relation aggregator
            p_tensor = (
                torch.Tensor(list(self.p.values())).view(-1, 1, 1).to(block.device)
            )
            h_homo = torch.sum(torch.stack(list(hr.values())) * p_tensor, dim=0)
            h_homo += feat
            if self.activation is not None:
                h_homo = self.activation(h_homo)

            return self.linear(h_homo), None


class CAREGNN(BaseModel):
    def __init__(self, in_feats, num_classes=2, h_feats=64, edges=None, num_layers=1, activation=None, step_size=0.02, **kwargs):
        super().__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.activation = None if activation is None else getattr(nn, activation)()
        self.step_size = step_size
        self.num_layers = num_layers
        self.output_linear = nn.Linear(h_feats, num_classes)
        self.layers = nn.ModuleList()
        self.layers.append(          # Input layer
            CAREConv(self.in_feats, self.num_classes, self.num_classes, activation=self.activation, step_size=self.step_size,))
        for i in range(self.num_layers - 1):  # Hidden layers with n - 2 layers
            self.layers.append(CAREConv(self.h_feats, self.h_feats, self.num_classes, activation=self.activation, step_size=self.step_size,))
            # self.layers.append(   # Output layer
                # CAREConv(self.h_feats, self.num_classes, self.num_classes, activation=self.activation, step_size=self.step_size,))

    def forward(self, blocks, x, epoch=0, **kwargs):            
        for layer in self.layers:
            feat = layer(blocks, x, epoch)
        return feat
    
    def postBackprop(self, graph, epoch, rl_idx, **kwargs):
        self.RLModule(graph, epoch, rl_idx)

    def RLModule(self, graph, rl_idx, epoch):
        for layer in self.layers:
            for etype in graph.canonical_etypes:
                if not layer.cvg[etype]:
                    # formula 5
                    eid = graph.in_edges(rl_idx, form='eid', etype=etype)
                    avg_dist = torch.mean(layer.dist[etype][eid])

                    # formula 6
                    if layer.last_avg_dist[etype] < avg_dist:
                        if layer.p[etype] - self.step_size > 0:
                            layer.p[etype] -=   self.step_size
                        layer.f[etype].append(-1)
                    else:
                        if layer.p[etype] + self.step_size <= 1:
                            layer.p[etype] += self.step_size
                        layer.f[etype].append(+1)
                    layer.last_avg_dist[etype] = avg_dist

                    # formula 7
                    if epoch >= 9 and abs(sum(layer.f[etype][-10:])) <= 2:
                        layer.cvg[etype] = True

## COFRAUD - NO LINK
## ACD - NO LINK
    
## GPRGNN -> https://github.com/jianhao2016/GPRGNN/tree/master
## GHRN -> https://github.com/squareRoot3/GADBench
## SEC-GFD - NO LINK