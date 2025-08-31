import torch
import dgl.nn.pytorch.conv as dglnn
from torch import nn
import numpy as np
import xgboost as xgb

class GIN_noparam(nn.Module):
    def __init__(self, num_layers=2, agg='mean', init_eps=-1, **kwargs):
        super().__init__()
        self.gnn = dglnn.GINConv(None, activation=None, init_eps=init_eps,aggregator_type=agg)
        self.num_layers = num_layers

    def forward(self, graph):
        h = graph.ndata['feature']
        h_final = h.detach().clone()
        for i in range(self.num_layers):
            h = self.gnn(graph, h)
            h_final = torch.cat([h_final, h], -1)
        return h_final

class GraphBoost():
    def __init__(self, boost_agg_backbone, boost_predictor, boost_metric, **kwargs):
        self.eval_metric = boost_metric
        self.agg_backbone = boost_agg_backbone().to('cpu') if boost_agg_backbone != None else None # TODO: pass model config
        self.predictor = boost_predictor(eval_metric=self.eval_metric)

    def __call__(self, graph, feats, **kwargs):
        agg_feats = self.agg_backbone(graph) if self.agg_backbone != None else graph.ndata['feature']        
        return torch.tensor(self.predictor.predict_proba(agg_feats)), None

    def train(self, graph, weight):
        feats = self.agg_backbone(graph) if self.agg_backbone != None else graph.ndata['feature']
        labels = graph.ndata['label']

        train_X = feats[graph.ndata['train_mask']].cpu().numpy()
        train_y = labels[graph.ndata['train_mask']].cpu().numpy()
        val_X = feats[graph.ndata['val_mask']].cpu().numpy()
        val_y = labels[graph.ndata['val_mask']].cpu().numpy()
    
        weights = np.where(train_y == 0, 1, weight)
        self.predictor.fit(train_X, train_y, sample_weight=weights, eval_set=[(val_X, val_y)])  # early_stopping_rounds =20
        
        pred_train = self.predictor.predict_proba(train_X)[:, 1]
        pred_val = self.predictor.predict_proba(val_X)[:, 1]

        train_score = self.eval_metric(train_y.flatten(), pred_train.flatten())
        val_score = self.eval_metric(val_y.flatten(), pred_val.flatten())

        return train_score, val_score
    
    def eval(self):
        return
    
    def postBackprop(self, **kwargs):
        return
    
    @staticmethod
    def prepare_graph(graph):
        return graph