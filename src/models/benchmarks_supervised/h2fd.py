from utils_func import verPrint, hinge_loss

import torch
import dgl
import copy
import numpy as np
import torch.nn.functional as F

from torch import nn
from models.base_model import BaseModel

### Model ###
class H2FDRelationAware(nn.Module):
    def __init__(
        self, in_feats, h_feats, 
        dropout_rate=0, verbose=0, **kwargs):
        """_summary_

        Args:
            in_feats (_type_): _description_
            h_feats (_type_): _description_
            dropout_rate (int, optional): _description_. Defaults to 0.
            verbose (int, optional): _description_. Defaults to 0.
        """
        # Set verbosity
        self.verbose=verbose       
        verPrint(self.verbose, 3, f'H2FDRelationAware:__init__ | {in_feats} {h_feats} {dropout_rate} {kwargs}')
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self.d_liner = nn.Linear(in_feats, h_feats)
        self.f_liner = nn.Linear(3 * h_feats, 1)
        self.tanh = nn.Tanh()

    def forward(self, src, dst):
        verPrint(self.verbose, 3, f'H2FDRelationAware:forward | {src} {dst}')
        src = self.d_liner(src)
        dst = self.d_liner(dst)
        diff = src-dst
        
        e_feats = torch.cat([src, dst, diff], dim=1)
        e_feats = self.dropout(e_feats)
        
        score = self.f_liner(e_feats).squeeze()
        score = self.tanh(score)
        return score
    
## H2-FD - Detector Module
class H2FDLayer(nn.Module):
    def __init__(
        self, in_feats, h_feats, att_heads, 
        etype, relation_aware, if_sum=False,
        verbose=0, **kwargs):
        """_summary_

        Args:
            in_feats (_type_): _description_
            h_feats (_type_): _description_
            att_heads (_type_): _description_
            etype (_type_): _description_
            relation_aware (_type_): _description_
            if_sum (bool, optional): _description_. Defaults to False.
            verbose (int, optional): _description_. Defaults to 0.
        """
        super().__init__()
        self.etype = etype
        self.att_heads = att_heads
        self.h_feats = h_feats
        self.if_sum = if_sum

        # Modules
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.relation_aware = relation_aware
        
        self.w_liner = nn.Linear(in_feats, h_feats * att_heads)
        self.atten = nn.Linear(2 * self.h_feats, 1)


    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['feature'] = h
            graph.apply_edges(self.sign_edges, etype=self.etype)
            h = self.w_liner(h)
            graph.ndata['h'] = h
            graph.update_all(message_func=self.message, reduce_func=self.reduce, etype=self.etype)
            out = graph.ndata['out']
            return out

    def message(self, edges):
        src = edges.src
        src_features = edges.data['sign'].view(-1,1) * src['h']
        src_features = src_features.view(-1, self.att_heads, self.h_feats)
        
        z = torch.cat([src_features, edges.dst['h'].view(-1, self.att_heads, self.h_feats)], dim=-1)
        
        alpha = self.atten(z)
        alpha = self.leakyrelu(alpha)
        
        return {'atten':alpha, 'sf':src_features}

    def reduce(self, nodes):
        alpha = self.softmax(nodes.mailbox['atten'])
        sf = nodes.mailbox['sf']

        out = torch.sum(alpha * sf, dim=1)
        out = out.sum(dim=-2) if self.if_sum else out.view(-1, self.att_heads * self.h_feats)

        return {'out':out}

    def sign_edges(self, edges):
        score = self.relation_aware(edges.src['feature'], edges.dst['feature'])
        return {'sign':torch.sign(score)}

## H2-FD - Multirelation Detector Module
class H2FDMultiRelationLayer(nn.Module):
    def __init__(
        self, in_feats, h_feats, att_heads, 
        relations, dropout_rate=0, if_sum=False, 
        verbose=0, **kwargs):
        """_summary_

        Args:
            in_feats (_type_): _description_
            h_feats (_type_): _description_
            att_heads (_type_): _description_
            relations (_type_): _description_
            dropout_rate (int, optional): _description_. Defaults to 0.
            if_sum (bool, optional): _description_. Defaults to False.
            verbose (int, optional): _description_. Defaults to 0.
        """
        super().__init__()

        # Set verbosity
        self.verbose=verbose       
        verPrint(self.verbose, 3, f'H2FDMultiRelationLayer:__init__ | {in_feats} {h_feats} {att_heads} {relations} {att_heads} {dropout_rate} {kwargs}')

        self.relations = copy.deepcopy(relations)
        
        self.relation_aware = H2FDRelationAware(in_feats, h_feats * att_heads, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self.liner = nn.Linear(len(self.relations) * h_feats, h_feats) if if_sum else nn.Linear(len(self.relations) * h_feats * att_heads, h_feats * att_heads)
        self.minelayers = nn.ModuleDict()
        for e in self.relations:
            self.minelayers[e] = H2FDLayer(in_feats, h_feats, att_heads, e, self.relation_aware, if_sum)
    
    def forward(self, graph, h):
        h = torch.cat([self.minelayers[e](graph, h) for e in self.relations], dim=1)
        h = self.dropout(h)
        h = self.liner(h)
        
        return h
    
    def loss(self, graph, h):

        with graph.local_scope():
            graph.ndata['feature'] = h

            # Edge clasification loss
            graph.apply_edges(self.score_edges, etype='homo')
            
            edge_train_mask = graph.edges['homo'].data['train_mask'].bool()

            edge_train_label = graph.edges['homo'].data['label'][edge_train_mask]
            edge_train_score = graph.edges['homo'].data['score'][edge_train_mask]            
                 
            edge_train_pos_index = (edge_train_label == 1).nonzero().flatten().detach().cpu().numpy()
            edge_train_neg_index = (edge_train_label == -1).nonzero().flatten().detach().cpu().numpy()
            edge_train_pos_index = np.random.choice(edge_train_pos_index, size=len(edge_train_neg_index))            
            index = np.concatenate([edge_train_pos_index, edge_train_neg_index])

            edge_diff_loss = hinge_loss(edge_train_label[index], edge_train_score[index])

            # Prototype Loss
            agg_h = self.forward(graph, h)
            train_mask = graph.ndata['train_mask'].bool()
            
            train_h = agg_h[train_mask]
            train_label = graph.ndata['label'][train_mask]

            train_pos_index = (train_label == 1).nonzero().flatten().detach().cpu().numpy()
            train_neg_index = (train_label == 0).nonzero().flatten().detach().cpu().numpy()

            pos_prototype = torch.mean(train_h[train_pos_index], dim=0).view(1,-1)
            neg_prototype = torch.mean(train_h[train_neg_index], dim=0).view(1,-1)            

            train_neg_index = np.random.choice(train_neg_index, size=len(train_pos_index)) # Undersample
            node_index = np.concatenate([train_neg_index, train_pos_index])
            
            train_h_loss = train_h[node_index]            
            pos_prototypes = pos_prototype.expand(train_h_loss.shape)
            neg_prototypes = neg_prototype.expand(train_h_loss.shape)
            
            diff_pos = - F.pairwise_distance(train_h_loss, pos_prototypes).view(-1,1)
            diff_neg = - F.pairwise_distance(train_h_loss, neg_prototypes).view(-1,1)

            diff = torch.cat([diff_neg, diff_pos], dim=1)
            diff_loss = F.cross_entropy(diff, train_label[node_index])

            return agg_h, edge_diff_loss, diff_loss
        
    def score_edges(self, edges):
        score = self.relation_aware(edges.src['feature'], edges.dst['feature'])
        verPrint(self.verbose, 2, f'Score: {score}')
        return {'score': score }

## H2-FD - Main Model
class H2FD(BaseModel):
    def __init__(
        self, in_feats, num_classes, etypes=['none'], n_layer=1, intra_dim=16,
        gamma1=1.2, gamma2=2, att_heads=2, dropout_rate=0.1, 
        verbose=1, **kwargs):
        """_summary_

        Args:
            in_feats (_type_): _description_
            num_classes (int, optional): _description_.
            etypes (_type_): _description_
            n_layer (int, optional): _description_. Defaults to 1.
            intra_dim (int, optional): _description_. Defaults to 16.
            gamma1 (float, optional): _description_. Defaults to 1.2.
            gamma2 (int, optional): _description_. Defaults to 2.
            att_heads (int, optional): _description_. Defaults to 2.
            dropout_rate (float, optional): _description_. Defaults to 0.1.
        """
        super().__init__()

        # Set verbosity
        self.verbose=verbose       
        verPrint(self.verbose, 3, f'H2FD:__init__ | {in_feats} {num_classes} {etypes} {n_layer} {intra_dim} {gamma1} {gamma2} {att_heads} {dropout_rate} {kwargs}')

        
        self.in_feats = in_feats
        self.num_classes = num_classes
        self.n_layer = n_layer 
        self.intra_dim = intra_dim 
        self.head = att_heads

        self.gamma1 = gamma1
        self.gamma2 = gamma2

        # Misc modules
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.relu = nn.ReLU()

        # Layers
        self.mine_layers = nn.ModuleList()
        if n_layer == 1:
            self.mine_layers.append(H2FDMultiRelationLayer(self.in_feats, self.num_classes, att_heads, etypes, dropout_rate, if_sum=True, verbose=self.verbose))
        else:
            self.mine_layers.append(H2FDMultiRelationLayer(self.in_feats, self.intra_dim, att_heads, etypes, dropout_rate, verbose=self.verbose))
            for _ in range(1, self.n_layer-1):
                self.mine_layers.append(H2FDMultiRelationLayer(self.intra_dim * att_heads, self.intra_dim, att_heads, etypes, dropout_rate, verbose=self.verbose))
            self.mine_layers.append(H2FDMultiRelationLayer(self.intra_dim * att_heads, self.num_classes, att_heads, etypes, dropout_rate, if_sum=True, verbose=self.verbose))

    
    def forward(self, graph, x):
        train_mask = graph.ndata['train_mask'].bool()
        train_label = graph.ndata['label'][train_mask]

        
        pos_index = (train_label == 1).nonzero().flatten().detach().cpu().numpy()
        neg_index = (train_label == 0).nonzero().flatten().detach().cpu().numpy()
        neg_index = np.random.choice(neg_index, size=len(pos_index), replace=False)
        index = np.concatenate([pos_index, neg_index])

        h, edge_loss, prototype_loss = self.mine_layers[0].loss(graph, x)
        for i in range(1, len(self.mine_layers)):
            h = self.relu(h)
            h = self.dropout(h)

            h, e_loss, p_loss = self.mine_layers[i].loss(graph, h)
            edge_loss += e_loss
            prototype_loss += p_loss
        
        model_loss = F.cross_entropy(h[train_mask][index], train_label[index])

        verPrint(self.verbose, 2, f'Model loss: {model_loss}, Edge loss: {edge_loss}, Prototype loss: {prototype_loss}')
        loss = model_loss + (self.gamma1 * edge_loss) + (self.gamma2 * prototype_loss)
        return h, loss
    
    @staticmethod
    def generate_edges_labels(edges, labels, train_idx):
        """
        A function that generates the class homophily label for edges in a graph with the rule:
            - Plus 1 for edges connecting nodes with same label
            - Minus 1 for edges connecting nodes with opposite label

        Args:
            edges ((torch.Tensor, torch.Tensor)): A pair of tuple containing the source and destination edge of the graph
            labels (torch.Tensor): Labels indicating anomaly class (0, 1) for each of the graph nodes
            train_idx (torch.Tensor): _description_

        Returns:
            (torch.Tensor, torch.Tensor): A pair of vector with identical dimension; the former indicates the label for each edge and the latter indicate train mask for the respective
        """
        print('train_idx', train_idx)
        srcs, dsts = edges[0].cpu(), edges[1].cpu()
        edge_labels = ((labels[srcs] == labels[dsts]).int() - 0.5).sign().long() # 1 if same label, -1 otherwise
        src_train_mask = torch.zeros_like(srcs).bool()
        dst_train_mask = torch.zeros_like(srcs).bool()

        for idx in train_idx: # Iterate due to memory load if not
            src_train_mask = src_train_mask | ((srcs == idx)) # True if both in train_idx, false otherwise
            dst_train_mask = dst_train_mask | ((dsts == idx)) # True if both in train_idx, false otherwise

        return edge_labels, (src_train_mask & dst_train_mask)

    @staticmethod
    def prepare_graph(graph):
        """
        A function that preprocess a multirelation graph to be used with H2F-Detector.

        Args:
            graph (DGLGraph): A DGL Graph with the following criteria
                - Can have multiple edge type but only one node type
                - Features stored in ndata needs to be Tensor
                - Have two features -> 'label' and 'train_mask'

        Returns:
            DGLGraph:The graph with an additional edges of relation type 'homo' which is just all edges flattened, containing label for hetero/homophily
                - Value is 1 for edges connecting similarly labeled
        """
        # Convert to heterogeneous if not yet
        if graph.is_homogeneous:
            graph.ndata['_TYPE'] = torch.zeros_like(torch.arange(graph.num_nodes()))
            graph.edata['_TYPE'] = torch.zeros_like(torch.arange(graph.num_edges()))
            graph = dgl.to_heterogeneous(graph, ['none'], ['none'])

        # Create new graph with a homogeneous etype containing all edge
        new_data = {c: graph[c].edges() for c in graph.canonical_etypes if c[1] != 'homo'}
        new_data[(graph.canonical_etypes[0][0], 'homo', graph.canonical_etypes[0][0])] = dgl.to_homogeneous(graph).edges()
        new_g = dgl.heterograph(new_data)

        # Copy ndata
        for feat in graph.ndata:
            new_g.ndata[feat] = graph.ndata[feat].clone()
        
        # Produce homophilic/heterophilic edge labels       
        homo_edges = new_g.edges(etype='homo')
        homo_labels, homo_train_mask = H2FD.generate_edges_labels(homo_edges, new_g.ndata['label'].cpu(), new_g.ndata['train_mask'].nonzero().squeeze(1).tolist())

        # Append to edata
        new_g.edges['homo'].data['label'] = homo_labels
        new_g.edges['homo'].data['train_mask'] = homo_train_mask
        
        for ntype in graph.ntypes:
            for key in graph.ndata.keys():
                new_g.nodes[ntype].data[key] = graph.nodes[ntype].data[key].clone()

        return new_g