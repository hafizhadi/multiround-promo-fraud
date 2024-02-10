import torch
import numpy as np

from numpy import random
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from torch.functional import F

### METHODS ###

## General
# Print but only if currently above specified verbosity level
def verPrint(verbose_status, verbose_threshold, msg):
    if verbose_status >= verbose_threshold:
        print(msg)

## Training related
def hinge_loss(labels, scores):
    margin = 1
    ls = labels * scores
    
    loss = F.relu(margin-ls)
    print("After ReLU", loss)

    loss = loss.mean()
    return loss

def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def eval_and_print(verbose_level, labels, preds, probs, msg):
    rec = recall_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    auc = roc_auc_score(labels, probs.detach().numpy()) if torch.unique(labels).shape[0] > 1 else -1

    verPrint(verbose_level, 1, f'{msg}: REC {rec*100:.2f} PRE {prec*100:.2f} MF1 {f1*100:.2f} AUC {auc*100:.2f}')

    return (rec, prec, f1, auc)
    
## Graph related

def random_duplicate(graph, n_instances=1, label=None, return_seed=False):
    pool_ids = torch.LongTensor(range(graph.num_nodes())) if label == None else (graph.ndata['label'] == label).nonzero().flatten()
    old_ids = pool_ids[random.choice(list(range(len(pool_ids))), size=n_instances, replace=False)]
    new_ids = (torch.tensor(list(range(len(old_ids)))) + graph.num_nodes()).int()
    id_dict = dict(zip(old_ids.tolist(), new_ids.tolist()))

    new_node_features = { 
        key: graph.ndata[key][old_ids] for key, _v in graph.node_attr_schemes().items() if key != '_ID'
    }
    
    new_edge_features = {}
    for etype in graph.etypes:
        in_src, in_dst, in_ids = graph.in_edges(old_ids, form='all', etype=etype)
        out_src, out_dst, out_ids = graph.out_edges(old_ids, form='all', etype=etype)

        in_dst = torch.from_numpy(np.fromiter((id_dict[i] for i in in_dst.tolist()), int))
        out_src = torch.from_numpy(np.fromiter((id_dict[i] for i in out_src.tolist()), int))

        edge_dst = torch.cat((in_dst, out_dst), 0)
        edge_src = torch.cat((in_src, out_src), 0)
        edge_ids = torch.cat((in_ids, out_ids), 0)

        new_edge_features[etype] = {
            key: graph.edges[etype].data[key][edge_ids] for key, _v in graph.edge_attr_schemes(etype).items() if key != '_ID'
        }

        new_edge_features[etype]['src'] = edge_src
        new_edge_features[etype]['dst'] = edge_dst
    
    if return_seed:
        return new_node_features, new_edge_features, old_ids
    else:
        return new_node_features, new_edge_features