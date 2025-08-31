import torch
import numpy as np

from numpy import random
from utils.utils_func import verPrint

##########################
# BASE ADVERSARIAL CLASS #
##########################
class BaseAdversary():
    def __init__(self, verbose=0):
        pass

    def generate(self, graph, n_instances=1, return_ids=False, **kwargs):
        return None
    
    @staticmethod
    def random_duplicate(graph, n_instances=1, label=None, return_ids=False, prio_pool=torch.tensor([], dtype=torch.long), verbose=0):
        verPrint(verbose, 3, f'START - BaseAdversary:random_duplicate | n_instances: {n_instances},  label: {label}, return_ids: {return_ids}, prio_pool: {prio_pool.tolist()}')

        # Split how many comes from prio pool how many comes from rest        
        prio_instances = min([prio_pool.shape[0], n_instances])
        rest_instances = max([0, n_instances - prio_pool.shape[0]])
        verPrint(verbose, 0, f'Seed distribution: {prio_instances} PRIO - {rest_instances} REST')

        # Create pool of IDs to randomly choose from
        pool_ids = list(range(graph.num_nodes())) if label == None else (graph.ndata['label'] == label).nonzero().flatten().tolist()
        pool_ids = torch.tensor(list(set(pool_ids) - set(prio_pool.tolist())), dtype=torch.long)

        # Choose seeds from pool and map new + old Node ID
        old_ids = torch.cat([prio_pool[random.choice(list(range(len(prio_pool))), size=prio_instances, replace=False)], pool_ids[random.choice(list(range(len(pool_ids))), size=rest_instances, replace=False)]])
        new_ids = (torch.tensor(list(range(len(old_ids)))) + graph.num_nodes()).int() # New Node IDs sequentially generated from original biggest ID
        id_dict = dict(zip(old_ids.tolist(), new_ids.tolist()))

        # Copy node features
        new_node_features = { key: graph.ndata[key][old_ids] for key, _v in graph.node_attr_schemes().items() if key != '_ID' }
        new_edge_features = {}

        # Copy edge features
        for etype in graph.etypes:
            in_src, in_dst, in_ids = graph.in_edges(old_ids, form='all', etype=etype)
            out_src, out_dst, out_ids = graph.out_edges(old_ids, form='all', etype=etype)

            in_dst = torch.from_numpy(np.fromiter((id_dict[i] for i in in_dst.tolist()), int))
            out_src = torch.from_numpy(np.fromiter((id_dict[i] for i in out_src.tolist()), int))

            new_edge_features[etype] = {}
            new_edge_features[etype]['in'] = { key: graph.edges[etype].data[key][in_ids] for key, _v in graph.edge_attr_schemes(etype).items() if key != '_ID' }
            new_edge_features[etype]['in']['src'] = in_src
            new_edge_features[etype]['in']['dst'] = in_dst

            new_edge_features[etype]['out'] = { key: graph.edges[etype].data[key][out_ids] for key, _v in graph.edge_attr_schemes(etype).items() if key != '_ID' }
            new_edge_features[etype]['out']['src'] = out_src
            new_edge_features[etype]['out']['dst'] = out_dst
        
        # Return
        verPrint(verbose, 3, f'FINISH - BaseAdversary:random_duplicate | new_node_feats: {new_node_features},  new_edge_feats: {new_edge_features}, old_ids: {old_ids}, new_ids: {new_ids}')
        if return_ids:
            return new_node_features, new_edge_features, old_ids, new_ids
        else:
            return new_node_features, new_edge_features, None, None