import torch
import numpy as np

from numpy import random


### BASE ADVERSARIAL CLASS ###
class BaseAdversary():
    def __init__(self):
        pass

    def generate(self, graph, n_instances=1, return_ids=False, **kwargs):
        return 0
    
    @staticmethod
    def random_duplicate(graph, n_instances=1, label=None, return_ids=False):
        """_summary_

        Args:
            graph (_type_): _description_
            n_instances (int, optional): _description_. Defaults to 1.
            label (_type_, optional): _description_. Defaults to None.
            return_ids (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
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

            new_edge_features[etype] = {}
            new_edge_features[etype]['in'] = { key: graph.edges[etype].data[key][in_ids] for key, _v in graph.edge_attr_schemes(etype).items() if key != '_ID' }
            new_edge_features[etype]['in']['src'] = in_src
            new_edge_features[etype]['in']['dst'] = in_dst

            new_edge_features[etype]['out'] = { key: graph.edges[etype].data[key][out_ids] for key, _v in graph.edge_attr_schemes(etype).items() if key != '_ID' }
            new_edge_features[etype]['out']['src'] = out_src
            new_edge_features[etype]['out']['dst'] = out_dst
        
        if return_ids:
            return new_node_features, new_edge_features, old_ids, new_ids
        else:
            return new_node_features, new_edge_features, None, None