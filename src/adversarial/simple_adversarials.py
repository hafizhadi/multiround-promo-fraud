import torch
import numpy as np

from numpy import random
from collections import Counter
from adversarial.adversarial import BaseAdversary
from utils_func import random_duplicate

class ReplayAdversary(BaseAdversary):
    def __init__(self, **kwargs):
        super().__init__()
        return
    
    def generate(self, graph, n_instances=1, return_ids=False, is_random=True, **kwargs):
        return random_duplicate(graph, n_instances=n_instances, label=1, return_ids=return_ids)
    
class PerturbationAdversary(BaseAdversary):
    def __init__(self, feat_budget, conn_budget, **kwargs):
        super().__init__()

        self.feat_budget = feat_budget
        self.conn_budget = conn_budget
    
    def generate(self, graph, n_instances=1, return_ids=False, is_random=True, **kwargs):
        replay_node, replay_edge, old_ids, new_ids =  random_duplicate(graph, n_instances=n_instances, label=1, return_ids=return_ids)

        ## FEATURE PERTURBATION ##
        feats = replay_node['feature'].clone()
        perturb_weight = torch.rand(feats.shape)
        perturb_weight = perturb_weight / perturb_weight.sum(dim=1).unsqueeze(1) # This is the distribution for the noise over the entire feature dimension for each node
        perturb_amount = torch.rand(feats.shape[0]) * self.feat_budget # This is the amount of noise for each node
        perturb_final = (perturb_weight * perturb_amount.unsqueeze(1)) * (torch.rand(feats.shape) - 0.5).sign() # Randomize noise sign
        replay_node['feature'] = feats + perturb_final

        ## STRUCTURAL PERTURBATION ##
        # TODO: WORK ON DIRECTED VERSION
        if sum([(sorted(replay_edge[etype]['in']['src']) != sorted(replay_edge[etype]['out']['dst'])) and 
                (sorted(replay_edge[etype]['in']['dst']) == sorted(replay_edge[etype]['out']['src'])) 
                for etype in b.keys()]) == 0: # Check if graph undirected (same edges for ingoing and outgoing)

            # Split budget over edge relations
            ren_count = len(graph.etypes)
            rounding_error = 1
            while rounding_error != 0:
                perturb_weight = torch.rand((1, ren_count))
                perturb_weight = perturb_weight / perturb_weight.sum(dim=1).unsqueeze(1)
                rel_budgets = (perturb_weight * self.conn_budget).round().long()
                rounding_error = rel_budgets.sum() - ren_count
            
            # Iterate over relation type
            for idx, val in enumerate(graph.etypes):
                rel_budget = rel_budgets[idx]

                # Get randomized perturbation amount based on the max budget
                counter = dict(Counter(replay_edge[val]['in']['dst'].tolist()))
                degrees = torch.tensor([counter[id] for id in new_ids.tolist()], dtype=torch.long)
                perturb_amount = torch.randint(0, rel_budget + 1, degrees.shape)

                # Split amount to deletion and addition
                max_minus = ((degrees + rel_budget) / 2).floor() # Maximum deletion possible
                perturb_minus = torch.minimum(((torch.rand(degrees.shape) * perturb_amount).round()), max_minus) # Deletion count capped by the max
                perturb_cancels = torch.minimum((degrees - perturb_minus), torch.zeros(degrees.shape)).abs() # Amount of plus and minus that cancels out if any

                perturb_minus = (perturb_minus - perturb_cancels).long()
                perturb_plus = (perturb_amount - perturb_minus - perturb_cancels).long()

                # Final to-do list per node
                to_dos = list(zip(new_ids.tolist(), perturb_minus.tolist(), perturb_plus.tolist()))

                # Iterate over all nodes
                reduceds, addeds = [], []
                for id, min_count, plus_count in to_dos:
                    current_index = (replay_edge[val]['in']['dst'] == id).nonzero().flatten().tolist() # All index of Node's edge

                    # Get index after reduction and index for addition base
                    reduced_index = sorted(np.random.choice(current_index, len(current_index) - min_count, replace=False)) # New list of index after reduction
                    added_index = sorted(np.random.choice(current_index, plus_count, replace=False))

                     # REDUCTION - Copy to new container
                    reduced_data = {}
                    reduced_data['in'] = { feat:replay_edge[val]['in'][feat][reduced_index].clone() for feat in replay_edge[val]['in'].keys() }
                    reduced_data['out'] = { feat:reduced_data['in'][feat].clone() for feat in reduced_data['in'].keys() }
                    reduced_data['out']['src'] = reduced_data['in']['dst'].clone()
                    reduced_data['out']['dst'] = reduced_data['in']['src'].clone()

                    # ADDITION - Rewire and add to new container
                    added_data = {}
                    added_data['in'] = { feat:replay_edge[val]['in'][feat][added_index].clone() for feat in replay_edge[val]['in'].keys() }
                    added_data['in']['src'] = torch.randint(min(new_ids), added_data['in']['src'].shape) # Rewiring
                    added_data['out'] = { feat:added_data['in'][feat].clone() for feat in reduced_data['in'].keys() }
                    added_data['out']['src'] = added_data['in']['dst'].clone()
                    added_data['out']['dst'] = added_data['in']['src'].clone()

                    reduceds.append(reduced_data)
                    addeds.append(added_data)

                replay_edge[val]['in'] = { feat: torch.cat([d['in'][feat] for d in reduceds + addeds]) for feat in reduced_data['in'].keys() }
                replay_edge[val]['out'] = { feat: torch.cat([d['out'][feat] for d in reduceds + addeds]) for feat in reduced_data['out'].keys() }
        
        return replay_node, replay_edge, old_ids, new_ids