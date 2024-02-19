import torch
import numpy as np

from numpy import random
from collections import Counter
from adversarial.adversarial import BaseAdversary
from utils_func import verPrint

#################
# SIMPLE REPLAY #
#################

class ReplayAdversary(BaseAdversary):
    def __init__(self, greedy_seed=False, verbose=0, **kwargs):
        verPrint(verbose, 3, f'START - ReplayAdversary:__init__ | greedy_seed:{greedy_seed}')

        super().__init__()
        self.verbose=verbose       
        self.greedy_seed = greedy_seed

        verPrint(verbose, 3, f'FINISH - ReplayAdversary:__init__')

    def generate(self, graph, n_instances=10, return_ids=False, **kwargs):
        verPrint(self.verbose, 3, f'START - ReplayAdversary:generate | n_instances: {n_instances},  return_ids: {return_ids}')        
        
        prio_pool = torch.tensor([], dtype=torch.long) if (not self.greedy_seed) else ((graph.ndata['predicted'] == False) & (graph.ndata['label'] == 1)).nonzero().flatten() # Get prioritized pool if greedy
        data = BaseAdversary.random_duplicate(graph, n_instances=n_instances, label=1, return_ids=return_ids, prio_pool=prio_pool, verbose=self.verbose)
        
        verPrint(self.verbose, 3, f'FINISH - ReplayAdversary:generate | prio_pool {prio_pool.tolist()}')
        return data

######################
# PERTURBATION BASED #
######################
    
class BasePerturbationAdversary(BaseAdversary):
    def __init__(self, feat_budget=1.0, conn_budget=0.1, greedy_seed=False, verbose=0, **kwargs):
        verPrint(verbose, 3, f'START - {type(self).__name__}:__init__ | feat_budget: {feat_budget},  conn_budget: {conn_budget}, greedy_seed: {greedy_seed}')

        super().__init__()
        self.verbose=verbose              
        self.greedy_seed = greedy_seed
        self.feat_budget = feat_budget
        self.conn_budget = conn_budget

        verPrint(verbose, 3, f'FINISH - {type(self).__name__}:__init__')
    
    @staticmethod
    def get_rewires(todos, edge_data, relname, baseid, verbose=0):
        verPrint(verbose, 3, f'START - BasePerturbationAdversary:get_rewires | todos: {todos},  edge_data_shape: {edge_data.shape}, relname: {relname}, baseid: {baseid}')

        reduceds, addeds = [], [] # Container
        for id, min_count, plus_count in todos:
            verPrint(verbose, 4, f'Rewiring node id {id} | removing {min_count} edges and adding {plus_count} edges')

            # Get index after reduction and index for addition base
            current_index = (edge_data[relname]['in']['dst'] == id).nonzero().flatten().tolist() # All index of Node's edge
            reduced_index = sorted(np.random.choice(current_index, len(current_index) - min_count, replace=False)) # New list of index after reduction
            added_index = sorted(np.random.choice(current_index, plus_count, replace=True))

            # REDUCTION - Copy to new container
            reduced_data = {}
            reduced_data['in'] = { feat:edge_data[relname]['in'][feat][reduced_index].clone() for feat in edge_data[relname]['in'].keys() }
            reduced_data['out'] = { feat:reduced_data['in'][feat].clone() for feat in reduced_data['in'].keys() }
            reduced_data['out']['src'] = reduced_data['in']['dst'].clone()
            reduced_data['out']['dst'] = reduced_data['in']['src'].clone()

            # ADDITION - Rewire and add to new container
            added_data = {}
            added_data['in'] = { feat:edge_data[relname]['in'][feat][added_index].clone() for feat in edge_data[relname]['in'].keys() }
            added_data['in']['src'] = torch.randint(baseid, added_data['in']['src'].shape) # Rewiring
            added_data['out'] = { feat:added_data['in'][feat].clone() for feat in reduced_data['in'].keys() }
            added_data['out']['src'] = added_data['in']['dst'].clone()
            added_data['out']['dst'] = added_data['in']['src'].clone()

            # Append to container
            reduceds.append(reduced_data)
            addeds.append(added_data)

        verPrint(verbose, 3, f'Finish - BasePerturbationAdversary:get_rewires | todos: reduceds_shape: {reduceds.shape}, addeds_shape: {addeds.shape}')
        return reduceds, addeds

    @staticmethod
    def split_connection_budget(degrees, perturb_amount, verbose=0):
        verPrint(verbose, 3, f'START - BasePerturbationAdversary:split_connection_budget | degrees: {degrees},  perturb_amount: {perturb_amount}')

        max_minus = ((degrees + perturb_amount) / 2).floor() - 1 # Maximum deletion possible, - 1 to prevent 0 degree node
        perturb_minus = torch.minimum(((torch.rand(degrees.shape) * perturb_amount).round()), max_minus) # Deletion count capped by the max
        perturb_cancels = torch.minimum((degrees - perturb_minus), torch.zeros(degrees.shape)).abs() # Amount of plus and minus that cancels out if any

        perturb_minus = (perturb_minus - perturb_cancels).long()
        perturb_plus = (perturb_amount - perturb_minus - perturb_cancels).long()

        verPrint(verbose, 3, f'FINISH - BasePerturbationAdversary:split_connection_budget | perturb_minus: {perturb_minus},  perturb_plus: {perturb_plus}')
        return perturb_minus, perturb_plus

class RelativePerturbationAdversary(BasePerturbationAdversary):    
    def generate(self, graph, n_instances=1, return_ids=False, **kwargs):
        verPrint(self.verbose, 3, f'START - RelativePerturbationAdversary:generate | n_instances: {n_instances},  return_ids: {return_ids}, greedy: {self.greedy_seed}')

        # Get seed node with priority pool if any
        prio_pool = torch.tensor([], dtype=torch.long) if (not self.greedy_seed) else ((graph.ndata['predicted'] == False) & (graph.ndata['label'] == 1)).nonzero().flatten()
        replay_node, replay_edge, old_ids, new_ids =  BaseAdversary.random_duplicate(graph, n_instances=n_instances, label=1, return_ids=return_ids, prio_pool=prio_pool, verbose=self.verbose)

        ## FEATURE PERTURBATION ##
        verPrint(self.verbose, 4, f'Perturbing each feature at max for {self.feat_budget} times its stdev...')
        feats = replay_node['feature'].clone()
        perturb_final = torch.std(feats, dim=0) * ((torch.rand(feats.shape) - 0.5) * 2) * self.feat_budget
        replay_node['feature'] = feats + perturb_final

        ## STRUCTURAL PERTURBATION ## TODO: Directed version
        for val in [r for r in graph.etypes if r != 'homo']:
            verPrint(self.verbose, 4, f'Perturbing structure for relation {val} at most for {self.conn_budget} times its degree')

            # Get randomized perturbation amount based on the max budget
            counter = dict(Counter(replay_edge[val]['in']['dst'].tolist()))
            degrees = torch.tensor([counter[id] for id in new_ids.tolist()], dtype=torch.long) # Get degrees of each nodes
            perturb_amount = (degrees * torch.rand(degrees.shape) * self.conn_budget).round()
            perturb_minus, perturb_plus = BasePerturbationAdversary.split_connection_budget(degrees, perturb_amount)
            
            # Get rewiring based on perturbation amount
            todos = list(zip(new_ids.tolist(), perturb_minus.tolist(), perturb_plus.tolist()))
            reduceds, addeds = BasePerturbationAdversary.get_rewires(todos, replay_edge, val, min(new_ids))

            # Replace data in seed container
            replay_edge[val]['in'] = { feat: torch.cat([d['in'][feat] for d in reduceds + addeds]) for feat in reduceds[0]['in'].keys() }
            replay_edge[val]['out'] = { feat: torch.cat([d['out'][feat] for d in reduceds + addeds]) for feat in reduceds[0]['out'].keys() }
        
        verPrint(self.verbose, 3, f'FINISH - RelativePerturbationAdversary:generate | replay_node: {replay_node}, replay_edge: {replay_edge}, old_ids: {old_ids},  return_ids: {return_ids}')
        return replay_node, replay_edge, old_ids, new_ids

class AbsolutePerturbationAdversary(BasePerturbationAdversary):    
    def generate(self, graph, n_instances=1, return_ids=False, **kwargs):
        verPrint(self.verbose, 3, f'START - AbsolutePerturbationAdversary:generate | n_instances: {n_instances},  return_ids: {return_ids}, greedy: {self.greedy_seed}')

        # Get seed node with priority pool if any
        prio_pool = torch.tensor([], dtype=torch.long) if (not self.greedy_seed) else ((graph.ndata['predicted'] == False) & (graph.ndata['label'] == 1)).nonzero().flatten()
        replay_node, replay_edge, old_ids, new_ids =  BaseAdversary.random_duplicate(graph, n_instances=n_instances, label=1, return_ids=return_ids, prio_pool=prio_pool, verbose=self.verbose)

        ## FEATURE PERTURBATION ##
        verPrint(self.verbose, 4, f'Perturbing feature with absolute an budget of {self.feat_budget}...')
        feats = replay_node['feature'].clone()
        perturb_weight = torch.rand(feats.shape)
        perturb_weight = perturb_weight / perturb_weight.sum(dim=1).unsqueeze(1) # This is the distribution for the noise over the entire feature dimension for each node
        perturb_amount = torch.rand(feats.shape[0]) * self.feat_budget # This is the amount of noise for each node
        perturb_final = (perturb_weight * perturb_amount.unsqueeze(1)) * (torch.rand(feats.shape) - 0.5).sign() # Randomize noise sign
        replay_node['feature'] = feats + perturb_final

        ## STRUCTURAL PERTURBATION ## TODO: Directed version
        verPrint(self.verbose, 4, f'Perturbing structure with an absolute budget of {self.conn_budget}...')
        
        # Distribute budget over relations
        rels = [r for r in graph.etypes if r != 'homo'] # Exception for H2F
        rounding_error = 1
        while rounding_error != 0: # Split budget over all edge relations
            perturb_weight = torch.rand((1, len(rels)))
            perturb_weight = perturb_weight / perturb_weight.sum(dim=1).unsqueeze(1)
            rel_budgets = (perturb_weight * self.conn_budget).round().long()
            rounding_error = rel_budgets.sum() - self.conn_budget
        
        # Iterate over relation type
        for idx, val in enumerate(rels):
            verPrint(self.verbose, 4, f'Perturbing structure for relation {val} with budget {rel_budgets[idx]}')

            # Get randomized perturbation amount based on the max budget
            counter = dict(Counter(replay_edge[val]['in']['dst'].tolist()))
            degrees = torch.tensor([counter[id] for id in new_ids.tolist()], dtype=torch.long)
            perturb_amount = torch.randint(0, rel_budgets[idx] + 1, degrees.shape)
            perturb_minus, perturb_plus = BasePerturbationAdversary.split_connection_budget(degrees, rel_budgets[idx])

            # Final to-do list per node
            todos = list(zip(new_ids.tolist(), perturb_minus.tolist(), perturb_plus.tolist()))
            reduceds, addeds = BasePerturbationAdversary.get_rewires(todos, replay_edge, val, min(new_ids), verbose=self.verbose)

            # Replace data in seed container
            replay_edge[val]['in'] = { feat: torch.cat([d['in'][feat] for d in reduceds + addeds]) for feat in reduceds[0]['in'].keys() }
            replay_edge[val]['out'] = { feat: torch.cat([d['out'][feat] for d in reduceds + addeds]) for feat in reduceds[0]['out'].keys() }
    
        verPrint(self.verbose, 3, f'FINISH - RelativePerturbationAdversary:generate | replay_node: {replay_node}, replay_edge: {replay_edge}, old_ids: {old_ids},  return_ids: {return_ids}')
        return replay_node, replay_edge, old_ids, new_ids