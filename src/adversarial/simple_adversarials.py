import torch
import numpy as np

from numpy import random
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

        # Perturb feature
        feats = replay_node['feature'].clone()
        perturb_weight = torch.rand(feats.shape)
        perturb_weight = perturb_weight / perturb_weight.sum(dim=1).unsqueeze(1) # This is the distribution for the noise over the entire feature dimension for each node
        perturb_amount = torch.rand(feats.shape[0]) * self.feat_budget # This is the amount of noise for each node
        perturb_final = (perturb_weight * perturb_amount.unsqueeze(1)) * (torch.rand(feats.shape) - 0.5).sign() # Randomize noise sign
        replay_node['feature'] = feats + perturb_final

        # Perturb connection
        for i in range(old_ids):
            print(i)
        
        return replay_node, replay_edge, old_ids, new_ids