import torch
import numpy as np

from numpy import random
from adversarial.adversarial import BaseAdversary
from utils_func import random_duplicate

class ReplayAdversary(BaseAdversary):
    def __init__(self):
        super().__init__()
        return
    
    def generate(self, graph, n_instances=1, is_random=True, return_seed=False):     
        return random_duplicate(graph, n_instances=n_instances, label=1, return_seed=return_seed)
    
class PerturbationAdversary(BaseAdversary):
    def __init__(self):
        super().__init__()
        return
    
    def generate(self, graph, n_instances=1, is_random=True):
        return 0