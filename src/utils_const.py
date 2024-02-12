from models.benchmarks_supervised.simple import GCN, GraphSAGE
from models.benchmarks_supervised.spectral import BWGNN
from models.benchmarks_supervised.h2fd import H2FD
from adversarial.simple_adversarials import ReplayAdversary, PerturbationAdversary

### CONSTANTS ###
# Model Dictionary
model_dict = {
    # Standard GNNs
    'GCN': GCN,
    'GraphSAGE': GraphSAGE,
    'H2F-DETECTOR': H2FD,
    'BWGNN': BWGNN
}

# Adversarial Dictionary
adversarial_dict = {
    'REPLAY': ReplayAdversary,
    'PERTURB': PerturbationAdversary
}

