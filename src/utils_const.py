from models.benchmarks_supervised.simple import GCN, GraphSAGE, GIN, GAT
from models.benchmarks_supervised.spectral import BWGNN
from models.benchmarks_supervised.camouflage import CAREGNN
from models.benchmarks_supervised.h2fd import H2FD
from models.benchmarks_supervised.booster import GraphBoost

from adversarial.simple_adversarials import ReplayAdversary, AbsolutePerturbationAdversary, RelativePerturbationAdversary, MixingAdversary

### CONSTANTS ###
# Model Dictionary
model_dict = {
    # Standard GNNs
    'GCN': GCN,
    'GraphSAGE': GraphSAGE,
    'GIN': GIN,
    'GAT': GAT,
    'H2F-DETECTOR': H2FD,
    'BWGNN': BWGNN,
    'CARE': CAREGNN,
    'XGB': GraphBoost
}

# Adversarial Dictionary
adversarial_dict = {
    'REPLAY': ReplayAdversary,
    'PERTURB-ABS': AbsolutePerturbationAdversary,
    'PERTURB-REL': RelativePerturbationAdversary,
    'MIXING': MixingAdversary
}
