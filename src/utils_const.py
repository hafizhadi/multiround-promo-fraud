from models.benchmarks_supervised import GCN, GHRN, H2FD, CAREGNN, BWGNN
from adversarial.simple_adversarials import ReplayAdversary, PerturbationAdversary

### CONSTANTS ###
# Model Dictionary
model_dict = {
    # Standard GNNs
    'GCN': GCN,
    'GHRN': GHRN,
    'H2F-DETECTOR': H2FD,
    'CARE-GNN':CAREGNN,
    'BWGNN': BWGNN
}

# Adversarial Dictionary
adversarial_dict = {
    'REPLAY': ReplayAdversary,
    'PERTURB': PerturbationAdversary
}

