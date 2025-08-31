from models.proposed_supervised.prototype import ProtoFraud, SplitProtoFraud
from models.benchmarks_supervised.simple import GCN, GraphSAGE, GIN, GAT
from models.benchmarks_supervised.spectral import BWGNN
from models.benchmarks_supervised.camouflage import CAREGNN
from models.benchmarks_supervised.h2fd import H2FD
from models.benchmarks_supervised.booster import GraphBoost

from adversarial.simple_adversarials import ReplayAdversary, AbsolutePerturbationAdversary, RelativePerturbationAdversary, MixingAdversary

### CONSTANTS ###

# Model Dictionary
MODEL_DICT = {
    # Standard GNNs
    'GCN': GCN,
    'GraphSAGE': GraphSAGE,
    'GIN': GIN,
    'GAT': GAT,
    'H2F-DETECTOR': H2FD,
    'BWGNN': BWGNN,
    'CARE': CAREGNN,
    'XGB': GraphBoost,
    'PROP-PROTO': ProtoFraud,
    'PROP-SPLITPROTO': SplitProtoFraud
}

# Adversarial Dictionary
ADVERSARIAL_DICT = {
    'REPLAY': ReplayAdversary,
    'PERTURB-ABS': AbsolutePerturbationAdversary,
    'PERTURB-REL': RelativePerturbationAdversary,
    'MIXING': MixingAdversary
}
