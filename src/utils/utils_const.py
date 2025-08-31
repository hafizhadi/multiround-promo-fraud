import xgboost as xgb

import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.optim import Adam

from models.proposed_supervised.mixed import EmbedBoost

from models.benchmarks_supervised.simple import GCN, GCNII, GraphSAGE, GIN, GAT
from models.benchmarks_supervised.spectral import BWGNN, GHRN
from models.benchmarks_supervised.booster import GraphBoost, GIN_noparam, RoundGIN_noparam, SplitRoundGIN_noparam

from meta_strategies.augment import NoneSampling, RandomReplaySampling, ReAge
from meta_strategies.prediction_addon import NoneAddon, FeatureDistThreshold, AggFeatureDistThreshold, DegreeActivityThreshold, DegreeFeatureThreshold, DegreeAggFeatureThreshold

from adversary.choose.simple_choose import RandomChoose, GreedyChoose, OGGreedyChoose
from adversary.modify.simple_mod import ReplayMod, AbsolutePerturbMod, RelativePerturbMod, MixingMod

#############
### DICTS ###
#############
MODEL_DICT = {
    # Standard GNNs
    'GCN': GCN,
    'GCNII': GCNII,
    'GraphSAGE': GraphSAGE,
    'GIN': GIN,
    'GAT': GAT,
    'BWGNN': BWGNN,
    'GHRN': GHRN,
    'XGB': GraphBoost,    
    'XGB-SP': EmbedBoost,
}


AUGMENT_DICT = {
    'NONE': NoneSampling,
    'RANDOM': RandomReplaySampling,
    'REAGE': ReAge,
}

ADVER_CHOOSE_DICT = {
    'RANDOM': RandomChoose,
    'GREEDY': GreedyChoose,
    'OGREEDY': OGGreedyChoose
}

ADVER_MOD_DICT = {
    'REPLAY': ReplayMod,
    'PERTURB-ABS': AbsolutePerturbMod,
    'PERTURB-REL': RelativePerturbMod,
    'MIXING': MixingMod
}

LOSS_DICT = {
    'ce': F.cross_entropy
}

BACKBONE_DICT = {
    'NONE': None,
    'GIN': GIN_noparam,
    'RGIN': RoundGIN_noparam,
    'SRGIN': SplitRoundGIN_noparam
}

ADDON_DICT = {
    'NONE': NoneAddon,
    'FTHR': FeatureDistThreshold,
    'AFTHR': AggFeatureDistThreshold,
    'DEGREE': DegreeActivityThreshold,
    'DFEAT': DegreeFeatureThreshold,
    'DAFEAT': DegreeAggFeatureThreshold
}

#######################
### DEFAULT CONFIGS ###
#######################

# Overarching configs containing setting on round number, etc
DEFAULT_MAIN_CONFIG = {
    'verbose': 4,
    'device': 'cuda:0',
    'save_df': True,
    'save_embedding': False,
    'save_temp_sne': False,

    'exp_type': 'ADVER', # Type of experiment, ADVER is adversarial, MULTI is multiround non-incremental, INCRE is class incremental
    'task_type': 'NODE', # Type of task, NODE for node classification, GRAPH for graph classification conversion
    'setting_type': 'INDUCTIVE', # Task of setting, TRANSDUCTIVE and INDUCTIVE
    'full_oracle': False,
    
    # Round setting parameter
    'round_num': 10,
    'node_expiration_time': 1,
    
    # Related to generation of new round instances
    'round_new_pos': 20,
    'round_new_neg': 400,
    'round_budget_pos': 10,
    'round_budget_neg': 200,
}

# Config related to model training
DEFAULT_TRAIN_CONFIG = {
    'verbose': 4,
    'random_state': 7777777,

    # Model retraining strategies
    'round_reset_model': False,
    'round_train_list': [],

    # Training hyperparams
    'num_epoch': 500,
    'num_round_epoch': 300,
    'early_stopping': 75,
    'stuck_stopping': 50,
    'stuck_threshold': 0,
    'train_max_retry': 10,

    'ratio_initial': 0.50,
    'ratio_train': 0.60,
    'ratio_val': 0.20,
    'ratio_test': 0.20,
    
    # Learning hyperparams
    'optimizer': Adam,
    'learning_rate': 0.01,
    'loss': F.cross_entropy,
    'loss_gamma': 5,
}

DEFAULT_MODEL_CONFIG = {
    'verbose': 4,
    'model_name': 'XGB',
    'embed_type': 'temporal',

    # General GNN params
    'h_feats': 64,
    'num_layers': 2,
    'mlp_feats': 64,
    'mlp_layers': 2,
    'att_heads': 2,
    'dropout_rate': 0.1,
    'act_name': 'LeakyReLU',
    'norm_name': 'layer',
    
    # Specific for boost models
    'boost_agg_backbone': GIN_noparam,
    'boost_predictor': xgb.XGBClassifier,
    'boost_metric': average_precision_score,
    'n_estimator' : 500,

    # Specific for Round model
    'round_window': 7,
    'temporal_agg': 'mean_final',
    'tloss_type': 'normal',

    # All-purpose common hyperparameter
    'alpha': 1,
    'beta': 1,
    'gamma': 1,
    'lambda_':1,
    'lambda1':1,
    'lambda2':1,
    'k':1, 

    # Specific ones
    'training_type': 'round',
    'loss_type': 'ndist',
    'tloss_type': 'normal',
    'loss_sample': True,
    'loss_sample_ratio': 0.1,
    'drop_rate': 0,

    # Addon prediction strategy
    'addon_name': 'NONE',
    'addon_perc': 0.05,
    'addon_round_window': 7,
    'addon_internal_agg': 'OR',
}

DEFAULT_STRAT_CONFIG = {  
    'verbose': 4,

    # Specific for resampling strat
    'augment_name': 'REAGE',
    'augment_pos_ratio': 0.5,
    'augment_neg_ratio': 0.5,
    'augment_round_split': 4,
    'augment_prob': 0.5,

    # Specific for pseudolabeling strat
    'pseudo_name': 'NONE',
    'pseudo_skip_initial': False,
    'pseudo_new_pos_only': False
}

DEFAULT_ADVER_CONFIG = {
    'verbose': 4,
    'adver_choose_name': 'GREEDY',
    'adver_mod_name': 'REPLAY',

    # Perturbation params
    'adver_feat_coef': 1,
    'adver_conn_coef': 0.1
}

# OTHER CONSTANTS
TEMP_MODEL_SAVE_PATH = '../checkpoint/working_model_file'
