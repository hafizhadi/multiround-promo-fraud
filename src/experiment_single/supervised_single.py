import dgl
from dgl.nn import GraphConv
from models.benchmarks_supervised import GCN, GHRN, H2FD, CAREGNN

import torch
import torch.nn as nn

import time
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix

## MODEL DICTIONARY
model_dict = {
    # Standard GNNs
    'GCN': GCN,
    'GHRN': GHRN,
    'H2F-DETECTOR': H2FD,
    'CARE-GNN':CAREGNN
}

### HELPERS ###
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

### BASE EXPERIMENT CLASS ###
class BaseExperiment(object):
    def __init__(self, model_config, train_config, graph):

        self.data = {'graph': graph}
        self.model_config = model_config
        self.train_config = train_config      
        
        features = self.graph.ndata['feature']
        labels = self.graph.ndata['label']
        
        in_dimension = features.shape[1]
        class_num =  labels.shape[0]

        index = list(range(len(labels)))

        # Train Test Split
        idx_train, idx_rest, y_train, y_rest = train_test_split(
            index, labels[index], stratify=labels[index],
            train_size = train_config['train_ratio'], random_state = train_config['random_state'], shuffle=True
        )
        idx_valid, idx_test, y_valid, y_test = train_test_split(
            idx_rest, y_rest, stratify=y_rest,
            test_size = train_config['test_ratio_from_rest'], random_state = train_config['random_state'], shuffle=True
        )

        # Assign masks
        self.data['train_mask'] = torch.zeros([len(labels)]).bool()
        self.data['val_mask'] = torch.zeros([len(labels)]).bool()
        self.data['test_mask'] = torch.zeros([len(labels)]).bool()

        self.data['train_mask'][idx_train] = 1
        self.data['val_mask'][idx_valid] = 1
        self.data['test_mask'][idx_test] = 1

        # Additional inits
        self.train_config['ce_weight'] = (1-labels[self.train_mask]).sum().item() / labels[self.train_mask].sum().item()
    
        # Model
        self.model = model_dict[model_config['model_name']](in_dimension, class_num, model_config)

    # Base train is just when model outputs logits and normal backprop
    def train(self):
        # Inits
        best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.
        self.optimizer = self.train_config['optimizer'](self.model.parameters(), lr=self.train_config['learning_rate'])
        self.loss = self.train_config['loss']

        features = self.data['graph'].ndata['feature']
        labels = self.data['graph'].ndata['label']

        # Main Training Loop
        time_start = time.time()
        for e in range(self.train_config['num_epoch']):
            # Forward pass
            self.model.train()
            logits = self.model(self.data['graph'])
            epoch_loss = self.loss(logits[self.data['train_mask']], labels[self.data['train_mask']], weight=torch.tensor([1., self.train_config['ce_weight']]))

            # Backward pass
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

            # Evaluate
            self.model.eval()
            probs = logits.softmax(1)

            f1, thres = get_best_f1(labels[self.data['val_mask']], probs[self.data['val_mask']])
            preds = np.zeros_like(labels)
            preds[probs[:, 1] > thres] = 1

            trec = recall_score(labels[self.data['test_mask']], preds[self.data['test_mask']])
            tpre = precision_score(labels[self.data['test_mask']], preds[self.data['test_mask']])
            tmf1 = f1_score(labels[self.data['test_mask']], preds[self.data['test_mask']], average='macro')
            tauc = roc_auc_score(labels[self.data['test_mask']], probs[self.data['test_mask']][:, 1].detach().numpy())

            if best_f1 < f1:
                best_f1 = f1
                final_trec = trec
                final_tpre = tpre
                final_tmf1 = tmf1
                final_tauc = tauc
            print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, epoch_loss, f1, best_f1))

        time_end = time.time()
        print('time cost: ', time_end - time_start, 's')
        print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec*100,
                                                                            final_tpre*100, final_tmf1*100, final_tauc*100))
        return final_tmf1, final_tauc
    
    def evaluate():
        return
