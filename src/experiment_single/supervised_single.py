from utils import verPrint
from models.benchmarks_supervised import GCN, GHRN, H2FD, CAREGNN


import dgl
import torch

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

### BASE EXPERIMENT CLASS ###
class BaseExperiment(object):
    ## Helpers
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

    ## Class Methods
    def __init__(self, model_config, train_config, graph, verbose=0):
        self.verbose=verbose

        self.dset = {'graph': graph}
        self.model_config = model_config
        self.train_config = train_config      
        
        features = self.dset['graph'].ndata['feature']
        labels = self.dset['graph'].ndata['label']
        
        in_dimension = features.shape[1]
        class_num =  labels.unique(return_counts=True)[0].shape[0]

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
        self.dset['train_mask'] = torch.zeros([len(labels)]).bool()
        self.dset['val_mask'] = torch.zeros([len(labels)]).bool()
        self.dset['test_mask'] = torch.zeros([len(labels)]).bool()

        self.dset['train_mask'][idx_train] = 1
        self.dset['val_mask'][idx_valid] = 1
        self.dset['test_mask'][idx_test] = 1

        # Additional inits
        self.train_config['ce_weight'] = (1-labels[self.dset['train_mask']]).sum().item() / labels[self.dset['train_mask']].sum().item()
        if train_config['train_mode'] == 'batch':
            model_config['train_mode'] = 'batch'

            self.dset['sampler'] = dgl.dataloading.MultiLayerFullNeighborSampler(model_config['num_layers'])
            self.dset['dataloader'] = dgl.dataloading.DataLoader(
                self.dset['graph'], idx_train, self.dset['sampler'],
                batch_size=train_config['batch_size'], shuffle=True, drop_last=False, num_workers=train_config['num_workers']
            )
            
    
        # Model
        self.model = model_dict[model_config['model_name']](in_dimension, class_num, model_config, verbose=self.verbose)
    
        if train_config['train_mode'] == 'batch':
            self.model.cuda()

    # Base train is just when model outputs logits and normal backprop
    def train(self):
        # Inits
        best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.
        self.optimizer = self.train_config['optimizer'](self.model.parameters(), lr=self.train_config['learning_rate'])
        self.loss = self.train_config['loss']

        features = self.dset['graph'].ndata['feature']
        labels = self.dset['graph'].ndata['label']

        # Main Training Loop
        time_start = time.time()
        verPrint(self.verbose, 1, 'Starting training!')
        for e in range(self.train_config['num_epoch']):
            self.model.train()
            self.logits = torch.zeros([len(labels), 2])

            if self.train_config['train_mode'] != 'batch':
                # Forward pass
                verPrint(self.verbose, 2, 'Forward')
                logits = self.model(self.dset['graph'], self.dset['graph'].ndata['feature'])
                self.logits = logits
                epoch_loss = self.loss(logits[self.dset['train_mask']], labels[self.dset['train_mask']], weight=torch.tensor([1., self.train_config['ce_weight']]))

                # Backward pass
                verPrint(self.verbose, 2, 'Backward')
                self.optimizer.zero_grad()
                epoch_loss.backward()
                self.optimizer.step()
            else:
                i = 0
                for input_nodes, output_nodes, blocks in self.dset['dataloader']:
                    i += 1

                    # Batch forward
                    verPrint(self.verbose, 2, f'Forward for batch {i}')
                    blocks = [b.to(torch.device('cuda')) for b in blocks]
                    input_features = blocks[0].srcdata['feature']
                    output_labels = blocks[-1].dstdata['label']

                    logits = self.model(blocks, input_features)
                    self.logits[output_nodes] = logits.cpu()

                    epoch_loss = self.loss(logits, output_labels, weight=torch.tensor([1., self.train_config['ce_weight']]).to(torch.device('cuda')))

                    # Backward pass
                    verPrint(self.verbose, 2, 'Backward')
                    self.optimizer.zero_grad()
                    epoch_loss.backward()
                    self.optimizer.step()

            # Evaluate
            verPrint(self.verbose, 1, 'Evaluate')
            self.model.eval()
            probs = self.logits.softmax(1)

            f1, thres = self.get_best_f1(labels[self.dset['val_mask']], probs[self.dset['val_mask']])
            preds = np.zeros_like(labels)
            preds[probs[:, 1] > thres] = 1

            trec = recall_score(labels[self.dset['test_mask']], preds[self.dset['test_mask']])
            tpre = precision_score(labels[self.dset['test_mask']], preds[self.dset['test_mask']])
            tmf1 = f1_score(labels[self.dset['test_mask']], preds[self.dset['test_mask']], average='macro')
            tauc = roc_auc_score(labels[self.dset['test_mask']], probs[self.dset['test_mask']][:, 1].detach().numpy())

            if best_f1 < f1:
                best_f1 = f1
                final_trec = trec
                final_tpre = tpre
                final_tmf1 = tmf1
                final_tauc = tauc
            verPrint(self.verbose, 1, 'Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, epoch_loss, f1, best_f1))

        time_end = time.time()
        verPrint(self.verbose, 1, 'time cost: ', time_end - time_start, 's')
        verPrint(self.verbose, 1, 'Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec*100,
                                                                            final_tpre*100, final_tmf1*100, final_tauc*100))
        return final_tmf1, final_tauc
    
    def evaluate():
        return
