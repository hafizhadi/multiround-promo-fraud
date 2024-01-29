from utils_func import verPrint, random_duplicate, get_best_f1, eval_and_print
from utils_const import model_dict, adversarial_dict

import dgl
import torch

import time, psutil, os
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, confusion_matrix

### MULTIROUND EXPERIMENT CLASS ###
class MultiroundExperiment(object):
    ## Class Methods
    def __init__(self, model_config, adver_config, train_config, graph, verbose=0):
        model_config['train_mode'] = train_config['train_mode']
        adver_config['train_mode'] = train_config['train_mode']

        self.verbose=verbose
        self.dset = {'graph': graph}
        self.model_config = model_config
        self.adver_config = adver_config
        self.train_config = train_config      

        features = self.dset['graph'].ndata['feature']
        labels = self.dset['graph'].ndata['label']

        # Initialize model
        in_dimension = features.shape[1]
        class_num =  labels.unique(return_counts=True)[0].shape[0]
        self.model = model_dict[model_config['model_name']](in_dimension, class_num, model_config, verbose=self.verbose)
        if train_config['train_mode'] == 'batch':
            self.model.cuda()

        # Initialize adver
        self.adver = adversarial_dict[adver_config['adver_name']]()

        # Train Test Split
        index = list(range(len(labels)))
        idx_train, idx_rest, y_train, y_rest = train_test_split(
            index, labels[index], stratify=labels[index],
            train_size = train_config['train_ratio'], random_state = train_config['random_state'], shuffle=True
        )
        idx_valid, idx_test, y_valid, y_test = train_test_split(
            idx_rest, y_rest, stratify=y_rest,
            test_size = train_config['test_ratio_from_rest'], random_state = train_config['random_state'], shuffle=True
        )

        self.dset['train_mask'] = torch.zeros([len(labels)]).bool()
        self.dset['val_mask'] = torch.zeros([len(labels)]).bool()
        self.dset['test_mask'] = torch.zeros([len(labels)]).bool()

        self.dset['train_mask'][idx_train] = 1
        self.dset['val_mask'][idx_valid] = 1
        self.dset['test_mask'][idx_test] = 1

        self.train_config['ce_weight'] = (1-labels[self.dset['train_mask']]).sum().item() / labels[self.dset['train_mask']].sum().item()

        # Initialize round information
        self.current_round = 0
        self.rounds = []
        self.dset['graph'].ndata['creation_round'] = torch.full([len(labels)], 0)

        # Sampler for Batch training
        if train_config['train_mode'] == 'batch':
            self.dset['sampler'] = dgl.dataloading.MultiLayerFullNeighborSampler(model_config['num_layers'])
            self.dset['dataloader'] = dgl.dataloading.DataLoader(
                self.dset['graph'], idx_train, self.dset['sampler'],
                batch_size=train_config['batch_size'], shuffle=True, drop_last=False, num_workers=train_config['num_workers']
            )
            

    # Initial training for the first round
    def model_initial_train(self):
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

                    # The following code is used to record the memory usage
                    py_process = psutil.Process(os.getpid())
                    verPrint(self.verbose, 3, f"CPU Memory Usage: {py_process.memory_info().rss / (1024 ** 3)} GB")
                    verPrint(self.verbose, 3, f"GPU Memory Usage: {torch.cuda.memory_reserved() / (1024 ** 3)} GB")

            # Evaluate
            verPrint(self.verbose, 2, 'Evaluate')
            self.model.eval()
            probs = self.logits.softmax(1)

            f1, thres = get_best_f1(labels[self.dset['val_mask']], probs[self.dset['val_mask']])
            preds = np.zeros_like(labels)
            preds[probs[:, 1] > thres] = 1

            trec, tpre, tmf1, tauc = eval_and_print(0, labels[self.dset['test_mask']], preds[self.dset['test_mask']], probs[self.dset['test_mask']][:, 1], f'Epoch {e}')
            if best_f1 < f1:
                best_f1 = f1
                final_trec = trec
                final_tpre = tpre
                final_tmf1 = tmf1
                final_tauc = tauc
            verPrint(self.verbose, 1, 'Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, epoch_loss, f1, best_f1))

        time_end = time.time()
        verPrint(self.verbose, 2, f'time cost: {str(time_end - time_start)} s')
        verPrint(self.verbose, 1, 'Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec*100,
                                                                            final_tpre*100, final_tmf1*100, final_tauc*100))
        return final_tmf1, final_tauc
    
    # Round training using additional data on round
    def model_round_train(self):
        return

    # Round prediction on round
    def model_round_predict(self):
        self.model.eval()
        
        labels = self.dset['graph'].ndata['label']
        probs = self.model(self.dset['graph'], self.dset['graph'].ndata['feature']).softmax(1)
        f1, thres = get_best_f1(labels[self.dset['val_mask']], probs[self.dset['val_mask']])
       
        preds = torch.zeros_like(self.dset['graph'].ndata['label'])
        preds[probs[:, 1] > thres] = 1
        
        # Model is provided false positives and true positives
        tp = ((labels == preds) & (labels == 1)).nonzero().flatten()
        fp = ((labels != preds) & (labels == 0)).nonzero().flatten()
        tn = ((labels == preds) & (labels == 0)).nonzero().flatten()
        fn = ((labels != preds) & (labels == 1)).nonzero().flatten()

        return preds, probs, tp, fp, tn, fn
    
    # adver training using provided data
    def adversary_round_train(self):
        return

    # adver generation of new positive instances
    def adversary_round_generate(self):
        return self.adver.generate(self.dset['graph'], num_instances=self.train_config['round_pos_count'])
    
    # Generate negative instances
    def round_generate_negatives(self):
        return random_duplicate(self.dset['graph'], self.train_config['round_neg_count'])
    
    # Execute 1 adver round based on the current state of the experiment
    def adver_round(self, round):
        r_idx = round - 1

        # Initialization and check to see if inputted round number is valid (i.e. the previous round has been conducted)
        if(len(self.rounds) < r_idx):
            return
    
        # Reset all subsequent round data

        # TODO: Torch snapshot to reset model and adver
        self.current_round = round
        self.rounds = self.rounds[:r_idx]
        self.dset['graph'] = dgl.remove_nodes(self.dset['graph'], (self.dset['graph'].ndata['creation_round'] >= round).nonzero().flatten())

        self.rounds.append({})

        # ROUND
        if round > 1:
            # Train model and adversary based on last round info
            # TODO: self.model_round_train(round)
            # TODO: self.adversary_round_train(round)

            # Generate additional data for round
            new_adv_nodes, new_adv_edges = self.adversary_round_generate(round)
            new_neg_nodes, new_neg_edges = self.round_generate_negatives(round)

            new_neg_edges['src'] = new_neg_edges['src'] + len(new_adv_nodes['label'])
            new_neg_edges['dst'] = new_neg_edges['dst'] + len(new_adv_nodes['label'])

            new_nodes = {key:torch.cat((new_adv_nodes[key], new_neg_nodes[key]), 0) for key in new_adv_nodes.key()}
            new_nodes['creation_round'] = torch.full([len(new_nodes['label'])], round)
            
            new_edges = {key:torch.cat((new_adv_edges[key], new_neg_edges[key]), 0) for key in new_adv_edges.key()}

            # Update graph
            edge_src = new_edges['src']
            edge_dst = new_edges['dst']
            del new_edges['src'], new_edges['dst']

            self.dset['graph'].add_nodes(len(new_nodes), new_nodes)
            self.dset['graph'].add_edges(edge_src, edge_dst, new_edges)

            # TODO: Update node and ground truth masks

        # Model predict
        round_preds, round_probs, round_tn, round_fp, round_tp, round_fn = self.model_round_predict()
        self.rounds[r_idx]['prediction'] = round_preds

        # Round evaluation
        round_mask = (self.dset['graph'].ndata['creation_round'] == round).nonzero()
        labels = self.dset['graph'].ndata['label']
        if len(labels[round_mask]) > 0:
            _ = eval_and_print(self.verbose, labels[round_mask], round_preds[round_mask], round_probs[round_mask], 'Round')
        else:
            verPrint(self.verbose, 1, 'No round prediction.')

        _ = eval_and_print(self.verbose, labels, round_preds, round_probs, 'Overall')
