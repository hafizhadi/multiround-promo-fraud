import time
import dgl
import torch
import numpy as np

from collections import Counter
from numpy import random
from sklearn.model_selection import train_test_split

from adversarial.adversarial import BaseAdversary
from utils.utils_func import verPrint, get_best_f1, eval_and_print
from utils.utils_const import MODEL_DICT, ADVERSARIAL_DICT

class MultiroundExperiment(object):
    #######################
    ### INITIALIZATIONS ###
    #######################

    # Class init
    def __init__(
            self, graph, 
            model_config, adver_config, train_config
        ):
        
        # Attributes
        model_config['train_mode'] = train_config['train_mode']
        adver_config['train_mode'] = train_config['train_mode']
        model_config['etypes'] = ['none'] if graph.is_homogeneous else graph.etypes
        
        self.model_config = model_config
        self.adver_config = adver_config
        self.train_config = train_config

        self.verbose = train_config['verbose']
        self.dset = { 'graph': graph }

        # Model & Adversary
        self.init_model()
        self.init_adversarial()

        # Initialize round information
        self.current_round, self.rounds = 0, []
        self.dset['graph'].ndata['creation_round'] = torch.full([graph.num_nodes()], -1, dtype=torch.long)
        self.dset['graph'].ndata['predicted'] = torch.full([graph.num_nodes()], True, dtype=torch.bool)

        # Other ndatas
        self.dset['graph'].ndata['train_mask'] = torch.zeros([graph.num_nodes()]).bool()
        self.dset['graph'].ndata['val_mask'] = torch.zeros([graph.num_nodes()]).bool()
        self.dset['graph'].ndata['test_mask'] = torch.zeros([graph.num_nodes()]).bool()
    
    # Initialize model
    def init_model(self):
        in_feats = self.dset['graph'].ndata['feature'].shape[1]
        num_classes =  self.dset['graph'].ndata['label'].unique(return_counts=True)[0].shape[0]
        self.model = MODEL_DICT[self.model_config['model_name']](**(self.model_config | {'in_feats':in_feats, 'num_classes':num_classes}))
        
        # Batch mode inits
        if self.train_config['train_mode'] == 'batch':
            self.model.cuda()
    
    # Initialize adversarial
    def init_adversarial(self):
        self.adver = ADVERSARIAL_DICT[self.adver_config['adver_name']](**self.adver_config)

        # TODO: Batch Mode Init
    
    ####################################
    ### GRAPH/DATA RELATED FUNCTIONS ###
    ####################################
        
    # Split train test
    def split_train_test(self, round, full_training):
        verPrint(self.verbose, 2, f"Alotting train-val-test split for round {round}")

        # Various sanity checks
        if (torch.sum(labels[index] == 0) < 2) or (torch.sum(labels[index] == 1) < 2):
            return None, None, None, None

        # Actual function code
        labels = self.dset['graph'].ndata['label']
        self.dset['graph'].ndata['train_mask'] = torch.zeros([len(labels)]).bool()
        self.dset['graph'].ndata['val_mask'] = torch.zeros([len(labels)]).bool()
        self.dset['graph'].ndata['test_mask'] = torch.zeros([len(labels)]).bool()

        # Build pool of training data
        if round == 0: 
            # For first round, just use the entire graph for training
            initial_pool, prediction_pool, budget_pool = [], [], []
            full_pool = torch.arange(len(labels), dtype=torch.long)
        else:
            # Otherwise, determine the rounds from which to generate the training data from
            init_round = 0 if full_training else -1
            source_rounds = list(range(round)) if full_training else [round-1]
         
            # Exctracting the data based on the pool
            initial_pool = (self.dset['graph'].ndata['creation_round'] < init_round).nonzero().flatten() 
            prediction_pool = torch.cat([torch.cat(self.rounds[i]['checks'][:1], 0) for i in source_rounds])
            budget_pool = torch.cat([torch.cat(self.rounds[i]['budgets'], 0) for i in source_rounds])
            full_pool = torch.cat([initial_pool, prediction_pool, budget_pool], 0).long()

        verPrint(self.verbose, 3, f"Initial pool: {len(initial_pool)}, Prediction pool: {len(prediction_pool)}, Budget pool: {len(budget_pool)}, Full pool: {len(full_pool)}")

        index = torch.arange(len(labels), dtype=torch.long)[full_pool]        
        nonindex = torch.ones_like(labels, dtype=bool)
        nonindex[full_pool] = False

        # Split
        idx_train, idx_valid, y_train, y_valid = train_test_split(
            index, labels[index], stratify=labels[index],
            train_size = self.train_config['train_ratio'], random_state = self.train_config['random_state'], shuffle=True
        )
        
        # Set result in graph data
        self.dset['graph'].ndata['train_mask'][idx_train] = 1
        self.dset['graph'].ndata['val_mask'][idx_valid] = 1
        self.dset['graph'].ndata['test_mask'][nonindex] = 1
 
        # Recalculate CE weight for training
        self.train_config['ce_weight'] = (1-labels[self.dset['graph'].ndata['train_mask']]).sum().item() / labels[self.dset['graph'].ndata['train_mask']].sum().item()
        verPrint(self.verbose, 2, f"Updated cross-entropy weight to {self.train_config['ce_weight']}")
        
        return idx_train, idx_valid, y_train, y_valid

    # Add generated new nodes to graph
    def add_generated_data(self, data):
        new_nodes, new_edges = data

        # Add nodes
        new_nodes['creation_round'] = torch.full([len(new_nodes['label'])], self.current_round)
        new_nodes['predicted'] = torch.full([len(new_nodes['label'])], False)

        new_nodes['train_mask'] = torch.full([len(new_nodes['label'])], 0).bool()
        new_nodes['val_mask'] = torch.full([len(new_nodes['label'])], 0).bool()
        new_nodes['test_mask'] = torch.full([len(new_nodes['label'])], 1).bool()
        self.dset['graph'].add_nodes(len(new_nodes['label']), new_nodes)
        
        # Add edges TODO: edge features?
        for etype in new_edges.keys():        
            for dir in new_edges[etype].keys(): # Incoming and outcoming edges
                edge_src = new_edges[etype][dir]['src'].long()
                edge_dst = new_edges[etype][dir]['dst'].long()
                del new_edges[etype][dir]['src'], new_edges[etype][dir]['dst']        
                self.dset['graph'].add_edges(edge_src, edge_dst, etype=etype)
    
    # Get budgeted ground truth for the round
    def get_round_budget(self, round_num):
        # POSITIVES
        all_new_positives = ((self.dset['graph'].ndata['creation_round'] >= 0) & (self.dset['graph'].ndata['label'] == 1)).nonzero().flatten().tolist()
        predicted_new_positives = torch.cat([self.rounds[i]['checks'][0] for i in list(range(round_num + 1))], 0).tolist()
        budget_new_positives = torch.cat([self.rounds[i]['budgets'][0] for i in list(range(round_num + 1))], 0).tolist()
        
        positive_budget_pool = list(set(all_new_positives) - set(predicted_new_positives) - set(budget_new_positives))
        round_budget = min([len(positive_budget_pool), self.train_config['round_budget_pos']])
        positive_budgets = torch.tensor(random.choice(positive_budget_pool, round_budget, replace=False))         
        
        # NEGATIVES
        base_negatives = ((self.dset['graph'].ndata['creation_round'] < 0) & (self.dset['graph'].ndata['label'] == 0)).nonzero().flatten().tolist()
        predicted_new_negatives = torch.cat([self.rounds[i]['checks'][1] for i in list(range(round_num + 1))], 0).tolist()

        negative_budget_pool = list(set(base_negatives).union(predicted_new_negatives))
        negative_budgets = torch.tensor(random.choice(negative_budget_pool, self.train_config['round_budget_neg'], replace=False))

        return positive_budgets, negative_budgets

    #######################
    ### MODEL FUNCTIONS ###
    #######################

    # Train procedure for conventional-variant models with a single fit call
    def model_train_classic(self):
        labels = self.dset['graph'].ndata['label']
        
        # Train
        _, _ = self.model.train(self.dset['graph'], self.train_config['ce_weight'])

        # Predictredict
        self.logits, _ = self.model(self.dset['graph'], self.dset['graph'].ndata['feature']) 
        probs = self.logits.softmax(1)
        f1, thres = get_best_f1(labels[self.dset['graph'].ndata['val_mask']], probs[self.dset['graph'].ndata['val_mask']])
        preds = np.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1

        # Eval
        _p, _cm  = eval_and_print(0, labels[self.dset['graph'].ndata['val_mask']], preds[self.dset['graph'].ndata['val_mask']], probs[self.dset['graph'].ndata['val_mask']][:, 1], f'Validation')
        trec, tpre, tmf1, tauc = _p
        tp, fp, tn, fn = _cm
        
        return trec, tpre, tmf1, tauc, tp, fp, tn, fn       
        
    # Train procedure for GNN-variant models with epochs
    def model_train_nn(self, num_epoch):
        features = self.dset['graph'].ndata['feature']
        labels = self.dset['graph'].ndata['label']
        train_mask = self.dset['graph'].ndata['train_mask']
        val_mask = self.dset['graph'].ndata['train_mask']
        
        # Initialize
        self.optimizer = self.train_config['optimizer'](self.model.parameters(), lr=self.train_config['learning_rate'])
        self.loss = self.train_config['loss']   
        best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.
        final_tp, final_fp, final_tn, final_fn = 0, 0, 0, 0

        # Misc parameters
        rl_idx = torch.nonzero(self.dset['graph'].ndata['train_mask'] & labels, as_tuple=False).squeeze(1)
        for e in range(num_epoch):
            
            # TRAIN
            self.model.train()
            self.logits = torch.zeros([len(labels), 2])

            if self.train_config['train_mode'] != 'batch':
                # Forward
                logits, loss = self.model(self.dset['graph'], features, **{'epoch': e})
                self.logits = logits

                if loss == None:
                    # No internal loss function returned by model, just normally calculate using logits and chosen loss
                    epoch_train_loss = self.loss(logits[train_mask], labels[train_mask], weight=torch.tensor([1., self.train_config['ce_weight']]))
                    epoch_val_loss = self.loss(logits[val_mask], labels[val_mask], weight=torch.tensor([1., self.train_config['ce_weight']]))
                else:
                    # Model returns some kind of loss
                    epoch_train_loss = loss
                    epoch_val_loss = loss

                # Backward
                self.optimizer.zero_grad()
                epoch_train_loss.backward()
                self.optimizer.step()

                # Additional stuff post backprop - CARE
                self.model.postBackprop(**{ 'graph': self.dset['graph'], 'epoch': e, 'rl_idx': rl_idx })           
            else:
                i = 0
                for input_nodes, output_nodes, blocks in self.dset['dataloader']:
                    i += 1

                    # Batch forward
                    blocks = [b.to(torch.device('cuda')) for b in blocks]
                    input_features = blocks[0].srcdata['feature']
                    output_labels = blocks[-1].dstdata['label']

                    logits, loss = self.model(blocks, input_features, **{'epoch': e})
                    self.logits[output_nodes] = logits.cpu()
                    epoch_train_loss = loss if loss != None else self.loss(logits, output_labels, weight=torch.tensor([1., self.train_config['ce_weight']]).to('cuda'))
                    epoch_val_loss = 0

                    # Batch backward
                    self.optimizer.zero_grad()
                    epoch_train_loss.backward()
                    self.optimizer.step()

                    # TODO: Postbackprop for batch
            
            # EVALUATE
            self.model.eval()

            # Predict
            probs = self.logits.softmax(1)
            f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
            preds = np.zeros_like(labels)
            preds[probs[:, 1] > thres] = 1

            # Eval
            _p, _cm  = eval_and_print(0, labels[val_mask], preds[val_mask], probs[val_mask][:, 1], f'Epoch {e}')
            trec, tpre, tmf1, tauc = _p
            tp, fp, tn, fn = _cm

            if best_f1 < f1:
                best_f1 = f1
                final_trec = trec
                final_tpre = tpre
                final_tmf1 = tmf1
                final_tauc = tauc
                final_tp = tp
                final_fp = fp
                final_tn = tn
                final_fn = fn
            
            # Print and log
            verPrint(self.verbose, 3, f'Epoch {e}, train loss: {epoch_train_loss:.4f}, val loss: {epoch_val_loss:.4f} val mf1: {f1:.4f}, (best {best_f1:.4f})')
            self.rounds[self.current_round]['log_train'].append((f'Round {self.current_round}', f'Epoch {e}', f'{epoch_train_loss:.4f}') + _p + _cm)

        return final_trec, final_tpre, final_tmf1, final_tauc, final_tp, final_fp, final_tn, final_fn       
        
    # Round training using additional data on round
    def model_round_train(self, round):
        verPrint(self.verbose, 1, 'Acquiring training data...')
        
        # Get training and validation split
        (idx_train, idx_valid, y_train, y_valid) = self.split_train_test(round, all_data=self.train_config['round_full_training'])
        if idx_train == None:
            verPrint(self.verbose, 1, 'Not enough data, continuing round without training')
            return
        else:
            verPrint(self.verbose, 1, f'Training set: {len(idx_train)} ({dict(Counter(y_train.tolist()))}) rows | Validation set:  {len(idx_valid)} ({dict(Counter(y_valid.tolist()))}) rows\n')

        # Initialize sampler in case of batch training
        if self.train_config['train_mode'] == 'batch':
            self.dset['sampler'] = dgl.dataloading.MultiLayerFullNeighborSampler(self.model_config['num_layers'])
            self.dset['dataloader'] = dgl.dataloading.DataLoader(
                self.dset['graph'], idx_train, self.dset['sampler'],
                batch_size=self.train_config['batch_size'], shuffle=True, drop_last=False, num_workers=self.train_config['num_workers']
            )
            
        self.dset['graph'] = MODEL_DICT[self.model_config['model_name']].prepare_graph(self.dset['graph']) # Additional graph preprocessing
        
        verPrint(self.verbose, 1, 'Starting training...')
        
        time_start = time.time()
        if self.model_config['model_name'] in ['XGB']:
            res = self.model_train_classic() 
        else:
            res = self.model_train_nn(self.train_config['num_epoch'] if round == 0 else self.train_config['num_round_epoch'])
        time_end = time.time()

        verPrint(self.verbose, 1, 'Ending training!')
        
        final_trec, final_tpre, final_tmf1, final_tauc, final_tp, final_fp, final_tn, final_fn = res        
        self.rounds[self.current_round]['log_eval'].append((f'round_{self.current_round}', f'val_set_best', (time_end - time_start), final_trec, final_tpre, final_tmf1, final_tauc, final_tp, final_fp, final_tn, final_fn))

        verPrint(self.verbose, 1, f'time cost: {str(time_end - time_start)} s')
        verPrint(self.verbose, 1, f'Best Val: REC {final_trec*100:.2f} PRE {final_tpre*100:.2f} MF1 {final_tmf1*100:.2f} AUC {final_tauc*100:.2f} TP {final_tp} FP {final_fp} TN {final_tn} FN {final_fn}')
        verPrint(self.verbose, 1, '=============')

    # Predict for entire dataset and return prediction; used to update round prediction data
    def model_round_predict(self):
        self.model.eval()
        
        labels = self.dset['graph'].ndata['label']
        logits, _ = self.model(self.dset['graph'], self.dset['graph'].ndata['feature'])
        probs = logits.softmax(1)
        _, thres = get_best_f1(labels, probs)
       
        preds = torch.zeros_like(self.dset['graph'].ndata['label'])
        preds[probs[:, 1] > thres] = 1
        
        # Model is provided false positives and true positives
        tp = ((labels == preds) & (labels == 1)).nonzero().flatten()
        fp = ((labels != preds) & (labels == 0)).nonzero().flatten()
        tn = ((labels == preds) & (labels == 0)).nonzero().flatten()
        fn = ((labels != preds) & (labels == 1)).nonzero().flatten()

        return preds, probs[:, 1], [tp, fp, tn, fn]
    
    ###########################
    ### ADVERSARY FUNCTIONS ###
    ###########################

    # TODO: Training phase for adversary
    def adversary_round_train(self):
        return

    # Adversarial generation of new fraud data
    def adversary_round_generate(self):
        return self.adver.generate(self.dset['graph'], n_instances=self.train_config['round_new_pos'], return_ids=True)
    
    # Generate negative instances
    def round_generate_negatives(self):
        return BaseAdversary.random_duplicate(self.dset['graph'], n_instances=self.train_config['round_new_neg'], label=0, return_ids=True)
   
    ###########################
    ### MAIN ROUND FUNCTION ### 
    ###########################

    # Execute 1 adver round based on the current state of the experiment
    def adver_round(self, round_num):
        verPrint(self.verbose, 1, f'\n=========\nSTARTING ROUND {round_num}!\n=========')
    
        # Reset all subsequent round data
        self.current_round = round_num
        self.rounds = self.rounds[:round_num]
        self.rounds.append({'budgets': [torch.tensor([]), torch.tensor([])], 'log_train': [], 'log_eval': []})
        self.dset['graph'] = dgl.remove_nodes(self.dset['graph'], (self.dset['graph'].ndata['creation_round'] >= max([round_num, 0])).nonzero().flatten())
        
        # Reset model if needed
        # TODO: If not return to model snapshot on previous step?
        if self.train_config['round_reset_model']:
            self.init_model()

        # Input new nodes to predict in the round
        if round_num == 0:
            # Set aside the initial inductive test data
            r0_num = round(self.dset['graph'].num_nodes() * self.train_config['initial_ratio'])
            r0_idx = torch.tensor(random.choice(list(range(self.dset['graph'].num_nodes())), r0_num, replace=False), dtype=torch.long)
  
            full_mask = torch.ones([self.dset['graph'].num_nodes()]).bool()
            full_mask[r0_idx] = False
            rest_idx = full_mask.nonzero()
            
            # Create subgraph from remaining nodes
            self.dset['temp_graph'] = self.dset['graph']
            self.dset['graph'] = dgl.remove_nodes(self.dset['temp_graph'], r0_idx, store_ids = False)
            isolateds = ((self.dset['graph'].in_degrees() == 0) & (self.dset['graph'].out_degrees() == 0)).nonzero().squeeze(1)
            self.dset['graph'] = dgl.remove_nodes(self.dset['graph'], isolateds, store_ids = False)
    
            # Train model using subgraph
            self.model_round_train(round_num) 

            # Restore to original graph
            self.dset['temp_graph'].ndata['train_mask'][rest_idx] = self.dset['graph'].ndata['train_mask'].unsqueeze(1)
            self.dset['temp_graph'].ndata['val_mask'][rest_idx] = self.dset['graph'].ndata['val_mask'].unsqueeze(1)
            self.dset['temp_graph'].ndata['test_mask'][rest_idx] = self.dset['graph'].ndata['test_mask'].unsqueeze(1)

            self.dset['graph'] = self.dset['temp_graph']
            self.dset['graph'].ndata['creation_round'][r0_idx] = 0
            self.dset['graph'].ndata['predicted'][r0_idx] = False

        else:
            self.model_round_train(round_num)  # Train model and adversary based on last round info
            # self.adversary_round_train(round) TODO: Train adversary

            # Adversarial positives
            verPrint(self.verbose, 2, f'Generating additional positive data from adversary...')
            new_adv_nodes, new_adv_edges, adv_seed, _ = self.adversary_round_generate()
            self.add_generated_data((new_adv_nodes, new_adv_edges))
            
            # Negatives
            verPrint(self.verbose, 2, f'Generating additional negative data by duplicating random nodes...')
            new_neg_nodes, new_neg_edges, neg_seed, _ = self.round_generate_negatives()
            self.add_generated_data((new_neg_nodes, new_neg_edges))

            self.rounds[round_num]['seeds_pos'] = adv_seed
            self.rounds[round_num]['seeds_neg'] = neg_seed

        # Predict and update prediction history
        self.rounds[round_num]['preds'], self.rounds[round_num]['probs'], self.rounds[round_num]['checks'] = self.model_round_predict()
        self.dset['graph'].ndata['predicted'][self.rounds[round_num]['checks'][0]] = self.dset['graph'].ndata['predicted'][self.rounds[round_num]['checks'][0]] | True
        self.dset['graph'].ndata['predicted'][self.rounds[round_num]['checks'][1]] = self.dset['graph'].ndata['predicted'][self.rounds[round_num]['checks'][1]] | True        
        
        # Get budget for next round
        verPrint(self.verbose, 2, f'Selecting data as budgeted ground truth for next round...')
        self.rounds[round_num]['budgets'] = self.get_round_budget(round_num)

        # Evaluate for logging
        self.adver_round_eval(round_num, adv_seed, neg_seed)
        
    # EVALUATE ON DIFFERENT SUBSETS OF DATA
    def adver_round_eval(self, round_num, adv_seed, neg_seed):
        labels = self.dset['graph'].ndata['label']

        verPrint(self.verbose, 1, 'PREDICTION RESULT - GRAPHS')
        _p, _cm = eval_and_print(self.verbose, labels, self.rounds[round_num]['preds'], self.rounds[round_num]['probs'], 'Dataset - Overall')
        self.rounds[round_num]['log_eval'].append((f'round_{round_num}', 'entire_graph', 0) + _p + _cm)

        train_mask = self.dset['graph'].ndata['train_mask']
        _p, _cm = eval_and_print(self.verbose, labels[train_mask], self.rounds[round_num]['preds'][train_mask], self.rounds[round_num]['probs'][train_mask], 'Dataset - Train')
        self.rounds[round_num]['log_eval'].append((f'round_{round_num}', 'train_set', 0) + _p + _cm)

        val_mask = self.dset['graph'].ndata['val_mask']
        _p, _cm = eval_and_print(self.verbose, labels[val_mask], self.rounds[round_num]['preds'][val_mask], self.rounds[round_num]['probs'][val_mask], 'Dataset - Val')
        self.rounds[round_num]['log_eval'].append((f'round_{round_num}', 'val_set', 0) + _p + _cm)

        for i in range(round_num + 1):
            i_round_mask = (self.dset['graph'].ndata['creation_round'] == i).nonzero().flatten()
            _p, _cm = eval_and_print(self.verbose, labels[i_round_mask], self.rounds[round_num]['preds'][i_round_mask], self.rounds[round_num]['probs'][i_round_mask], f'Dataset - Round {i}')
            self.rounds[round_num]['log_eval'].append((f'round_{round_num}', f'round_{i}_nodes', 0) + _p + _cm)

        if (round_num > 0) and (self.verbose >= 2):
            verPrint(self.verbose, 2, '---\nPREDICTION RESULT - SEEDS')
            _p, _cm = eval_and_print(self.verbose, labels[torch.cat([adv_seed, neg_seed], 0)], self.rounds[round_num]['preds'][torch.cat([adv_seed, neg_seed], 0)], self.rounds[round_num]['probs'][torch.cat([adv_seed, neg_seed], 0)], 'Seeds - Current')
            self.rounds[round_num]['log_eval'].append((f'round_{round_num}', f'seed_current_pred', 0) + _p + _cm)

            _p, _cm = eval_and_print(self.verbose, labels[torch.cat([adv_seed, neg_seed], 0)], self.rounds[round_num-1]['preds'][torch.cat([adv_seed, neg_seed], 0)], self.rounds[round_num]['probs'][torch.cat([adv_seed, neg_seed], 0)], 'Seeds - Prev')
            self.rounds[round_num]['log_eval'].append((f'round_{round_num}', f'seed_prev_pred', 0) + _p + _cm)
        
        pstatus = [f'{str(p[0])}-{str(p[1])}' for p in list(zip(labels.tolist(), self.dset['graph'].ndata['predicted'].tolist()))]
        verPrint(self.verbose, 1, f'PREDICTION STATUS - {dict(Counter(pstatus))}')