import dgl
import torch

import time, psutil, os
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from adversarial.adversarial import BaseAdversary

from utils_func import verPrint, get_best_f1, eval_and_print
from utils_const import model_dict, adversarial_dict

### MULTIROUND EXPERIMENT CLASS ###
class MultiroundExperiment(object):
    ## Class Methods
    def __init__(self, model_config, adver_config, train_config, graph):
        # Attributes
        model_config['train_mode'] = train_config['train_mode']
        adver_config['train_mode'] = train_config['train_mode']
        model_config['etypes'] = ['none'] if graph.is_homogeneous else graph.etypes

        self.verbose = train_config['verbose']
        self.dset = { 'graph': graph }
        self.model_config, self.adver_config, self.train_config  = model_config, adver_config, train_config

        # Model & Adversary
        self.init_model()
        self.init_adversarial()

        # Initialize round information
        self.current_round, self.rounds = 0, []
        self.dset['graph'].ndata['creation_round'] = torch.full([graph.num_nodes()], 0, dtype=torch.long)
    
    # Initialize model
    def init_model(self):
        in_dimension = self.dset['graph'].ndata['feature'].shape[1]
        class_num =  self.dset['graph'].ndata['label'].unique(return_counts=True)[0].shape[0]
        self.model = model_dict[self.model_config['model_name']](in_dimension, class_num, **self.model_config)
        if self.train_config['train_mode'] == 'batch':
            self.model.cuda()
    
    # Initialize adversarial
    def init_adversarial(self):
        self.adver = adversarial_dict[self.adver_config['adver_name']](**self.adver_config)

    # Split train test
    def split_train_test(self, round, all_data=True):
        
        labels = self.dset['graph'].ndata['label']

        self.dset['graph'].ndata['train_mask'] = torch.zeros([len(labels)]).bool()
        self.dset['graph'].ndata['val_mask'] = torch.zeros([len(labels)]).bool()
        self.dset['graph'].ndata['test_mask'] = torch.zeros([len(labels)]).bool()

        if round > 0:
            initial_pool = (self.dset['graph'].ndata['creation_round'] == 0).nonzero().flatten() if all_data else torch.tensor([], dtype=torch.long)
            positive_preds = torch.cat([torch.cat(self.rounds[i]['checks'][:2], 0) for i in (list(range(round)) if all_data else [round-1])], 0)
            full_pool = torch.cat([initial_pool, positive_preds], 0)
        else:
            full_pool = torch.arange(len(labels), dtype=torch.long)
            
        index = torch.arange(len(labels), dtype=torch.long)[full_pool]
        nonindex = torch.ones_like(labels, dtype=bool)
        nonindex[full_pool] = False

        # Train Test Split
        if (torch.sum(labels[index] == 0) < 2) or (torch.sum(labels[index] == 1) < 2):
            return None
        idx_train, idx_rest, y_train, y_rest = train_test_split(
            index, labels[index], stratify=labels[index],
            train_size = self.train_config['train_ratio'], random_state = self.train_config['random_state'], shuffle=True
        )

        if (torch.sum(y_rest == 0) < 2) or (torch.sum(y_rest == 1) < 2):
            return None
        idx_valid, idx_test, y_valid, y_test = train_test_split(
            idx_rest, y_rest, stratify=y_rest,
            test_size = self.train_config['test_ratio_from_rest'], random_state = self.train_config['random_state'], shuffle=True
        )

        self.dset['graph'].ndata['train_mask'][idx_train] = 1
        self.dset['graph'].ndata['val_mask'][idx_valid] = 1
        self.dset['graph'].ndata['test_mask'][idx_test] = 1
        self.dset['graph'].ndata['test_mask'][nonindex] = 1

        self.train_config['ce_weight'] = (1-labels[self.dset['graph'].ndata['train_mask']]).sum().item() / labels[self.dset['graph'].ndata['train_mask']].sum().item()

        return idx_train, idx_valid, idx_test, y_train, y_valid, y_test

    # Train model normally on entire dataset
    def model_train(self):
        # Inits
        best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.
        self.optimizer = self.train_config['optimizer'](self.model.parameters(), lr=self.train_config['learning_rate'])
        self.loss = self.train_config['loss']

        features = self.dset['graph'].ndata['feature']
        labels = self.dset['graph'].ndata['label']

        # Main Training Loop
        time_start = time.time()
        verPrint(self.verbose, 1, 'Starting training...')

        # Various out of loop variables
        rl_idx = torch.nonzero(self.dset['graph'].ndata['train_mask'] & labels, as_tuple=False).squeeze(1)

        for e in range(self.train_config['num_epoch']):
            self.model.train()
            self.logits = torch.zeros([len(labels), 2])

            if self.train_config['train_mode'] != 'batch':
                # Forward pass
                verPrint(self.verbose, 2, 'Forward')
                logits, loss = self.model(self.dset['graph'], features **{'epoch': e})
                self.logits = logits
                epoch_loss = loss if loss != None else self.loss(logits[self.dset['graph'].ndata['train_mask']], labels[self.dset['graph'].ndata['train_mask']], weight=torch.tensor([1., self.train_config['ce_weight']]))

                # Backward pass
                verPrint(self.verbose, 2, 'Backward')
                self.optimizer.zero_grad()
                epoch_loss.backward()
                self.optimizer.step()

                # Additional stuff post backprop
                self.model.postBackprop(**{ 'graph': self.dset['graph'], 'epoch': e, rl_idx: rl_idx })           
            else:
                i = 0
                for input_nodes, output_nodes, blocks in self.dset['dataloader']:
                    i += 1

                    # Batch forward
                    verPrint(self.verbose, 2, f'Forward for batch {i}')
                    blocks = [b.to(torch.device('cuda')) for b in blocks]
                    input_features = blocks[0].srcdata['feature']
                    output_labels = blocks[-1].dstdata['label']

                    logits, loss = self.model(blocks, input_features, **{'epoch': e})
                    self.logits[output_nodes] = logits.cpu()
                    epoch_loss = loss if loss != None else self.loss(logits, output_labels, weight=torch.tensor([1., self.train_config['ce_weight']]).to(torch.device('cuda')))

                    # Backward pass
                    verPrint(self.verbose, 2, 'Backward')
                    self.optimizer.zero_grad()
                    epoch_loss.backward()
                    self.optimizer.step()

                    # The following code is used to record the memory usage
                    py_process = psutil.Process(os.getpid())
                    verPrint(self.verbose, 3, f'CPU Memory Usage: {py_process.memory_info().rss / (1024 ** 3)} GB')
                    verPrint(self.verbose, 3, f'GPU Memory Usage: {torch.cuda.memory_reserved() / (1024 ** 3)} GB')

                    # TODO: Postbackprop for batch

            
            # Evaluate
            verPrint(self.verbose, 2, 'Evaluate')
            self.model.eval()
            probs = self.logits.softmax(1)

            f1, thres = get_best_f1(labels[self.dset['graph'].ndata['val_mask']], probs[self.dset['graph'].ndata['val_mask']])
            preds = np.zeros_like(labels)
            preds[probs[:, 1] > thres] = 1

            trec, tpre, tmf1, tauc = eval_and_print(0, labels[self.dset['graph'].ndata['test_mask']], preds[self.dset['graph'].ndata['test_mask']], probs[self.dset['graph'].ndata['test_mask']][:, 1], f'Epoch {e}')
            if best_f1 < f1:
                best_f1 = f1
                final_trec = trec
                final_tpre = tpre
                final_tmf1 = tmf1
                final_tauc = tauc
            verPrint(self.verbose, 1, 'Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, epoch_loss, f1, best_f1))

        time_end = time.time()
        verPrint(self.verbose, 2, f'time cost: {str(time_end - time_start)} s')
        verPrint(self.verbose, 1, 'Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec*100, final_tpre*100, final_tmf1*100, final_tauc*100))
        verPrint(self.verbose, 1, 'Ending training!\n=========')
        return final_tmf1, final_tauc
    
    # Round training using additional data on round
    def model_round_train(self, round):
        # Prepare training data for the round
        split_res = self.split_train_test(round, all_data=self.train_config['round_all_data'])
        if split_res == None:
            verPrint(self.verbose, 1, 'No additional dataset to train with!')
            return

        (idx_train, idx_valid, idx_test, y_train, y_valid, y_test) = split_res

        # Initialize sampler in catch of batch training
        if self.train_config['train_mode'] == 'batch':
            self.dset['sampler'] = dgl.dataloading.MultiLayerFullNeighborSampler(self.model_config['num_layers'])
            self.dset['dataloader'] = dgl.dataloading.DataLoader(
                self.dset['graph'], idx_train, self.dset['sampler'],
                batch_size=self.train_config['batch_size'], shuffle=True, drop_last=False, num_workers=self.train_config['num_workers']
        )
            
        # Reset model if needed
        # TODO: If not return to model snapshot on previous step?
        if self.train_config['round_reset_model']:
            self.init_model()

        # Prepare graph for model training
        self.dset['graph'] = model_dict[self.model_config['model_name']].prepare_graph(self.dset['graph'])

        # Train
        self.model_train()

    # Round prediction on round
    def model_round_predict(self):
        self.model.eval()
        
        labels = self.dset['graph'].ndata['label']
        logits, loss = self.model(self.dset['graph'], self.dset['graph'].ndata['feature'])
        probs = logits.softmax(1)
        f1, thres = get_best_f1(labels, probs)
       
        preds = torch.zeros_like(self.dset['graph'].ndata['label'])
        preds[probs[:, 1] > thres] = 1
        
        # Model is provided false positives and true positives
        tp = ((labels == preds) & (labels == 1)).nonzero().flatten()
        fp = ((labels != preds) & (labels == 0)).nonzero().flatten()
        tn = ((labels == preds) & (labels == 0)).nonzero().flatten()
        fn = ((labels != preds) & (labels == 1)).nonzero().flatten()

        return preds, probs[:, 1], [tp, fp, tn, fn]
    
    # adver training using provided data
    def adversary_round_train(self):
        return

    # adver generation of new positive instances
    def adversary_round_generate(self):
        return self.adver.generate(self.dset['graph'], n_instances=self.train_config['round_pos_count'], return_ids=True)
    
    # Generate negative instances
    def round_generate_negatives(self):
        return BaseAdversary.random_duplicate(self.dset['graph'], n_instances=self.train_config['round_neg_count'], label=0, return_ids=True)
    
    def add_generated_data(self, data):
        new_nodes, new_edges = data

        # Add nodes
        new_nodes['creation_round'] = torch.full([len(new_nodes['label'])], self.current_round)
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
    
    # Execute 1 adver round based on the current state of the experiment
    def adver_round(self, round):
        verPrint(self.verbose, 1, f'Starting round {round}!\n=========')

        # Initialization and check to see if inputted round number is valid (i.e. the previous round has been conducted)
        if(len(self.rounds) < round):
            return
    
        # Reset all subsequent round data
        self.current_round = round
        self.rounds = self.rounds[:round]
        self.rounds.append({})
        self.dset['graph'] = dgl.remove_nodes(self.dset['graph'], (self.dset['graph'].ndata['creation_round'] >= max([round, 1])).nonzero().flatten())
        
        # Generate additional adversarial data for round
        if round > 0:
            
            # Train model and adversary based on last round info
            # TODO: self.adversary_round_train(round)

            # Generate additional data for round
            verPrint(self.verbose, 2, f'Generating additional positive data from adversary...')
            new_adv_nodes, new_adv_edges, adv_seed, _ = self.adversary_round_generate()
            self.add_generated_data((new_adv_nodes, new_adv_edges))
            
            verPrint(self.verbose, 2, f'Generating additional negative data by duplicating random nodes...')
            new_neg_nodes, new_neg_edges, neg_seed, _ = self.round_generate_negatives()
            self.add_generated_data((new_neg_nodes, new_neg_edges))

        self.model_round_train(round)
        self.rounds[round]['preds'], self.rounds[round]['probs'], self.rounds[round]['checks'] = self.model_round_predict()
        round_mask = (self.dset['graph'].ndata['creation_round'] == round).nonzero()
        labels = self.dset['graph'].ndata['label']
        if round > 0:
            _ = eval_and_print(self.verbose, labels[round_mask], self.rounds[round]['preds'][round_mask], self.rounds[round]['probs'][round_mask], 'Round')
            _ = eval_and_print(self.verbose, labels[torch.cat([adv_seed, neg_seed], 0)], self.rounds[round]['preds'][torch.cat([adv_seed, neg_seed], 0)], self.rounds[round]['probs'][torch.cat([adv_seed, neg_seed], 0)], 'Round - Seeds')
            _ = eval_and_print(self.verbose, labels[torch.cat([adv_seed, neg_seed], 0)], self.rounds[round-1]['preds'][torch.cat([adv_seed, neg_seed], 0)], self.rounds[round]['probs'][torch.cat([adv_seed, neg_seed], 0)], 'Prev round - Seeds')
        else:
            verPrint(self.verbose, 1, 'No prediction this round.')

        _ = eval_and_print(self.verbose, labels, self.rounds[round]['preds'], self.rounds[round]['probs'], 'Overall')
