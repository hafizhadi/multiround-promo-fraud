import dgl
from dgl.nn import GraphConv

import torch
import torch.nn as nn

import time
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix

def train(g, model):
  features = g.ndata['feature']
  labels = g.ndata['label']
  index = list(range(len(labels)))

  # Train Test Split
  idx_train, idx_rest, y_train, y_rest = train_test_split(
      index, labels[index], stratify=labels[index],
      train_size = 0.8, random_state = 7, shuffle=True
  )
  idx_valid, idx_test, y_valid, y_test = train_test_split(
      idx_rest, y_rest, stratify=y_rest,
      test_size = 0.67, random_state = 7, shuffle=True
  )

  train_mask = torch.zeros([len(labels)]).bool()
  val_mask = torch.zeros([len(labels)]).bool()
  test_mask = torch.zeros([len(labels)]).bool()

  train_mask[idx_train] = 1
  val_mask[idx_valid] = 1
  test_mask[idx_test] = 1

  # Optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  # Inits
  best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.
  weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
  print('cross entropy weight: ', weight)

  # Main Loop
  time_start = time.time()
  for e in range(50):
      # TRAIN
      model.train()

      logits = model(g, features)
      loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # EVAL
      model.eval()

      probs = logits.softmax(1)
      f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
      preds = numpy.zeros_like(labels)
      preds[probs[:, 1] > thres] = 1

      trec = recall_score(labels[test_mask], preds[test_mask])
      tpre = precision_score(labels[test_mask], preds[test_mask])
      tmf1 = f1_score(labels[test_mask], preds[test_mask], average='macro')
      tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())

      if best_f1 < f1:
          best_f1 = f1
          final_trec = trec
          final_tpre = tpre
          final_tmf1 = tmf1
          final_tauc = tauc
      print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, loss, f1, best_f1))

  time_end = time.time()
  print('time cost: ', time_end - time_start, 's')
  print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec*100,
                                                                    final_tpre*100, final_tmf1*100, final_tauc*100))
  return final_tmf1, final_tauc