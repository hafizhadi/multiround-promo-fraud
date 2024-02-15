import torch
import numpy as np

from numpy import random
from collections import Counter
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
from torch.functional import F

### METHODS ###

## General
# Print but only if currently above specified verbosity level
def verPrint(verbose_status, verbose_threshold, msg):
    if verbose_status >= verbose_threshold:
        print(msg)

## Training related
def hinge_loss(labels, scores, margin=1):
    ls = labels * scores
    loss = F.relu(margin-ls)
    loss = loss.mean()
    return loss

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

def eval_and_print(verbose_level, labels, preds, probs, msg):
    rec = recall_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    auc = roc_auc_score(labels, probs.detach().numpy()) if torch.unique(labels).shape[0] > 1 else -1

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    print(labels)
    verPrint(verbose_level, 1, f'{msg}: REC {rec*100:.2f} PRE {prec*100:.2f} MF1 {f1*100:.2f} AUC {auc*100:.2f} TP {tp} FP {fp} TN {tn} FN {fn} | {dict(Counter(labels.tolist()))}')

    return (rec, prec, f1, auc)