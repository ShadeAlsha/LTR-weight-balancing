from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import sklearn
from utils.dataset_CIFAR100LT import *
    
def print_accuracy(model, dataloaders, new_labelList, device = 'cup', test_aug = True):
    model.eval()
    
    if test_aug:
        model = horizontal_flip_aug(model)

    predList = np.array([])
    grndList = np.array([])
    for sample in dataloaders['test']:
        with torch.no_grad():
            images, labels = sample
            images = images.to(device)
            labels = labels.type(torch.long).view(-1).numpy()
            logits = model(images)
            softmaxScores = F.softmax(logits, dim=1)   

            predLabels = softmaxScores.argmax(dim=1).detach().squeeze().cpu().numpy()
            predList = np.concatenate((predList, predLabels))    
            grndList = np.concatenate((grndList, labels))

    
    confMat = sklearn.metrics.confusion_matrix(grndList, predList)

    # normalize the confusion matrix
    a = confMat.sum(axis=1).reshape((-1,1))
    confMat = confMat / a

    acc_avgClass = 0
    for i in range(confMat.shape[0]):
        acc_avgClass += confMat[i,i]

    acc_avgClass /= confMat.shape[0]
    print('acc avgClass: ', "{:.1%}".format(acc_avgClass))

    breakdownResults = shot_acc(predList, grndList, np.array(new_labelList), many_shot_thr=100, low_shot_thr=20, acc_per_cls=False)
    print('Many:', "{:.1%}".format(breakdownResults[0]), 'Medium:', "{:.1%}".format(breakdownResults[1]), 'Few:', "{:.1%}".format(breakdownResults[2]))


def horizontal_flip_aug(model):
    def aug_model(data):
        logits = model(data)
        h_logits = model(data.flip(3))
        return (logits+h_logits)/2
    return aug_model

def mic_acc_cal(preds, labels):
    # This function is excerpted from a publicly available code [commit 01e52ed, BSD 3-Clause License]
    # https://github.com/zhmiao/OpenLongTailRecognition-OLTR/blob/master/utils.py
    if isinstance(labels, tuple):
        assert len(labels) == 3
        targets_a, targets_b, lam = labels
        acc_mic_top1 = (lam * preds.eq(targets_a.data).cpu().sum().float() \
                       + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) / len(preds)
    else:
        acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1


def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    # This function is excerpted from a publicly available code [commit 01e52ed, BSD 3-Clause License]
    # https://github.com/zhmiao/OpenLongTailRecognition-OLTR/blob/master/utils.py
    
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))    
 
    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)] 
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)
    
def get_per_class_acc(model, dataloaders, nClasses= 100, device = 'cpu'):
    predList = np.array([])
    grndList = np.array([])
    model.eval()
    for sample in dataloaders['test']:
        with torch.no_grad():
            images, labels = sample
            images = images.to(device)
            labels = labels.type(torch.long).view(-1).numpy()
            logits = model(images)
            softmaxScores = F.softmax(logits, dim=1)   

            predLabels = softmaxScores.argmax(dim=1).detach().squeeze().cpu().numpy()
            predList = np.concatenate((predList, predLabels))    
            grndList = np.concatenate((grndList, labels))


    confMat = sklearn.metrics.confusion_matrix(grndList, predList)

    # normalize the confusion matrix
    a = confMat.sum(axis=1).reshape((-1,1))
    confMat = confMat / a

    acc_avgClass = 0
    for i in range(confMat.shape[0]):
        acc_avgClass += confMat[i,i]

    acc_avgClass /= confMat.shape[0]
    
    acc_per_class = [0] * nClasses

    for i in range(nClasses):
        acc_per_class[i] = confMat[i,i]
    
    return acc_per_class   
    
def createMontage(imList, dims, times2rot90=0):
    '''
    imList isi N x HxWx3
    making a montage function to assemble a set of images as a single image to display
    '''
    imy, imx, k = dims
    rows = round(math.sqrt(k))
    cols = math.ceil(k/rows)
    imMontage = np.zeros((imy*rows, imx*cols, 3))
    idx = 0
    
    y = 0
    x = 0
    for idx in range(k):
        imMontage[y*imy:(y+1)*imy, x*imx:(x+1)*imx, :] = imList[idx, :,:,:] #np.rot90(imList[:,:,idx],times2rot90)
        if (x+1)*imx >= imMontage.shape[1]:
            x = 0
            y += 1
        else:
            x+=1
    return imMontage
