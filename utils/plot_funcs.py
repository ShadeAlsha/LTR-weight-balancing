from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sklearn
from utils.eval_funcs import *

def plot_per_epoch_accuracy(trackRecords):
    train_acc = trackRecords['acc_train']
    test_acc = trackRecords['acc_test'] 
    
    plt.title("Training and validation accuracy per epoch")
    plt.plot(torch.Tensor(train_acc).cpu(), label='Train accuracy')
    plt.plot(torch.Tensor(test_acc).cpu(), label='Validation accuracy')

    plt.xlabel('training epochs')
    plt.ylabel('accuracy')
    plt.legend()

def plot_weights_evolution(trackRecords):
    # visualizing how norms of per-class weights change in the classifier while training
    W = np.concatenate(trackRecords['weightNorm'])
    W = W.reshape((-1, 100))

    plt.imshow(W, cmap= 'jet', vmin = 0, vmax=2)
    plt.colorbar()
    plt.xlabel('class ID sorted by cardinality')
    plt.ylabel('training epochs')
    plt.title('norms of per-class weights in the classifier')
    
def plot_norms(model, labelnames, y_range=None):
    # per-class weight norms vs. class cardinality
    W = model.encoder.fc.weight.cpu()
    tmp = torch.linalg.norm(W, ord=2, dim=1).detach().numpy()
    
    if y_range==None:
        max_val, mid_val, min_val = tmp.max(), tmp.mean(), tmp.min()
        c = min(1/mid_val, mid_val)
        y_range = [min_val-c, max_val+c]
    
    
    fig = plt.figure(figsize=(15,3), dpi=64, facecolor='w', edgecolor='k')
    plt.xticks(list(range(100)), labelnames, rotation=90, fontsize=8);  # Set text labels.
    ax1 = fig.add_subplot(111)

    ax1.set_ylabel('norm', fontsize=16)
    ax1.set_ylim(y_range)
    
    plt.plot(tmp, linewidth=2)
    plt.title('norms of per-class weights from the learned classifier vs. class cardinality', fontsize=20)


def plot_per_class_accuracy(models_dict, dataloaders, labelnames, img_num_per_cls, nClasses=100, device = 'cuda'):
    result_dict = {}
    for label in models_dict:
        model = models_dict[label]
        acc_per_class = get_per_class_acc(model, dataloaders, nClasses= nClasses, device= device)
        result_dict[label] = acc_per_class

    plt.figure(figsize=(15,4), dpi=64, facecolor='w', edgecolor='k')
    plt.xticks(list(range(100)), labelnames, rotation=90, fontsize=8);  # Set text labels.
    plt.title('per-class accuracy vs. per-class #images', fontsize=20)
    ax1 = plt.gca()    
    ax2=ax1.twinx()
    for label in result_dict:
        ax1.bar(list(range(100)), result_dict[label], alpha=0.7, width=1, label= label, edgecolor = "black")
        
    ax1.set_ylabel('accuracy', fontsize=16, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=16)

    ax2.set_ylabel('#images', fontsize=16, color='r')
    ax2.plot(img_num_per_cls, linewidth=4, color='r')
    ax2.tick_params(axis='y', labelcolor='r', labelsize=16)
    
    ax1.legend(prop={'size': 14})
