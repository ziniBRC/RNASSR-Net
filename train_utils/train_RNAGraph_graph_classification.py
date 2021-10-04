"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import dgl
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, roc_auc_score
from train_utils.metrics import accuracy_MNIST_CIFAR as accuracy

"""
    For GCNs
"""
def getNodeFeat(batch_graphs):
    node_feat = torch.tensor([])
    for grh in batch_graphs:
        node_feat = torch.cat([node_feat, grh.ndata['feat'].unsqueeze(0)], dim=0)

def plot_heatmap(input_array):
    f, ax = plt.subplots(figsize=(501))
    ax = sns.heatmap(input_array, vmin=0, vmax=1)

def matrix2seq(one_hot_matrices):
    d = {'A': torch.tensor([[1., 0., 0., 0.]]),
         'G': torch.tensor([[0., 1., 0., 0.]]),
         'C': torch.tensor([[0., 0., 1., 0.]]),
         'U': torch.tensor([[0., 0., 0., 1.]])}
    seq_list = []
    for i in range(one_hot_matrices.shape[0]):
        one_hot_matrice = one_hot_matrices[i, 0, :]
        seq = ""
        for loc in range(one_hot_matrice.shape[0]):
            if one_hot_matrice[loc, 0] == 1:
                seq += 'A'
            elif one_hot_matrice[loc, 1] == 1:
                seq += 'G'
            elif one_hot_matrice[loc, 2] == 1:
                seq += 'C'
            elif one_hot_matrice[loc, 3] == 1:
                seq += 'U'
            else:
                seq += 'N'
        seq_list.append(seq)

    return seq_list

def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    parts = 1
    for iter, (graphs, labels) in enumerate(data_loader):
        batch_size = labels.shape[0]
        for i in range(parts):
            batch_graphs = dgl.batch(graphs[i*batch_size//parts:(i+1)*batch_size//parts])
            batch_graphs.ndata['feat'] = batch_graphs.ndata['feat'].to(device)
            batch_graphs.edata['feat'] = batch_graphs.edata['feat'].to(device)
            batch_x = batch_graphs.ndata['feat'] # num x feat
            batch_e = batch_graphs.edata['feat']
            batch_labels = labels[i*batch_size//parts:(i+1)*batch_size//parts].to(device)
            optimizer.zero_grad()

            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            # batch_scores = model.forward(batch_graphs, batch_feature, batch_x, batch_e)

            loss = model.loss(batch_scores, batch_labels)
            loss.backward(retain_graph=True)
            optimizer.step()
            epoch_loss += loss.detach().item()
            epoch_train_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)*parts
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    epoch_auc = 0
    nb_data = 0
    batch_scores_array, batch_labels_array = np.array([]), np.array([])
    start_flag = 1
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = dgl.batch(batch_graphs)
            batch_graphs.ndata['feat'] = batch_graphs.ndata['feat'].to(device)
            batch_graphs.edata['feat'] = batch_graphs.edata['feat'].to(device)
            batch_x = batch_graphs.ndata['feat'] # num x feat
            batch_e = batch_graphs.edata['feat']
            batch_labels = batch_labels.to(device)

            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            # one_hot_matrices = model.sequence.detach().cpu().numpy()
            # sequence = matrix2seq(one_hot_matrices)
            # base_weight = model.base_weight.detach().cpu().numpy()
            # node_weight = model.node_weight.detach().cpu().numpy()
            # with open("base_weight.txt", "w") as f:
            #     for i in range(base_weight[0, :].shape[0]):
            #         f.write(str(base_weight[0, i]) + "\n")
            # with open("node_weight.txt", "w") as f:
            #     for i in range(node_weight[0, :].shape[0]):
            #         f.write(str(node_weight[0, i]) + "\n")

            # batch_scores = model.forward(batch_graphs, batch_feature, batch_x, batch_e)

            loss = model.loss(batch_scores, batch_labels)
            epoch_loss += loss.detach().item()
            epoch_acc += accuracy(batch_scores, batch_labels)

            batch_scores_numpy = F.softmax(batch_scores, dim=1).detach().cpu().numpy()
            batch_labels_numpy = batch_labels.cpu().numpy()
            if start_flag:
                batch_scores_array, batch_labels_array = batch_scores_numpy, batch_labels_numpy
                start_flag = 0
            else:
                batch_scores_array = np.vstack((batch_scores_array, batch_scores_numpy))
                batch_labels_array = np.hstack((batch_labels_array, batch_labels_numpy))

            nb_data += batch_labels.size(0)
        epoch_loss /= (iter + 1)
        epoch_acc /= nb_data

        epoch_auc = roc_auc_score(batch_labels_array, batch_scores_array[:, 1])

    return epoch_loss, epoch_acc, epoch_auc
