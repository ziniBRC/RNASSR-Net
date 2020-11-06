import torch
import math
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
from layers.gcn_layer import GCNLayer, ConvReadoutLayer, GNNPoolLayer, WeightCrossLayer
from layers.mlp_readout_layer import MLPReadout
from layers.conv_layer import ConvLayer, MAXPoolLayer


class GCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        self.device = net_params['device']
        self.n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = net_params['n_classes']
        self.pre_gnn, self.pre_cnn = None, None
        self.base_weight = None
        self.node_weight = None
        self.sequence = None
        self.filter_out = None

        window_size = 501
        conv_kernel1, conv_kernel2 = [9, 4], [9, 1]
        conv_padding, conv_stride = [conv_kernel1[0]//2, 0], 1
        pooling_kernel = [3, 1]
        pooling_padding, pooling_stride = [pooling_kernel[0]//2, 0], 2
        width_o1 = math.ceil((window_size - conv_kernel1[0] + 2 * conv_padding[0] + 1) / conv_stride)
        width_o1 = math.ceil((width_o1 - pooling_kernel[0] + 2 * pooling_padding[0] + 1) / pooling_stride)
        width_o2 = math.ceil((width_o1 - conv_kernel2[0] + 2 * conv_padding[0] + 1) / conv_stride)
        width_o2 = math.ceil((width_o2 - pooling_kernel[0] + 2 * pooling_padding[0] + 1) / pooling_stride)

        # GNN start
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers_gnn = nn.ModuleList()
        self.layers_gnn.append(GCNLayer(hidden_dim, hidden_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))
        for _ in range(self.n_layers * 2 - 2):
            self.layers_gnn.append(GCNLayer(hidden_dim, hidden_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))
        self.layers_gnn.append(GCNLayer(hidden_dim, out_dim, F.leaky_relu, dropout, self.batch_norm, self.residual))
        # GNN end

        # CNN start
        self.conv_readout_layer = ConvReadoutLayer(self.readout)
        self.layers_cnn = nn.ModuleList()
        self.layers_cnn.append(ConvLayer(1, 32, conv_kernel1, F.leaky_relu, self.batch_norm, residual=False, padding=conv_padding))
        for _ in range(self.n_layers - 1):
            self.layers_cnn.append(
                ConvLayer(32, 32, conv_kernel2, F.leaky_relu, self.batch_norm, residual=False, padding=conv_padding))

        self.layers_pool = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers_pool.append(MAXPoolLayer(pooling_kernel, stride=pooling_stride, padding=pooling_padding))
        # CNN end

        # self.cross_weight_layer = nn.ModuleList()
        # self.cross_weight_layer.append(WeightCrossLayer(in_dim=501, out_dim=501//2+1))
        # self.cross_weight_layer.append(WeightCrossLayer(in_dim=501, out_dim=501//4+1))
        self.batchnorm_weight = nn.BatchNorm1d(501)

        input_dim = width_o2*32
        self.MLP_layer = MLPReadout(501*32 + input_dim, self.n_classes)

    def forward(self, g, h, e):
        batch_size = len(g.batch_num_nodes)
        window_size = g.batch_num_nodes[0]
        similar_loss = 0
        cnn_node_weight = 0
        weight2gnn_list = []
        weight2cnn_list = []

        h2 = self._graph2feature(g)
        self.sequence = h2
        h2 = h2.to(self.device)

        h1 = self.embedding_h(h)
        h1 = self.in_feat_dropout(h1)
        # h2 = torch.unsqueeze(feature, dim=1)
        for i in range(self.n_layers):
            # GNN
            h1 = self.layers_gnn[2*i](g, h1)
            h1 = self.layers_gnn[2*i + 1](g, h1)
            # g, h1, _ = GNNPoolLayer(batch_size=batch_size, node_num=math.ceil(window_size / 2 ** i))(g, h1)

            # CNN
            h2 = self.layers_cnn[i](h2)
            if i == 0:
                self.filter_out = h2
                cnn_node_weight = torch.mean(h2, dim=1).squeeze(-1)
                self.base_weight = self.batchnorm_weight(cnn_node_weight)
                cnn_node_weight = torch.sigmoid(self.batchnorm_weight(cnn_node_weight))
                # cnn_node_weight = cnn_node_weight.detach()
            h2 = self.layers_pool[i](h2)

            # weight cross
            weight2gnn = torch.flatten(h2.squeeze(-1).permute(0, 2, 1), end_dim=1)
            weight2gnn_list.append(torch.mean(weight2gnn, dim=1).unsqueeze(-1))

            weight2cnn = torch.mean(self.conv_readout_layer(g, h1), dim=1).squeeze(-1)
            weight2cnn = self.batchnorm_weight(weight2cnn)
            weight2cnn_list.append(weight2cnn)

            # weight2cnn = self.cross_weight_layer[i](weight2cnn_list[-1].squeeze(-1))
            # h2 = torch.mul(h2, weight2cnn.unsqueeze(1).unsqueeze(-1))

        # similar_loss += torch.mean(torch.norm(cnn_node_weight - weight2cnn_list[-1], dim=1))
        # self.similar_loss = similar_loss

        g.ndata['h'] = h1

        hg = self.conv_readout_layer(g, h1)
        gnn_node_weight = torch.mean(hg, dim=1).squeeze(-1)
        self.node_weight = self.batchnorm_weight(gnn_node_weight)
        hg = torch.mul(hg, cnn_node_weight.unsqueeze(1).unsqueeze(-1))

        hg = torch.flatten(hg, start_dim=1)
        hc = torch.flatten(h2, start_dim=1)

        h_final = torch.cat([hg, hc], dim=1)
        pred = self.MLP_layer(h_final)

        return pred

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        # loss = criterion(self.pre_cnn, label) + criterion(self.pre_gnn, label)
        # loss += 0.01 * self.similar_loss
        return loss

    def _graph2feature(self, g):
        feat = g.ndata['feat']
        start, first_flag = 0, 0
        for batch_num in g.batch_num_nodes:
            if first_flag == 0:
                output = torch.transpose(feat[start:start + batch_num], 1, 0).unsqueeze(0)
                first_flag = 1
            else:
                output = torch.cat([output, torch.transpose(feat[start:start + batch_num], 1, 0).unsqueeze(0)], dim=0)
            start += batch_num
        output = torch.transpose(output, 1, 2)
        output = output.unsqueeze(1)
        return output

# self.similar_loss = torch.norm(
#     torch.mean(torch.mean(h2, dim=1).squeeze(-1), dim=0) -
#     torch.mean(weight2cnn_list[-1].squeeze(1).squeeze(-1), dim=0))