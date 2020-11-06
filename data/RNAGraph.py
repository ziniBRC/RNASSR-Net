import os
import pickle
from scipy.spatial.distance import cdist
import numpy as np
import multiprocessing as mp
import itertools
import utils
from utils.general_utils import Pool
from utils.rna_utils import load_mat, load_seq

import dgl
import torch
import torch.utils.data
from tqdm import tqdm
import time

import csv
from sklearn.model_selection import StratifiedShuffleSplit

class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])


class RNAGraphDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, dataset, split, p=None, debias="False", **kwargs):

        self.split = split
        self.debias = debias

        self.graph_lists = []
        self.graph_labels = []
        self.seq_list = []
        self.label_list = []
        self.id_list = []

        if p is None:
            pool = Pool(min(int(mp.cpu_count() * 2 / 3), 12))
        else:
            pool = p

        probabilistic = kwargs.get('probabilistic', True)
        # nucleotide_label = kwargs.get('nucleotide_label', False)

        print("read data")
        path_template = os.path.join(data_dir, 'GraphProt_CLIP_sequences', '{}', '{}', '{}', 'data.fa')
        if self.debias == 'True':
            pos_id, pos_seq, neg_id, neg_seq = self.data_debias(data_dir, dataset, split)
        else:
            pos_id, pos_seq = load_seq(path_template.format(dataset, split, 'positives'))
            neg_id, neg_seq = load_seq(path_template.format(dataset, split, 'negatives'))

        self.all_id = pos_id + neg_id
        self.seq_list = pos_seq + neg_seq
        self.label_list = np.array([1] * len(pos_id) + [0] * (len(neg_id)))
        self.graph_labels = torch.tensor(self.label_list)

        print("convert seq to graph")
        pos_matrix = load_mat(path_template.format(dataset, split, 'positives')
                                              , pool, load_dense=False, **kwargs)
        neg_matrix = load_mat(path_template.format(dataset, split, 'negatives')
                                              , pool, load_dense=False, **kwargs)

        pos_adjacency_matrix, pos_probability_matrix = pos_matrix
        neg_adjacency_matrix, neg_probability_matrix = neg_matrix
        adjacency_matrix = np.concatenate([pos_probability_matrix, neg_probability_matrix], axis=0)

        print("convert graph to dglgraph")
        self.graph_lists = self._convert2dglgraph(self.seq_list, adjacency_matrix)

        # data_path = os.path.join(data_dir, dataset + '_graphs_' + split + '.pkl')
        # with open(data_path, "rb") as f:
        #     f = pickle.load(f)
        #     self.graph_lists = f[0]
        #     self.graph_labels = f[1]
        self.n_samples = len(self.graph_lists)
        print("prepare data")
        self._prepare()

    def _prepare(self):
        # self.Adj_matrices, self.node_features, self.edges_lists, self.edge_features = [], [], [], []
        window_size = 501
        for graph in tqdm(self.graph_lists):
            graph.add_nodes(window_size-graph.batch_num_nodes[0])
            # adj = graph.adjacency_matrix().to_dense().numpy()
            # self.Adj_matrices.append(graph.adjacency_matrix())
            # self.node_features.append(graph.ndata['feat'].numpy())
            # self.edges_lists = [[] for i in range(graph.ndata['feat'].shape[0])]
            # edges = graph.all_edges()
            # for i in range(len(edges[0])):
            #     self.edges_lists[edges[1][i].item()].append(edges[0][i].item())
            # self.edge_features.append(np.ones(len(edges[0])))
            # graph.edata['feat'] = torch.ones(len(edges[0])).unsqueeze(1)

    def data_debias(self, basedir, rbp, split):
        path_template = os.path.join(basedir, 'GraphProt_CLIP_sequences', '{}', '{}', '{}', 'data.fa')
        dataset = {}

        pos_id, pos_seq = load_seq(path_template.format(rbp, split, 'positives'))
        neg_id, neg_seq = load_seq(path_template.format(rbp, split, 'negatives'))
        all_id = pos_id + neg_id
        all_seq = pos_seq + neg_seq

        size_pos = len(pos_id)
        # nucleotide level label
        all_label = []
        for i, seq in enumerate(all_seq):
            if i < size_pos:
                all_label.append((np.array(list(seq)) <= 'Z').astype(np.int32))
            else:
                all_label.append(np.array([0] * len(seq)))
        dataset['label'] = np.array(all_label)

        path_template = path_template[:-7] + 'modified_data.fa'
        if not os.path.exists(path_template.format(rbp, 'train', 'positives')) or \
                not os.path.exists(path_template.format(rbp, 'train', 'negatives')):
            all_modified_seq = []
            for seq, label in zip(all_seq, dataset['label']):
                seq = list(seq)
                if np.max(label) == 1:
                    pos_idx = np.where(np.array(label) == 1)[0]
                else:
                    pos_idx = np.where((np.array(seq) <= 'Z').astype(np.int32) == 1)[0]
                # overkill for PARCLIP_ELAVL1A, CLIPSEQ_AGO2, CLIPSEQ_ELAVL1
                if rbp in ['CAPRIN1_Baltz2012', 'PARCLIP_IGF2BP123', 'PARCLIP_MOV10_Sievers', 'ZC3H7B_Baltz2012',
                           'C22ORF28_Baltz2012', 'PARCLIP_ELAVL1A', 'PARCLIP_TAF15', 'PARCLIP_FUS', 'PARCLIP_EWSR1',
                           'PARCLIP_HUR', 'PARCLIP_PUM2', 'PARCLIP_AGO1234', 'ALKBH5_Baltz2012',
                           'C17ORF85_Baltz2012', 'PARCLIP_QKI', 'PARCLIP_ELAVL1', 'CLIPSEQ_SFRS1', 'CLIPSEQ_AGO2',
                           'CLIPSEQ_ELAVL1']:
                    indices = [pos_idx[-1] - 1, pos_idx[-1], pos_idx[-1] + 1, pos_idx[0], pos_idx[0] - 1,
                               pos_idx[0] - 2]
                elif rbp in ['ICLIP_HNRNPC', 'ICLIP_TDP43', 'ICLIP_TIA1', 'ICLIP_TIAL1', 'PTBv1']:
                    raise ValueError('Warning, %s is not biased...' % (rbp))
                else:
                    raise ValueError('unrecognized %s for the modify_leaks option' % (rbp))

                for idx in indices:
                    try:
                        if idx in [pos_idx[-1] - 1, pos_idx[-1], pos_idx[0]]:
                            seq[idx] = np.random.choice(['A', 'C', 'G', 'U'])
                        else:
                            seq[idx] = np.random.choice(['a', 'c', 'g', 'u'])
                    except IndexError:
                        pass
                all_modified_seq.append(''.join(seq))

            all_seq = all_modified_seq
            pos_seq = all_seq[:len(pos_id)]
            neg_seq = all_seq[len(pos_id):]

            # cache temporary modified files
            # with open(path_template.format(rbp, 'train', 'positives'), 'w') as file:
            #     for id, seq in zip(pos_id, pos_seq):
            #         file.write('%s\n%s\n' % (id, seq))
            # with open(path_template.format(rbp, 'train', 'negatives'), 'w') as file:
            #     for id, seq in zip(neg_id, neg_seq):
            #         file.write('%s\n%s\n' % (id, seq))
            # print('modified sequences have been cached')
        return pos_id, pos_seq, neg_id, neg_seq

    def _convert2dglgraph(self, seq_list, csr_matrixs):
        dgl_graph_list = []
        for i in tqdm(range(len(csr_matrixs))):
            dgl_graph_list.append(self._constructGraph(seq_list[i], csr_matrixs[i]))

        return dgl_graph_list

    def _constructGraph(self, seq, csr_matrix):
        seq_upper = seq.upper()
        d = {'A': torch.tensor([[1., 0., 0., 0.]]),
             'G': torch.tensor([[0., 1., 0., 0.]]),
             'C': torch.tensor([[0., 0., 1., 0.]]),
             'U': torch.tensor([[0., 0., 0., 1.]]),
             'T': torch.tensor([[0., 0., 0., 1.]])}

        grh = dgl.DGLGraph(csr_matrix)

        grh.ndata['feat'] = torch.zeros((grh.number_of_nodes(), 4))

        for i in range(len(seq)):
            grh.ndata['feat'][i] = d[seq_upper[i]]

        grh.edata['feat'] = csr_matrix.data
        grh.edata['feat'] = grh.edata['feat'].unsqueeze(1)

        for i in range(len(seq)):
            for j in range(-2, 3):
                if j == 0 or j == -1 or j == 1:
                    continue
                if i + j < 0 or i + j > len(seq) - 1:
                    continue
                if grh.has_edge_between(i, i + j):
                    continue
                grh.add_edges(i, i + j)
                grh.edges[i, i + j].data['feat'] = torch.tensor([[1/j]], dtype=torch.float64)

        return grh

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]

    def __setitem__(self, idx, v):
        self.graph_lists[idx], self.graph_labels[idx] = v
        self.labels[idx] = v[1].item()
        pass


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']


        This function is called inside a function in SuperPixDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


class RNADataset(torch.utils.data.Dataset):
    def __init__(self, name, config, window_size=501):
        """
            Loading Superpixels datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        if config['debias'] == "True":
            print("data debiased!")
            data_dir = 'data/GraphProt_CLIP_sequences/RNAGraphProb_debias/'
        else:
            print("data biased!")
            data_dir = 'data/GraphProt_CLIP_sequences/RNAGraphProb/'
        # data_dir = 'data/RNAGraph/'
        with open(data_dir + name + '.pkl', "rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]

        num_val = len(self.val.graph_lists)
        all_train_graphs = self.train.graph_lists + self.val.graph_lists
        all_train_labels = torch.cat([self.train.graph_labels, self.val.graph_labels], dim=0)

        inds = np.random.permutation(np.arange(0, int(len(all_train_graphs))))

        print("val data shuffle")
        _val_graphs = [all_train_graphs[ind] for ind in inds[:num_val]]
        # _val_sequences = self._construct_sequence_features(_val_graphs, window_size)
        _val_labels = torch.tensor([all_train_labels[ind] for ind in inds[:num_val]])

        print("train data shuffle")
        _train_graphs = [all_train_graphs[ind] for ind in inds[num_val:]]
        # _train_sequences = self._construct_sequence_features(_train_graphs, window_size)
        _train_labels = torch.tensor([all_train_labels[ind] for ind in inds[num_val:]])

        self.train = DGLFormDataset(_train_graphs, _train_labels)
        self.val = DGLFormDataset(_val_graphs, _val_labels)

        print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        if len(samples[0]) == 3:
            graphs, seqs, labels = map(list, zip(*samples))
        else:
            graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        # features = torch.tensor([])
        # for seq in seqs:
        #     feature = seq.unsqueeze(0)
        #     features = torch.cat([features, feature], dim=0)
        for idx, graph in enumerate(graphs):
            graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
            graphs[idx].edata['feat'] = graph.edata['feat'].float()

        return graphs, labels

    # prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
    def collate_dense_gnn(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        # tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        # tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        # snorm_n = tab_snorm_n[0][0].sqrt()

        # batched_graph = dgl.batch(graphs)

        g = graphs[0]
        adj = self._sym_normalize_adj(g.adjacency_matrix().to_dense())
        """
            Adapted from https://github.com/leichen2018/Ring-GNN/
            Assigning node and edge feats::
            we have the adjacency matrix in R^{n x n}, the node features in R^{d_n} and edge features R^{d_e}.
            Then we build a zero-initialized tensor, say T, in R^{(1 + d_n + d_e) x n x n}. T[0, :, :] is the adjacency matrix.
            The diagonal T[1:1+d_n, i, i], i = 0 to n-1, store the node feature of node i. 
            The off diagonal T[1+d_n:, i, j] store edge features of edge(i, j).
        """

        zero_adj = torch.zeros_like(adj)

        in_dim = g.ndata['feat'].shape[1]

        # use node feats to prepare adj
        adj_node_feat = torch.stack([zero_adj for j in range(in_dim)])
        adj_node_feat = torch.cat([adj.unsqueeze(0), adj_node_feat], dim=0)

        for node, node_feat in enumerate(g.ndata['feat']):
            adj_node_feat[1:, node, node] = node_feat

        x_node_feat = adj_node_feat.unsqueeze(0)

        return x_node_feat, labels

    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim=0)  # .squeeze()
        deg_inv = torch.where(deg > 0, 1. / torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))

    def _add_self_loops(self):

        # function for adding self loops
        # this function will be called only if self_loop flag is True

        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

        self.train = DGLFormDataset(self.train.graph_lists, self.train.graph_labels)
        self.val = DGLFormDataset(self.val.graph_lists, self.val.graph_labels)
        self.test = DGLFormDataset(self.test.graph_lists, self.test.graph_labels)


class RNAGraphDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name, num_val=0.1, window_size=501, debias='False'):
        """
            Takes input standard image dataset name (MNIST/CIFAR10)
            and returns the superpixels graph.

            This class uses results from the above SuperPix class.
            which contains the steps for the generation of the Superpixels
            graph from a superpixel .pkl file that has been given by
            https://github.com/bknyaz/graph_attention_pool

            Please refer the SuperPix class for details.
        """
        t_data = time.time()
        self.name = name

        print("processing test data")
        self.test = RNAGraphDGL("./data/", dataset=self.name, split='ls',
                                fold_algo='rnaplfold', debias=debias, probabilistic=True, )

        print("processing train data")
        self.train_ = RNAGraphDGL("./data/", dataset=self.name, split='train',
                                  fold_algo='rnaplfold', debias=debias, probabilistic=True)

        inds = np.random.permutation(np.arange(0, int(len(self.train_))))

        print("val data shuffle")
        _val_graphs = [self.train_.graph_lists[ind] for ind in inds[:int(len(self.train_)*num_val)]]
        # _val_sequences = self._construct_sequence_features(_val_graphs, window_size)
        _val_labels = torch.tensor([self.train_.graph_labels[ind] for ind in inds[:int(len(self.train_)*num_val)]])

        print("train data shuffle")
        _train_graphs = [self.train_.graph_lists[ind] for ind in inds[int(len(self.train_)*num_val):]]
        # _train_sequences = self._construct_sequence_features(_train_graphs, window_size)
        _train_labels = torch.tensor([self.train_.graph_labels[ind] for ind in inds[int(len(self.train_)*num_val):]])

        print("test data shuffle")
        inds = np.random.permutation(np.arange(0, int(len(self.test))))
        _test_graphs = [self.test.graph_lists[ind] for ind in inds]
        # _test_sequences = self._construct_sequence_features(_test_graphs, window_size)
        _test_labels = torch.tensor([self.test.graph_labels[ind] for ind in inds])

        # _val_graphs, _val_labels = self.train_[:int(len(self.train_)*num_val)]
        # _train_graphs, _train_labels = self.train_[int(len(self.train_)*num_val):]

        print("data convert to DGLFormDataset")
        self.train = DGLFormDataset(_train_graphs, _train_labels)
        self.val = DGLFormDataset(_val_graphs, _val_labels)
        self.test = DGLFormDataset(_test_graphs, _test_labels)

        print("[I] Data load time: {:.4f}s".format(time.time() - t_data))

    def _feature_padding(self, feature, window_size):
        # pre_padding_len = 3
        #
        # pre_padding = torch.zeros((pre_padding_len, 4))
        # post_padding = torch.zeros((window_size - pre_padding_len - feature.shape[0], 4))
        #
        # feature = torch.cat([pre_padding, feature, post_padding], dim=0)

        return feature

    def _construct_sequence_features(self, graph_list, window_size=501):
        features = torch.tensor([])
        for grh in graph_list:
            feature = grh.ndata['feat']
            feature = self._feature_padding(feature, window_size=window_size)
            feature = feature.unsqueeze(0)
            features = torch.cat([features, feature], dim=0)
            pass
        return features



