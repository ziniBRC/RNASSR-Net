import re
import os
import sys
import RNA
import torch
import gzip
import pickle
import dgl
from utils.seq_motifs import get_motif
import random
import subprocess
import numpy as np
import scipy.sparse as sp
from functools import partial
import forgi.graph.bulge_graph as fgb

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.general_utils import Pool


def adj_to_bias(adj, nhood=1):
    # [batch_size, nb_nodes, nb_nodes]
    mt = np.stack([np.eye(adj.shape[1])] * adj.shape[0], axis=0)  # self-connection
    for _ in range(nhood):
        mt = np.matmul(mt, (adj + np.stack([np.eye(adj.shape[1])] * adj.shape[0], axis=0)))
    mt = np.greater(mt, 0.).astype(np.float32)
    return -1e9 * (1.0 - mt)


# >>>> equilibrium probability using RNAplfold >>>>>>>

def fold_seq_rnashapes(seq, winsize, iterations=100):
    stride = winsize // 4
    cmd = 'echo %s | RNAshapes -w %d -W %d -i %d -A -t 1 -c %d -M 0 -o 1' % (seq, winsize, stride, iterations, 10)
    ret = subprocess.check_output(cmd, shell=True)
    lines = re.sub(' +', ' ', ret.decode('utf-8')).rstrip().split('\n')

    # assemble adjacency matrix
    row_col, link, prob, norm = [], [], [], []
    length = len(seq)
    for i in range(length):
        if i != length - 1:
            row_col.append((i, i + 1))
            link.append(1)
            prob.append(1.)
            norm.append(1)
        if i != 0:
            row_col.append((i, i - 1))
            link.append(2)
            prob.append(1.)
            norm.append(1)
    for line in lines:
        if len(line) > 0:
            if line[0] >= '0' and line[0] <= '9':
                # indices
                start_idx = int(line.split(' ')[0]) - 1
                local_idx = []
            elif line[0] in ['(', '.', ')']:
                # secondary structures
                tokens = line.split(' ')
                struct = tokens[0]
                probability = float(tokens[2])

                bg = fgb.BulgeGraph.from_dotbracket(struct)
                for i, ele in enumerate(struct):
                    if ele == '(':
                        pair_from = i + start_idx
                        pair_to = bg.pairing_partner(i + 1) - 1 + start_idx
                        if not (pair_from, pair_to) in row_col:
                            row_col.append((pair_from, pair_to))
                            link.append(3)
                            prob.append(probability)
                            norm.append(1)
                            local_idx.append((pair_from, pair_to))
                            # symmetric
                            row_col.append((pair_to, pair_from))
                            link.append(4)
                            prob.append(probability)
                            norm.append(1)
                        else:
                            idx = row_col.index((pair_from, pair_to))
                            prob[idx] += probability
                            prob[idx + 1] += probability
                            if not (pair_from, pair_to) in local_idx:
                                local_idx.append((pair_from, pair_to))
                                norm[idx] += 1
                                norm[idx + 1] += 1

    prob = np.array(prob) / np.array(norm)
    print(norm)
    return (sp.csr_matrix((link, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length)),
            sp.csr_matrix((prob, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length)),)


# >>>> equilibrium probability using RNAplfold >>>>>>>
# when on compute canada, make sure this is happening on the compute nodes

def fold_seq_rnaplfold(seq, w, l, cutoff, no_lonely_bps):
    np.random.seed(random.seed())
    name = str(np.random.rand())
    # Call RNAplfold on command line.
    no_lonely_bps_str = ""
    if no_lonely_bps:
        no_lonely_bps_str = "--noLP"
    cmd = 'echo %s | RNAplfold -W %d -L %d -c %.4f --id-prefix %s %s' % (seq, w, l, cutoff, name, no_lonely_bps_str)
    ret = subprocess.call(cmd, shell=True)

    # assemble adjacency matrix
    row_col, link, prob = [], [], []
    length = len(seq)
    for i in range(length):
        if i != length - 1:
            row_col.append((i, i + 1))
            link.append(1)
            prob.append(1.)
        if i != 0:
            row_col.append((i, i - 1))
            link.append(2)
            prob.append(1.)
    # Extract base pair information.
    name += '_0001_dp.ps'
    start_flag = False
    with open(name) as f:
        for line in f:
            if start_flag:
                values = line.split()
                if len(values) == 4:
                    source_id = int(values[0]) - 1
                    dest_id = int(values[1]) - 1
                    avg_prob = float(values[2])
                    # source_id < dest_id
                    row_col.append((source_id, dest_id))
                    link.append(3)
                    prob.append(avg_prob ** 2)
                    row_col.append((dest_id, source_id))
                    link.append(4)
                    prob.append(avg_prob ** 2)
            if 'start of base pair probability data' in line:
                start_flag = True
    # delete RNAplfold output file.
    os.remove(name)
    # placeholder for dot-bracket structure
    return (sp.csr_matrix((link, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length)),
            sp.csr_matrix((prob, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length)),)


# >>>> Boltzmann sampling using RNAsubopt >>>>>>>

def sample_one_seq(seq, passes=10):
    max_len = len(seq)
    adj_mat_tmp = np.zeros((max_len, max_len), dtype=np.int32)
    for i in range(max_len):
        if i != max_len - 1:
            adj_mat_tmp[i, i + 1] = 1
        if i != 0:
            adj_mat_tmp[i, i - 1] = 2
    cmd = 'echo "%s" | RNAsubopt --stochBT=%d' % (seq, passes)
    all_struct = subprocess.check_output(cmd, shell=True). \
                     decode('utf-8').rstrip().split('\n')[1:]
    all_adj_mat = []
    for struct in all_struct:
        adj_mat = adj_mat_tmp.copy()
        bg = fgb.BulgeGraph.from_dotbracket(struct)
        for i, ele in enumerate(struct):
            if ele == '(':
                adj_mat[i, bg.pairing_partner(i + 1) - 1] = 3
            elif ele == ')':
                adj_mat[i, bg.pairing_partner(i + 1) - 1] = 4
        all_adj_mat.append(adj_mat)
    all_adj_mat = np.stack(all_adj_mat, axis=0)
    return all_adj_mat


def fold_seq_subopt(seq, probabilistic=False, sampling_amount=1000):
    # RNAfold is only suitable for short RNA sequences within 100 nucleotides
    if probabilistic:
        # sampling from a boltzmann ensemble
        cmd = 'echo "%s" | RNAsubopt --stochBT=%d' % (seq, sampling_amount)
        struct_list = subprocess.check_output(cmd, shell=True). \
                          decode('utf-8').rstrip().split('\n')[1:]
    else:
        struct_list, energy_list = [], []

        def collect_subopt_result(structure, energy, *args):
            if not structure == None:
                struct_list.append(structure)
                energy_list.append(energy)

        # Enumerate all structures 100 dacal/mol = 1 kcal/mol around
        # default deltaEnergy is the MFE
        RNA.fold_compound(seq).subopt_cb(100, collect_subopt_result, None)

        # sort
        struct_list = list(np.array(struct_list)[np.argsort(energy_list)])

    # merging all structures into a single adjacency matrix
    # probability returning two matrices
    matrix = adj_mat_subopt(struct_list, probabilistic)
    # process the structures
    return structural_content(struct_list), matrix


def structural_content(struct_list):
    size = len(struct_list)
    length = len(struct_list[0])
    content = np.zeros((length, 3), dtype=np.int32)
    for i in range(length):
        for j in range(size):
            idx = '.()'.index(struct_list[j][i])
            content[i][idx] += 1
    return content.astype(np.float32) / size


def adj_mat_subopt(struct_list, probabilistic):
    # create sparse matrix
    row_col, data = [], []
    length = len(struct_list[0])
    counts = []
    for i in range(length):
        if i != length - 1:
            row_col.append((i, i + 1))
            data.append(1)
            counts.append(0)
        if i != 0:
            row_col.append((i, i - 1))
            data.append(2)
            counts.append(0)
    if probabilistic:
        for struct in struct_list:
            bg = fgb.BulgeGraph.from_dotbracket(struct)
            for i, ele in enumerate(struct):
                if ele == '(':
                    if not (i, bg.pairing_partner(i + 1) - 1) in row_col:
                        row_col.append((i, bg.pairing_partner(i + 1) - 1))
                        data.append(3)
                        counts.append(1)
                    else:
                        idx = row_col.index((i, bg.pairing_partner(i + 1) - 1))
                        counts[idx] += 1
                elif ele == ')':
                    if not (i, bg.pairing_partner(i + 1) - 1) in row_col:
                        row_col.append((i, bg.pairing_partner(i + 1) - 1))
                        data.append(4)
                        counts.append(1)
                    else:
                        idx = row_col.index((i, bg.pairing_partner(i + 1) - 1))
                        counts[idx] += 1
        # normalize each row into probabilities
        for i in range(len(row_col)):
            if counts[i] > 0:
                # have to be a hydrogen bond
                counts[i] /= len(struct_list)
            else:
                # covalent bond that forms the stem
                counts[i] = 1.
        return (sp.csr_matrix((data, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length)),
                sp.csr_matrix((counts, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length)),)
    else:
        for struct in struct_list:
            bg = fgb.BulgeGraph.from_dotbracket(struct)
            for i, ele in enumerate(struct):
                if ele == '(':
                    if not (i, bg.pairing_partner(i + 1) - 1) in row_col:
                        row_col.append((i, bg.pairing_partner(i + 1) - 1))
                        data.append(3)
                elif ele == ')':
                    if not (i, bg.pairing_partner(i + 1) - 1) in row_col:
                        row_col.append((i, bg.pairing_partner(i + 1) - 1))
                        data.append(4)
        return sp.csr_matrix((data, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])),
                             shape=(length, length))


# >>>> MFE structure using RNAfold >>>>>>>

def fold_seq_rnafold(seq):
    '''fold sequence using RNAfold'''
    struct = RNA.fold(seq)[0]
    matrix = adj_mat(struct)
    return structural_content([struct]), matrix


def adj_mat(struct):
    # create sparse matrix
    row_col, data = [], []
    length = len(struct)
    for i in range(length):
        if i != length - 1:
            row_col.append((i, i + 1))
            data.append(1)
        if i != 0:
            row_col.append((i, i - 1))
            data.append(2)
    bg = fgb.BulgeGraph.from_dotbracket(struct)
    for i, ele in enumerate(struct):
        if ele == '(':
            row_col.append((i, bg.pairing_partner(i + 1) - 1))
            data.append(3)
        elif ele == ')':
            row_col.append((i, bg.pairing_partner(i + 1) - 1))
            data.append(4)
    return sp.csr_matrix((data, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])),
                         shape=(length, length))


def augment_features(path):
    '''
    try to avoid this as much as possible.
    we are mostly interested in an end-to-end learning scenario
    '''
    # region type: 101 x 5
    region_types = np.loadtxt(gzip.open(os.path.join(path, "matrix_RegionType.tab.gz")), skiprows=1)
    assert (region_types.shape[1] == 505)  # 4 region types
    region_types = np.transpose(region_types.reshape((region_types.shape[0], 5, 101)), [0, 2, 1])

    coclip = np.loadtxt(gzip.open(os.path.join(path, "matrix_Cobinding.tab.gz")), skiprows=1)
    assert (coclip.shape[1] % 101 == 0)
    nb_exprs = coclip.shape[1] // 101
    coclip = np.transpose(coclip.reshape((coclip.shape[0], nb_exprs, 101)), [0, 2, 1])

    return np.concatenate([region_types, coclip], axis=-1)


def load_fasta_format(file):
    all_id = []
    all_seq = []
    seq = ''
    for row in file:
        if type(row) is bytes:
            row = row.decode('utf-8')
        row = row.rstrip()
        if row.startswith('>'):
            all_id.append(row)
            if seq != '':
                all_seq.append(seq)
                seq = ''
        else:
            seq += row
    all_seq.append(seq)
    return all_id, all_seq


def load_dotbracket(filepath, pool=None, fold_algo='rnafold', probabilistic=False, **kwargs):
    prefix = '%s_%s_' % (fold_algo, probabilistic)
    if fold_algo == 'rnaplfold' or fold_algo == 'rnashapes':
        prefix += '%d_' % (kwargs.get('w', 150))
    full_path = os.path.join(os.path.dirname(filepath), '{}structures.npy'.format(prefix))
    if not os.path.exists(full_path):
        print(full_path, 'is missing. Begin folding from scratch.')
        fold_rna_from_file(filepath, pool, fold_algo, probabilistic, **kwargs)
    # load secondary structures
    all_struct = np.load(
        os.path.join(os.path.dirname(filepath), '{}structures.npy'.format(prefix)))
    return all_struct


def load_mat(filepath, pool=None, fold_algo='rnafold', probabilistic=False, **kwargs):
    prefix = '%s_%s_' % (fold_algo, probabilistic)
    # folding length is crucial hyperparam for local RNA folding, therefore should be marked
    if fold_algo == 'rnaplfold' or fold_algo == 'rnashapes':
        prefix += '%d_' % (kwargs.get('w', 150))
    if kwargs.get('modify_leaks', False):
        # this will make sure we load the modified sequences
        # incorrect secondary structure may still give away information / data statistics
        prefix = 'modified_' + prefix

    if not os.path.exists(
            os.path.join(os.path.dirname(filepath), '{}rel_mat.obj'.format(prefix))) or probabilistic and \
            not os.path.exists(os.path.join(os.path.dirname(filepath), '{}prob_mat.obj'.format(prefix))):
        print('adj mat or prob mat is missing. Begin folding from scratch.')
        fold_rna_from_file(filepath, pool, fold_algo, probabilistic, **kwargs)

    load_dense = kwargs.get('load_dense', True)
    sp_rel_matrix = pickle.load(open(os.path.join(os.path.dirname(filepath), '{}rel_mat.obj'.format(prefix)), 'rb'))
    if load_dense:
        adjacency_matrix = np.array([mat.toarray() for mat in sp_rel_matrix])
    else:
        adjacency_matrix = np.array(sp_rel_matrix)

    if probabilistic:
        sp_prob_matrix = pickle.load(
            open(os.path.join(os.path.dirname(filepath), '{}prob_mat.obj'.format(prefix)), 'rb'))
        if load_dense:
            probability_matrix = np.array([mat.toarray() for mat in sp_prob_matrix])
        else:
            probability_matrix = np.array(sp_prob_matrix)
        matrix = (adjacency_matrix, probability_matrix)
    else:
        matrix = adjacency_matrix

    return matrix


def load_seq(filepath):
    if filepath.endswith('.fa'):
        file = open(filepath, 'r')
    else:
        file = gzip.open(filepath, 'rb')

    all_id, all_seq = load_fasta_format(file)
    for i in range(len(all_seq)):
        seq = all_seq[i]
        # seq = seq[:-1].upper()
        all_seq[i] = seq.replace('T', 'U')
        all_seq[i] = seq.replace('t', 'u')
    return all_id, all_seq


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


def detect_motifs(model, data_loader, device, output_dir='motifs'):
    seq_list = []
    base_weight_list = []
    filter_out_list = []
    node_weight_list = []
    filter_weights = None
    for i, param in enumerate(model.layers_cnn.parameters()):
        if i == 2:
            filter_weights = param.data.cpu().numpy()
            break
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)

    start_flag = 1
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            pos_ind = batch_labels == 1
            batch_graphs = dgl.batch(batch_graphs)
            batch_graphs.ndata['feat'] = batch_graphs.ndata['feat'].to(device)
            batch_graphs.edata['feat'] = batch_graphs.edata['feat'].to(device)
            batch_x = batch_graphs.ndata['feat'] # num x feat
            batch_e = batch_graphs.edata['feat']
            batch_labels = batch_labels.to(device)

            _ = model.forward(batch_graphs, batch_x, batch_e)
            one_hot_matrices = model.sequence.detach().cpu().numpy()
            seq_list.extend(matrix2seq(one_hot_matrices[pos_ind, :]))

            # should this place be replaced by individual filter?
            if start_flag == 1:
                filter_out_list = model.filter_out.detach().cpu().numpy()[pos_ind, :]
                node_weight_list = model.node_weight.detach().cpu().numpy()[pos_ind, :]
                start_flag = 0
            else:
                filter_out = model.filter_out.detach().cpu().numpy()[pos_ind, :]
                node_weight = model.node_weight.detach().cpu().numpy()[pos_ind, :]
                filter_out_list = np.vstack((filter_out_list, filter_out))
                node_weight_list = np.vstack((node_weight_list, node_weight))

            if len(seq_list) >= 10000:
                break

    get_motif(filter_weights[:, 0, :, :], filter_out_list, seq_list, dir1=output_dir + '/size7', filter_size=7)


def fold_rna_from_file(filepath, p=None, fold_algo='rnafold', probabilistic=False, **kwargs):
    assert (fold_algo in ['rnafold', 'rnasubopt', 'rnaplfold'])
    if fold_algo == 'rnafold':
        assert (probabilistic is False)
    if fold_algo == 'rnaplfold':
        assert (probabilistic is True)
    print('Parsing', filepath)
    _, all_seq = load_seq(filepath)

    # compatible with already computed structures with RNAfold
    prefix = '%s_%s_' % (fold_algo, probabilistic)
    if fold_algo == 'rnaplfold' or fold_algo == 'rnashapes':
        prefix += '%d_' % (kwargs.get('w', 150))
    if kwargs.get('modify_leaks', False):
        prefix = 'modified_' + prefix

    if p is None:
        pool = Pool(int(os.cpu_count() * 2 / 3))
    else:
        pool = p

    if fold_algo == 'rnafold':
        fold_func = fold_seq_rnafold
        res = list(pool.imap(fold_func, all_seq))
        sp_rel_matrix = []
        structural_content = []
        for struct, matrix in res:
            structural_content.append(struct)
            sp_rel_matrix.append(matrix)
        np.save(os.path.join(os.path.dirname(filepath), '{}structures.npy'.format(prefix)),
                np.array(structural_content))  # [size, length, 3]
        pickle.dump(sp_rel_matrix,
                    open(os.path.join(os.path.dirname(filepath), '{}rel_mat.obj'.format(prefix)), 'wb'))
    elif fold_algo == 'rnasubopt':
        fold_func = partial(fold_seq_subopt, fold_algo=fold_algo, probabilistic=probabilistic)
        res = list(pool.imap(fold_func, all_seq))
        sp_rel_matrix = []
        sp_prob_matrix = []
        structural_content = []
        for struct, matrix in res:
            structural_content.append(struct)
            if probabilistic:
                rel_mat, prob_mat = matrix
                sp_prob_matrix.append(prob_mat)
            else:
                rel_mat = matrix
            sp_rel_matrix.append(rel_mat)
        np.save(os.path.join(os.path.dirname(filepath), '{}structures.npy'.format(prefix)),
                np.array(structural_content))  # [size, length, 3]
        pickle.dump(sp_rel_matrix,
                    open(os.path.join(os.path.dirname(filepath), '{}rel_mat.obj'.format(prefix)), 'wb'))
        if probabilistic:
            pickle.dump(sp_prob_matrix,
                        open(os.path.join(os.path.dirname(filepath), '{}prob_mat.obj'.format(prefix)), 'wb'))
    elif fold_algo == 'rnaplfold':
        winsize = kwargs.get('w', 150)
        print('running rnaplfold with winsize %d' % (winsize))
        fold_func = partial(fold_seq_rnaplfold, w=winsize, l=min(winsize, 150), cutoff=1e-4, no_lonely_bps=True)
        res = list(pool.imap(fold_func, all_seq))
        sp_rel_matrix = []
        sp_prob_matrix = []
        for rel_mat, prob_mat in res:
            sp_rel_matrix.append(rel_mat)
            sp_prob_matrix.append(prob_mat)
        pickle.dump(sp_rel_matrix,
                    open(os.path.join(os.path.dirname(filepath), '{}rel_mat.obj'.format(prefix)), 'wb'))
        pickle.dump(sp_prob_matrix,
                    open(os.path.join(os.path.dirname(filepath), '{}prob_mat.obj'.format(prefix)), 'wb'))
    else:
        raise ValueError('Supported folding algorithms are ' + ', '.join(['rnafold', 'rnasubopt', 'rnaplfold']))

    if p is None:
        pool.close()
        pool.join()

    print('Parsing', filepath, 'finished')


def fold_and_check_hairpin(seq, return_label=True):
    regex = r'^\(\(\(\.\.\.\)\)\)[\.\(]|[\.\)]\(\(\(\.\.\.\)\)\)$|[\.\)]\(\(\(\.\.\.\)\)\)|\(\(\(\.\.\.\)\)\)[\.\(]'
    '''return label, or an annotation over the entire seq'''
    '''fold rna and check if the structure contains a hairpin of 3 loose nucleotide connected by a stem of 3 basepairs'''
    struct = RNA.fold(seq)[0]
    mat = adj_mat(struct)
    if return_label:
        match = re.search(regex, struct)
        return struct, mat, int(match is not None)
    else:
        annotation = [0] * len(seq)
        for match in re.finditer(regex, struct):
            start_idx = struct[match.start(): match.end()].index('(((...)))') + match.start()
            annotation[start_idx:start_idx + 9] = [1] * 9
        return struct, mat, annotation


def fold_and_check_element(seq, element_symbol, return_label=True):
    '''simply use forgi to annotate the whole string and check if it contains the element'''
    '''return label, or an annotation over the entire seq'''
    '''fold rna and check if the structure contains a hairpin of 3 loose nucleotide connected by a stem of 3 basepairs'''
    struct = RNA.fold(seq)[0]
    mat = adj_mat(struct)
    bg = fgb.BulgeGraph.from_dotbracket(struct)
    annotation = bg.to_element_string(())
    if return_label:
        return struct, mat, int(element_symbol in annotation)
    else:
        return struct, mat, [int(element_symbol == c) for c in annotation]


def generate_hairpin_dataset(n, length, p=None, return_label=True):
    '''
    generate toy dataset
    positive examples: RNA sequences that contain specific structural motifs:
        1. a hairpin of three nucleotides connected by a stem of 3 base-pairs
        2. nucleotidal composition does not matter.
    negative examples: RNA sequences that do not contain this specific motifs
    '''
    data_path = os.path.join(basedir, 'Data/toy-data/hairpin/%s' % ('label' if return_label else 'annotation'))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if os.path.exists(os.path.join(data_path, 'seq-and-struct.fa')) and \
            os.path.exists(os.path.join(data_path, 'adj_mat.obj')):
        all_labels = []
        all_seqs = []
        all_struct = []
        with open(os.path.join(data_path, 'seq-and-struct.fa'), 'r') as file:
            for line in file:
                if line[0] == '>':
                    label = line.rstrip().split(' ')[-1].split(':')[-1]
                    if return_label:
                        all_labels.append(int(label))
                    else:
                        all_labels.append([int(c) for c in label.split(',')])
                elif line[0] in 'ACGT':
                    all_seqs.append(['ACGT'.index(c) for c in line.rstrip()])
                elif line[0] in '.()':
                    all_struct.append(['.()'.index(c) for c in line.rstrip()])

        all_seqs = np.array(all_seqs)
        sp_adj_matrix = pickle.load(open(os.path.join(data_path, 'adj_mat.obj'), 'rb'))
    else:
        all_seqs = np.zeros((n, length), dtype=int)
        for j in range(length):
            all_seqs[:, j] = np.random.choice([0, 1, 2, 3], n, p=[0.25, 0.25, 0.25, 0.25])
        seqs_str = [''.join(['ACGT'[c] for c in seq]) for seq in all_seqs]

        if p is None:
            pool = Pool(8)
        else:
            pool = p

        res = list(pool.imap(partial(fold_and_check_hairpin, return_label=return_label), seqs_str))

        sp_adj_matrix = []
        all_labels = []
        all_struct = []
        with open(os.path.join(data_path, 'seq-and-struct.fa'), 'w') as file:
            for seq, (struct, mat, label) in zip(seqs_str, res):
                file.writelines('> label:%s\n%s\n%s\n' %
                                (label if return_label else ','.join([str(c) for c in label]), seq, struct))
                sp_adj_matrix.append(mat)
                all_labels.append(label)
                all_struct.append(['.()'.index(c) for c in struct])

        pickle.dump(sp_adj_matrix, open(os.path.join(data_path, 'adj_mat.obj'), 'wb'))
        if p is None:
            pool.close()
            pool.join()

    all_labels = np.array(all_labels)
    adjacency_matrix = np.stack([mat.toarray() for mat in sp_adj_matrix], axis=0)
    all_struct = np.array(all_struct)
    return all_seqs, adjacency_matrix, all_labels, all_struct


def generate_element_dataset(n, length, element_symbol, p=None, return_label=True):
    '''
    generate toy dataset
    positive examples: RNA sequences that contain specific structural motifs:
        1. a hairpin of three nucleotides connected by a stem of 3 base-pairs
        2. nucleotidal composition does not matter.
    negative examples: RNA sequences that do not contain this specific motifs
    '''
    assert (len(element_symbol) == 1 and str.isalpha(element_symbol))
    data_path = os.path.join(basedir,
                             'Data/toy-data/%s/%s' % (element_symbol, 'label' if return_label else 'annotation'))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if os.path.exists(os.path.join(data_path, 'seq-and-struct.fa')) and \
            os.path.exists(os.path.join(data_path, 'adj_mat.obj')):
        all_labels = []
        all_seqs = []
        all_struct = []
        with open(os.path.join(data_path, 'seq-and-struct.fa'), 'r') as file:
            for line in file:
                if line[0] == '>':
                    label = line.rstrip().split(' ')[-1].split(':')[-1]
                    if return_label:
                        all_labels.append(int(label))
                    else:
                        all_labels.append([int(c) for c in label.split(',')])
                elif line[0] in 'ACGT':
                    all_seqs.append(['ACGT'.index(c) for c in line.rstrip()])
                elif line[0] in '.()':
                    all_struct.append(['.()'.index(c) for c in line.rstrip()])

        all_seqs = np.array(all_seqs)
        sp_adj_matrix = pickle.load(open(os.path.join(data_path, 'adj_mat.obj'), 'rb'))
    else:
        all_seqs = np.zeros((n, length), dtype=int)
        for j in range(length):
            all_seqs[:, j] = np.random.choice([0, 1, 2, 3], n, p=[0.25, 0.25, 0.25, 0.25])
        seqs_str = [''.join(['ACGT'[c] for c in seq]) for seq in all_seqs]

        if p is None:
            pool = Pool(8)
        else:
            pool = p

        res = list(pool.imap(partial(fold_and_check_element,
                                     element_symbol=element_symbol,
                                     return_label=return_label), seqs_str))

        sp_adj_matrix = []
        all_labels = []
        all_struct = []
        with open(os.path.join(data_path, 'seq-and-struct.fa'), 'w') as file:
            for seq, (struct, mat, label) in zip(seqs_str, res):
                file.writelines('> label:%s\n%s\n%s\n' %
                                (label if return_label else ','.join([str(c) for c in label]), seq, struct))
                sp_adj_matrix.append(mat)
                all_labels.append(label)
                all_struct.append(['.()'.index(c) for c in struct])

        pickle.dump(sp_adj_matrix, open(os.path.join(data_path, 'adj_mat.obj'), 'wb'))
        if p is None:
            pool.close()
            pool.join()

    all_labels = np.array(all_labels)
    adjacency_matrix = np.stack([mat.toarray() for mat in sp_adj_matrix], axis=0)
    all_struct = np.array(all_struct)
    return all_seqs, adjacency_matrix, all_labels, all_struct


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf, edgeitems=30, linewidth=100000, )

    # res = fold_seq_rnaplfold(
    #     'cgcgggacgcggcccgaggccgtgcgcgagccggggcaccgggcggcggcggcggcggcgcgcgccatgtcgttcagtgaaatgaaccgcaggacgctggcgttccgaggaggcgggttggtcaccgctagcggcggcggctccacgaacAATAACGCTGGCGGGGAGGCCTCAGcttggcctccgcagccccagccgagacagcccccgccgccagcgccgcccgcgcttcagccgcctaatgggcggggggccgacgaggaagtggaattggagggcctggagccccaagacctggaggcctccgccgggccggccgccggcg',
    #     150, 150, 0.0001, True)
    # print(res[1].todense().sum(axis=-1))
    res = fold_seq_rnashapes(
        'cgcgggacgcggcccgaggccgtgcgcgagccggggcaccgggcggcggcggcggcggcgcgcgccatgtcgttcagtgaaatgaaccgcaggacgctggcgttccgaggaggcgggttggtcaccgctagcggcggcggctccacgaacAATAACGCTGGCGGGGAGGCCTCAGcttggcctccgcagccccagccgagacagcccccgccgccagcgccgcccgcgcttcagccgcctaatgggcggggggccgacgaggaagtggaattggagggcctggagccccaagacctggaggcctccgccgggccggccgccggcg',
        150, iterations=100)
    # print(res[0].todense())
    print(res[1].todense().sum(axis=-1))

    # with open('boltzmann-sampling-acc.txt', 'w') as file:
    #     for amount in [5, 10, 100, 1000, 5000, 10000]:
    #         rel_diff, prob_diff = [], []
    #         for replcate in range(100):
    #             _, res = fold_seq_subopt(
    #                 'TGTGAAGCGCGGCTAGCTGCCGGGGTTCGAGGTGGGTCCCAGGGTTAAAATCCCTTGTTGTCTTACTGGTGGCAGCAAGCTAGGACTATACTCCTCGGTCG',
    #                 'rnafold', True, amount)
    #             _, new_res = fold_seq_subopt(
    #                 'TGTGAAGCGCGGCTAGCTGCCGGGGTTCGAGGTGGGTCCCAGGGTTAAAATCCCTTGTTGTCTTACTGGTGGCAGCAAGCTAGGACTATACTCCTCGGTCG',
    #                 'rnafold', True, amount)
    #
    #             diff = (res[0].todense() != new_res[0].todense()).astype(np.int32)
    #             rel_diff.append(np.sum(diff))
    #
    #             diff = np.abs(res[1].todense() - new_res[1].todense())
    #             prob_diff.append(np.mean(np.max(diff, axis=-1)))
    #         file.writelines('sampling amount %d, relation difference: %.4f\u00b1%.4f, probability difference: %.4f\u00b1%.4f' %
    #               (amount, np.mean(rel_diff), np.std(rel_diff), np.mean(prob_diff), np.std(prob_diff)))
    #         print('sampling amount %d, relation difference: %.4f\u00b1%.4f, probability difference: %.4f\u00b1%.4f' %
    #               (amount, np.mean(rel_diff), np.std(rel_diff), np.mean(prob_diff), np.std(prob_diff)))

    # annotation for the multiloop elements
    # all_seqs, adjacency_matrix, all_labels, _ = generate_element_dataset(80000, 101, 'i', return_label=False)
    # print(all_labels.shape)
    # print(np.where(np.count_nonzero(all_labels, axis=-1) > 0)[0].__len__())
    #
    # all_seqs, adjacency_matrix, all_labels, _ = generate_hairpin_dataset(80000, 101, 'm', return_label=False)
    # print(all_labels.shape)
    # print(np.where(np.count_nonzero(all_labels, axis=-1) > 0)[0].__len__())
