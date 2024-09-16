# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import sys
import math
from tqdm import tqdm
import random
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)
import scipy.io as scio


def train_test_split_edges_direct(data, val_ratio: float = 0.05,
                                  test_ratio: float = 0.1):
    r"""Splits the edges of a :class:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges.
    As such, it will replace the :obj:`edge_index` attribute with
    :obj:`train_pos_edge_index`, :obj:`train_pos_neg_adj_mask`,
    :obj:`val_pos_edge_index`, :obj:`val_neg_edge_index` and
    :obj:`test_pos_edge_index` attributes.
    If :obj:`data` has edge features named :obj:`edge_attr`, then
    :obj:`train_pos_edge_attr`, :obj:`val_pos_edge_attr` and
    :obj:`test_pos_edge_attr` will be added as well.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation edges.
            (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test edges.
            (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr
    data.edge_index = data.edge_attr = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    if edge_attr is not None:
        edge_attr = edge_attr[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.val_pos_edge_attr = edge_attr[:n_v]

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.test_pos_edge_attr = edge_attr[n_v:n_v + n_t]

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    # if edge_attr is not None:
    #     out = to_undirected(data.train_pos_edge_index, edge_attr[n_v + n_t:])
    #     data.train_pos_edge_index, data.train_pos_edge_attr = out
    # else:
    #     data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data


def do_edge_split_direct(dataset, fast_split=False, val_ratio=0.0, test_ratio=0.2):
    data = dataset.clone()
    # data1:42 data2:2024
    random.seed(2024)
    torch.manual_seed(2024)

    if not fast_split:
        data = train_test_split_edges_direct(data, val_ratio, test_ratio)
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))
    else:
        num_nodes = data.num_nodes
        row, col = data.edge_index
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=num_nodes,
            num_neg_samples=row.size(0))
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.test_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.test_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge


class EdgeLoader(object):
    def __init__(self, train_edges, train_edge_false, batch_size, remain_delet=True, shuffle=True):
        self.shuffle = shuffle
        self.index = 0
        self.index_false = 0
        self.pos_edge = train_edges
        self.neg_edge = train_edge_false
        self.id_index = list(range(train_edges.shape[0]))
        self.data_len = len(self.id_index)
        self.remain_delet = remain_delet
        self.batch_size = batch_size
        if self.shuffle:
            self._shuffle()

    def __iter__(self):
        return self

    def _shuffle(self):
        random.shuffle(self.id_index)

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.remain_delet:
            if self.index + self.batch_size > self.data_len:
                self.index = 0
                self.index_false = 0
                self._shuffle()
                raise StopIteration
            batch_index = self.id_index[self.index: self.index + self.batch_size]
            batch_x = self.pos_edge[batch_index]
            batch_y = self.neg_edge[batch_index]
            self.index += self.batch_size

        else:
            if self.index >= self.data_len:
                self.index = 0
                self._shuffle()
                # raise StopIteration
            end_ = min(self.index + self.batch_size, self.data_len)
            batch_index = self.id_index[self.index: end_]
            batch_x = self.pos_edge[batch_index]
            batch_y = self.neg_edge[batch_index]
            self.index += self.batch_size
        return np.array(batch_x), np.array(batch_y)


class IndexLoader(object):
    def __init__(self, num_node, batch_size, drop_last=False, shuffle=True):
        self.shuffle = shuffle
        self.index = 0
        self.index_false = 0
        self.num_node = num_node
        data = np.array(range(num_node)).reshape(-1)
        self.data = torch.from_numpy(data)
        self.id_index = list(range(num_node))
        self.data_len = len(self.id_index)
        self.drop_last = drop_last
        self.batch_size = batch_size
        if self.shuffle:
            self._shuffle()

    def __iter__(self):
        return self

    def _shuffle(self):
        random.shuffle(self.id_index)

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.drop_last:
            if self.index + self.batch_size > self.data_len:
                self.index = 0
                self.index_false = 0
                self._shuffle()
                raise StopIteration
            batch_index = self.id_index[self.index: self.index + self.batch_size]
            batch_x = self.data[batch_index]
            self.index += self.batch_size

        else:
            if self.index >= self.data_len:
                self.index = 0
                self._shuffle()
                # raise StopIteration
            end_ = min(self.index + self.batch_size, self.data_len)
            batch_index = self.id_index[self.index: end_]
            batch_x = self.data[batch_index]
            self.index += self.batch_size
        return batch_x


def sparse_to_tuple(sparse_mx):
    if not ssp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def edgemask_um(mask_ratio, split_edge, device, num_nodes):
    if isinstance(split_edge, torch.Tensor):
        edge_index = split_edge
    else:
        edge_index = split_edge['train']['edge']
    num_edge = len(edge_index)
    index = np.arange(num_edge)
    np.random.shuffle(index)
    mask_num = int(num_edge * mask_ratio)
    pre_index = torch.from_numpy(index[0:-mask_num]).long()
    mask_index = torch.from_numpy(index[-mask_num:]).long()
    edge_index_train = edge_index[pre_index].t()
    edge_index_mask = edge_index[mask_index].t()
    edge_index = to_undirected(edge_index_train)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = SparseTensor.from_edge_index(edge_index).t()
    return adj, edge_index, edge_index_mask.to(device)
