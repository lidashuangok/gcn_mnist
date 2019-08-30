# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
import time
import numpy as np
import utils
import torch
import scipy.sparse as sp
from scipy import sparse

#adj =np.load("adj.npy")
#print(adj)
#_,_,feature,label = utils.load()


def onehot_encode(labes):
    classes = set(labes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labes_onehot = np.array(list(map(classes_dict.get,labes)), dtype=np.int32)
    #print(labes_onehot.shape)
    return labes_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data():
    print("load {} dataset ..... ".format("mnist"))
    _, _, feature, label = utils.load()
    features = sp.csr_matrix(feature, dtype=np.float32)
    labels = onehot_encode(label)
    #adj =np.load("adj.npy")
    adj = sparse.load_npz('adj_sp.npz')
    #print(adj.shape)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    idx_train = range(1000)
    idx_val = range(3000, 4000)
    idx_test = range(5000, 6000)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    # print(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

if __name__ == '__main__':
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    #print(idx_test)