import sys
import os
import psutil
import scipy.sparse as sp
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import gzip
import gc
import pickle as pkl
import networkx as nx
import copy
from typing import Dict, Tuple
import inspect


from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, normalize
import torch_geometric.transforms as T
from torch_geometric.utils.convert import to_networkx

from utils.graphs import *
import os


def load_data(dataset_str='cora', split_seed=0, renormalize=False, args=None):
    """Load data."""
    if  os.path.exists("./dataset/{}".format(dataset_str)):
        path = "dataset/{}".format(dataset_str)
    else:
        path = "dataset/"

    if dataset_str == 'aminer':
        
        adj = pkl.load(open(os.path.join(path, "{}.adj.sp.pkl".format(dataset_str)), "rb"))
        features = pkl.load(
            open(os.path.join(path, "{}.features.pkl".format(dataset_str)), "rb"))
        labels = pkl.load(
            open(os.path.join(path, "{}.labels.pkl".format(dataset_str)), "rb"))
        random_state = np.random.RandomState(split_seed)
        idx_train, idx_val, idx_test = get_train_val_test_split(
            random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
        idx_unlabel = np.concatenate((idx_val, idx_test))
        features = col_normalize(features)
    
    elif dataset_str in ['reddit']:
        adj = sp.load_npz(os.path.join(path, '{}_adj.npz'.format(dataset_str)))
        features = np.load(os.path.join(path, '{}_feat.npy'.format(dataset_str)))
        labels = np.load(os.path.join(path, '{}_labels.npy'.format(dataset_str))) 
        print(labels.shape, list(np.sum(labels, axis=0)))
        random_state = np.random.RandomState(split_seed)
        idx_train, idx_val, idx_test = get_train_val_test_split(
            random_state, labels, train_examples_per_class=20, val_examples_per_class=30)    
        idx_unlabel = np.concatenate((idx_val, idx_test))
        print(dataset_str, features.shape)
    
    elif dataset_str in ['Amazon2M']:
        adj = sp.load_npz(os.path.join(path, '{}_adj.npz'.format(dataset_str)))
        features = np.load(os.path.join(path, '{}_feat.npy'.format(dataset_str)))
        labels = np.load(os.path.join(path, '{}_labels.npy'.format(dataset_str)))
        random_state = np.random.RandomState(split_seed)
        class_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_size=20* class_num, val_size=30 * class_num)
        idx_unlabel = np.concatenate((idx_val, idx_test))
        print(labels.shape, list(np.sum(labels, axis=0)), type(idx_train))
        print("********: ", idx_train.shape, idx_val.shape, idx_test.shape)

    elif dataset_str in ['cora', 'citeseer', 'pubmed']:
        renormalize = True
        if os.path.exists("dataset/citation"):
           path = 'dataset/citation' 
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(os.path.join(path,"ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(
            os.path.join(path, "ind.{}.test.index".format(dataset_str)))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        # normalize
        features = normalize(features)
        features = features.todense()

        graph = nx.from_dict_of_lists(graph) 
        adj = nx.adjacency_matrix(graph)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        idx_train = np.arange(len(y)) #140
        idx_val = np.arange(len(y), len(y)+500) # 140  640 
        idx_test = np.asarray(test_idx_range.tolist()) # 1000
        idx_unlabel = np.arange(len(y), labels.shape[0])
    else:
        raise NotImplementedError

    if renormalize:
        adj = adj + sp.eye(adj.shape[0])
        D1 = np.array(adj.sum(axis=1))**(-0.5)
        D2 = np.array(adj.sum(axis=0))**(-0.5)
        D1 = sp.diags(D1, format='csr')
        D2 = sp.diags(D2, format='csr')

        A = adj.dot(D1)
        A = D2.dot(A)

    if 'ogbn' not in dataset_str:
         
        if dataset_str not in ['cora', 'citeseer', 'pubmed']:
            graph = nx.from_scipy_sparse_array(adj)

        n_classes =  labels.shape[1]
        labels = torch.from_numpy(labels)
        labels = torch.argmax(labels, -1)

        features = torch.from_numpy(features).float()

        idx_train = torch.from_numpy(idx_train)
        idx_val = torch.from_numpy(idx_val)
        idx_test = torch.from_numpy(idx_test)
        idx_unlabel = torch.from_numpy(idx_unlabel)

        A = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        print("To torch tensor! ")

    return graph, features, labels, idx_train, idx_val, idx_test, idx_unlabel, n_classes, sparse_mx_to_torch_sparse_tensor(A)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index




def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
    print(len(set(train_indices)), len(train_indices))
    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])
