# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from itertools import islice
from math import floor
import numpy as np
from torchvision import datasets

def mnist_iid(dataset, num_users):
    # TODO: need to be changed when changing dataset
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users):
    # TODO: need to be changed when changing dataset
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def amazon_670k_iid(dataset, num_users):
    # TODO: need to be changed when changing dataset
    """
    Args:
        dataset:
        num_users:
    Returns:
    """
    num_shards = 100
    total_training_data = floor(490450/100) * num_shards
    num_data = int(total_training_data / num_shards)
    idxs = np.arange(num_shards*num_data)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(idxs, num_data, replace=False))
        idxs = list(set(idxs) - dict_users[i])
    return dict_users

def amazon_670k_noniid(dataset, num_users):
    # TODO: need to be changed when changing dataset
    """
    Args:
        dataset:
        num_users:
    Returns:
    """
    num_shards = 100
    total_training_data = floor(490450/100) * num_shards
    num_data = int(total_training_data / num_shards)
    idxs = np.arange(num_shards*num_data)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(idxs, num_data, replace=False))
        idxs = list(set(idxs) - dict_users[i])
    return dict_users

def wiki10_iid(dataset, num_users):
    # TODO: need to be changed when changing dataset
    """
    Args:
        dataset:
        num_users:
    Returns:
    """
    num_shards = 10
    total_training_data = floor(14146/10) * num_shards
    num_data = int(total_training_data / num_shards)
    idxs = np.arange(num_shards*num_data)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(idxs, num_data, replace=False))
        idxs = list(set(idxs) - dict_users[i])
    return dict_users

def wiki10_noniid(dataset, num_users):
    # TODO: need to be changed when changing dataset
    """
    Args:
        dataset:
        num_users:
    Returns:
    """
    num_shards = 10
    num_labels = 30938
    num_training_data = 14146
    topk = num_shards
    file_array = []
    labels =[]
    labels_count = np.zeros(num_labels, dtype=int)
    with open(dataset, 'r', encoding='utf-8') as f:
        file_array = f.readlines()
        for line in file_array:
            itms = line.strip().split()
            this_label = []
            for itm in itms[0].split(','):
                this_label.append(int(itm))
                labels_count[int(itm)] = labels_count[int(itm)] + 1
            labels.append(this_label)
    # find the idx of topk elements. (idx contain the labels)
    idx = np.argpartition(labels_count, -topk)[-topk:]
    # find the idx of data point with topk labels.
    label_for_each_idx = {i: np.array([],dtype=int) for i in idx}
    for id in idx:
        for i in range(len(labels)):
            if id in labels[i]:
                label_for_each_idx[id] = np.append(label_for_each_idx[id], int(i))

    total_training_data = floor(num_training_data/num_shards) * num_shards
    num_data = int(total_training_data / num_shards)
    idxs = np.arange(num_shards*num_data)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    for i in range(num_users):
        dict_users[i] = np.append(dict_users[i], label_for_each_idx[idx[i]])
        if len(dict_users[i]) < num_data:
            dict_users[i] = np.append(dict_users[i], np.array(np.random.choice(idxs, num_data - len(dict_users[i]), replace=False)))
        else:
            dict_users[i] = np.array(np.random.choice(dict_users[i], num_data, replace=False))
    return dict_users





def Eurlex_noniid(dataset, num_users):
    # # TODO: need to be changed when changing dataset
    # """
    # Args:
    #     dataset:
    #     num_users:
    # Returns:
    # """
    # num_shards = 10
    # total_training_data = floor(12920/10) * num_shards
    # num_data = int(total_training_data / num_shards)
    # idxs = np.arange(num_shards*num_data)
    # dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    # for i in range(num_users):
    #     dict_users[i] = set(np.random.choice(idxs, num_data, replace=False))
    #     idxs = list(set(idxs) - dict_users[i])
    # return dict_users

    # TODO: need to be changed when changing dataset
    """
    Args:
        dataset:
        num_users:
    Returns:
    """
    num_shards = 10
    num_labels = 3993
    num_training_data = 15539
    topk = num_shards
    file_array = []
    labels =[]
    labels_count = np.zeros(num_labels, dtype=int)
    with open(dataset, 'r', encoding='utf-8') as f:
        file_array = f.readlines()
        for line in file_array:
            itms = line.strip().split(' ')
            try:
                this_label = []
                for itm in itms[0].split(','):
                    this_label.append(int(itm))
                    labels_count[int(itm)] = labels_count[int(itm)] + 1
            except:
                this_label = [0]
            labels.append(this_label)
    # find the idx of topk elements. (idx contain the labels)
    idx = np.argpartition(labels_count, -topk)[-topk:]
    # find the idx of data point with topk labels.
    label_for_each_idx = {i: np.array([],dtype=int) for i in idx}
    for id in idx:
        for i in range(len(labels)):
            if id in labels[i]:
                label_for_each_idx[id] = np.append(label_for_each_idx[id], int(i))

    total_training_data = floor(num_training_data/num_shards) * num_shards
    num_data = int(total_training_data / num_shards)
    idxs = np.arange(num_shards*num_data)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    for i in range(num_users):
        dict_users[i] = np.append(dict_users[i], label_for_each_idx[idx[i]])
        if len(dict_users[i]) < num_data:
            dict_users[i] = np.append(dict_users[i], np.array(np.random.choice(idxs, num_data - len(dict_users[i]), replace=False)))
        else:
            dict_users[i] = np.array(np.random.choice(dict_users[i], num_data, replace=False))
    return dict_users





def Eurlex_iid(dataset, num_users):
    # TODO: need to be changed when changing dataset
    """
    Args:
        dataset:
        num_users:
    Returns:
    """
    num_shards = num_users
    total_training_data = floor(15539/num_users) * num_shards
    num_data = int(total_training_data / num_shards)
    idxs = np.arange(num_shards*num_data)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(idxs, num_data, replace=False))
        idxs = list(set(idxs) - dict_users[i])
    return dict_users

