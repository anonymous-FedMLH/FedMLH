# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from itertools import islice
from math import floor
import numpy as np
from torchvision import datasets
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--write_loc", default='../lookup', type=str)
parser.add_argument("--dataset", default="../data/Eurlex/train.txt", type=str)
args = parser.parse_args()

def count_frequency(dataset):
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
    num_labels = 3993
    num_training_data = 15539
    file_array = []
    labels_count = np.zeros(num_labels, dtype=float)
    with open(dataset, 'r', encoding='utf-8') as f:
        file_array = f.readlines()
        for line in file_array:
            itms = line.strip().split(' ')
            try:
                this_label = []
                for itm in itms[0].split(','):
                    this_label.append(int(itm))
                    labels_count[int(itm)] = labels_count[int(itm)] + 1/num_training_data
            except:
                continue
    np.save(args.write_loc+'/label_frequency'+'.npy', labels_count)
    return None


count_frequency(args.dataset)