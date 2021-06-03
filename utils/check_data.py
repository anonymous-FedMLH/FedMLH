#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import time
from itertools import islice
import torch
from torch import nn, autograd
import numpy as np
import pdb

dataset = "../data/Eurlex/train.txt"
num_sample = 15539

def check_data(files,num_sample):
    with open(files, 'r', encoding='utf-8') as f:
        file_array = f.readlines()
    print(len(file_array))
    
    lines = []
    lines += [file_array[i] for i in range(num_sample)]
    count = 0
    for line in lines:
        itms = line.strip().split()
        try:
            y_idxs = [int(itm) for itm in itms[0].split(',')]
        except:
            print(line)

check_data(dataset, num_sample)
        