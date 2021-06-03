#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn
import torch.nn.functional as F
import argparse
from .Options import args_parser_fedavg

args = args_parser_fedavg()

#TODO: need to be changed when changing dataset
# def get_hash_feature_dim():
#     feature_dim = 1000
#     return feature_dim

def get_feature_dim():
    if args.feature_hash:
        feature_dim = args.num_hash_features
    else:
        feature_dim = 5000
    return feature_dim

def get_hidden_dim1():
    hidden_dim1 = args.dim1
    return hidden_dim1

def get_hidden_dim2():
    hidden_dim2 = args.dim2
    return hidden_dim2

# TODO: cannot be called by fedMach
def get_output_dim():
    output_dim = 3993
    return output_dim

class MLP(nn.Module):
    #TODO: extend this to a three layer MLP
    def __init__(self, dim_in, dim1_hidden, dim2_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim1_hidden, bias=True)
        self.relu_input = nn.ReLU()
        self.layer_hidden = nn.Linear(dim1_hidden, dim2_hidden, bias=True)
        self.relu_hidden = nn.ReLU()
        self.layer_output = nn.Linear(dim2_hidden, dim_out, bias=True)

    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu_input(x)
        x = self.layer_hidden(x)
        x = self.relu_hidden(x)
        x = self.layer_output(x)
        return x
