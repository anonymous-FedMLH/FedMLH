#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from itertools import islice
import matplotlib

from models.Options import args_parser_fedavg
from utils.sampling import amazon_670k_noniid, wiki10_noniid, Eurlex_noniid, Eurlex_iid

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import random
import argparse
from models.Update import LocalUpdate, topN_precision_test_hash, topN_precision_test_nohash, topN_precision_test_ensemble
from models.Nets import MLP,get_feature_dim,get_hidden_dim1,get_hidden_dim2,get_output_dim
from models.Fed import FedAvg
from models.test import test_img

import pdb
'''
    # !!Note!!: WHEN YOU WANT TO change dataset, try to grep "need to be changed when changing dataset" in codebase
'''

if __name__ == '__main__':
    # parse args
    args = args_parser_fedavg()
    # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.device = torch.device('cuda:{}'.format(args.gpu))
    # load dataset and split users
    dict_users = []
    if args.dataset == 'Eurlex':
        # TODO: need to be changed when changing dataset
        if args.feature_hash:
            dataset_train = "./data/Eurlex/train_feature_hash_{}.txt".format(args.num_hash_features)
            dataset_test = "./data/Eurlex/test_feature_hash_{}.txt".format(args.num_hash_features)
        else:
            dataset_train = "./data/Eurlex/train.txt"
            dataset_test = "./data/Eurlex/test.txt"
        if args.iid:
            dict_users = Eurlex_iid(dataset_train, args.num_users)
        else:
            dict_users = Eurlex_noniid(dataset_train, args.num_users)
        # Remove the heading from the dataset_train
        with open(dataset_train, 'r', encoding='utf-8') as f:
            file_array = f.readlines()
        print('Number of lines:', len(file_array))
        try: 
            if int(file_array[0].strip().split()[0]) == 15539:
                with open(dataset_train, 'w', encoding='utf-8') as f:
                    f.writelines(file_array[1:])
        except:
            print('Dataset heading is removed')
        # Remove the heading from the dataset_test
        with open(dataset_test, 'r', encoding='utf-8') as f:
            file_array = f.readlines()
        print('Number of lines:', len(file_array))
        try: 
            if int(file_array[0].strip().split()[0]) == 3809:
                with open(dataset_test, 'w', encoding='utf-8') as f:
                    f.writelines(file_array[1:])
        except:
            print('Dataset heading is removed')
    elif args.dataset == 'Amazon-670K':
        # TODO: need to be changed when changing dataset
        dataset_train = "./data/Amazon-670K/train.txt"
        dataset_test = "./data/Amazon-670K/test.txt"
        dict_users = amazon_670k_noniid(dataset_train, args.num_users)
    elif args.dataset == 'Wiki10':
        # TODO: need to be changed when changing dataset
        dataset_train = "./data/Wiki10/train.txt"
        dataset_test = "./data/Wiki10/test.txt"
        dict_users = wiki10_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')

    print("Training dataset: ", dataset_train)
    print("Test dataset: ", dataset_test)
    print("Samples are iid partitioned: {}".format(args.iid))
    print("Ensemble the global models: {}".format(args.model_avg))
    print("Total User Number: {:3d} ".format(len(dict_users)))
    print("Selected User Number: {:3d} ".format(max(int(args.frac * args.num_users), 1)))
    #print("Total Number of Hash Table: {:3d} ".format(args.R))
    #print("Size of one Hash Table: {:3d} ".format(args.B))
    print("Batch Size: {:3d}".format(args.local_bs))

    print('Number of features: {}'.format(get_feature_dim()))
    n_train = len(dict_users[0])
    print("N_Train: {:3d} ".format(n_train))
    n_train_steps_per_epoch = n_train // args.local_bs
    print("N_steps_per_epoch: {:3d} ".format(n_train_steps_per_epoch))
    args.step_per_epoch = n_train_steps_per_epoch
    # make sure step_per_epoch has been changed.
    assert args.step_per_epoch!=1

    n_test = args.n_test
    print("N_Test: {:3d} ".format(n_test))
    n_test_steps = n_test // args.local_bs
    print("N_steps_per_epoch: {:3d} ".format(n_test_steps))
    args.total_test_steps = n_test_steps
    # make sure total test steps has been changed
    assert args.total_test_steps!=1

    # build model
    if args.model == 'mlp':
        # TODO: https://discuss.pytorch.org/t/can-i-store-multiple-models-in-a-normal-python-dictionary/42854
        if args.model_avg:
            net_glob_list = []
            for i in range(args.num_model_avg):
                random.seed(100*i)
                net_glob = MLP(dim_in=get_feature_dim(), dim1_hidden=get_hidden_dim1(), dim2_hidden=get_hidden_dim2(), dim_out=get_output_dim())
                net_glob.to(args.device)
                net_glob_list += [net_glob]
            net_glob_ensem = MLP(dim_in=get_feature_dim(), dim1_hidden=get_hidden_dim1(), dim2_hidden=get_hidden_dim2(), dim_out=get_output_dim())
            net_glob_ensem.to(args.device)
        else:
            net_glob = MLP(dim_in=get_feature_dim(), dim1_hidden=get_hidden_dim1(), dim2_hidden=get_hidden_dim2(), dim_out=get_output_dim())
            net_glob.to(args.device)
    else:
        exit('Error: unrecognized model')

    # print(net_glob)
    if args.model_avg:
        w_glob_list = []
        for net_glob in net_glob_list:
            net_glob.train()
            w_glob_list += [net_glob.state_dict()]
    else:
        net_glob.train()
        w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    prec_list, prec_freq_list, prec_tail_list = [], [], []

    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        if args.model_avg:
            w_locals, loss_locals = [[] for _ in range(args.num_model_avg)], [[] for _ in range(args.num_model_avg)]
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for i in range(len(idxs_users)):
            if args.model_avg:
                for j in range(args.num_model_avg):
                    local = LocalUpdate(args=args, dataset_path=dataset_train, idxs=dict_users[idxs_users[i]])
                    import time
                    start = time.time()
                    w, loss = local.train(net=copy.deepcopy(net_glob_list[j]).to(args.device))
                    end = time.time()
                    elapsed_time = end - start
                    print("\nSingle Epoch Elapsed time: %.2fsec" % elapsed_time + "\n")
                    w_locals[j].append(copy.deepcopy(w))
                    loss_locals[j].append(copy.deepcopy(loss)) 
            else:
                
                local = LocalUpdate(args=args, dataset_path=dataset_train, idxs=dict_users[idxs_users[i]])

                import time
                start = time.time()
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                end = time.time()
                elapsed_time = end - start
                print("\nSingle Epoch Elapsed time: %.2fsec" % elapsed_time + "\n")
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))

        if args.model_avg:
            for j in range(args.num_model_avg):
                # update global weights
                w_glob_list[j] = FedAvg(w_locals[j])
                # copy weight to net_glob
                net_glob_list[j].load_state_dict(w_glob_list[j])
            # ensemble the global nets
            w_glob_ensem = FedAvg(w_glob_list)
            net_glob_ensem.load_state_dict(w_glob_ensem)
        else:
            # update global weights
            w_glob = FedAvg(w_locals)
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

        if args.algorithm == 'fedmach':
            prec, prec_freq, prec_tail = topN_precision_test_hash(net_glob, dataset_test, args)
            prec_list.append(prec)
            prec_freq_list.append(prec_freq)
            prec_tail_list.append(prec_tail)
        elif args.algorithm == 'fedavg':
            if args.model_avg:
                prec, prec_freq, prec_tail = topN_precision_test_ensemble(net_glob_list, dataset_test, args)
            else:
                prec, prec_freq, prec_tail = topN_precision_test_nohash(net_glob, dataset_test, args)
            prec_list.append(prec)
            prec_freq_list.append(prec_freq)
            prec_tail_list.append(prec_tail)

        # print loss
        if args.model_avg:
            for j in range(len(loss_locals)):
                loss_avg = sum(loss_locals[j]) / (len(loss_locals) * args.num_model_avg)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)            
        else:
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)

        # plot loss curve
        if args.num_users > 1:
            plt.figure()
            plt.plot(range(len(loss_train)), loss_train)
            plt.ylabel('train_loss')
            if args.feature_hash:
                if args.dim1 > 1000:
                    plt.savefig('./save/fedavg_{}_{}_{}_C{}_iid{}_wide_ensemble{}_R{}_FeatureHash{}_H{}_reweight{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.model_avg, args.num_model_avg, args.feature_hash, args.num_hash_features, args.reweight_feature))                
                    np.save('./save/fedavg_Eurlex_iid{}_wide_ensemble{}_R{}_FeatureHash{}_H{}_reweight{}.npy'.format(args.iid, args.model_avg, args.num_model_avg, args.feature_hash, args.num_hash_features, args.reweight_feature), np.array(prec_list))
                    np.save('./save/fedavg_Eurlex_iid{}_wide_ensemble{}_R{}_FeatureHash{}_H{}_reweight{}_frequent.npy'.format(args.iid, args.model_avg, args.num_model_avg, args.feature_hash, args.num_hash_features, args.reweight_feature), np.array(prec_freq_list))
                    np.save('./save/fedavg_Eurlex_iid{}_wide_ensemble{}_R{}_FeatureHash{}_H{}_reweight{}_tail.npy'.format(args.iid, args.model_avg, args.num_model_avg, args.feature_hash, args.num_hash_features, args.reweight_feature), np.array(prec_tail_list))
                else:
                    plt.savefig('./save/fedavg_{}_{}_{}_C{}_iid{}_ensemble{}_R{}_FeatureHash{}_H{}_reweight{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.model_avg, args.num_model_avg, args.feature_hash, args.num_hash_features, args.reweight_feature))                
                    np.save('./save/fedavg_Eurlex_iid{}_ensemble{}_R{}_FeatureHash{}_H{}_reweight{}.npy'.format(args.iid, args.model_avg, args.num_model_avg, args.feature_hash, args.num_hash_features, args.reweight_feature), np.array(prec_list))
                    np.save('./save/fedavg_Eurlex_iid{}_ensemble{}_R{}_FeatureHash{}_H{}_reweight{}_frequent.npy'.format(args.iid, args.model_avg, args.num_model_avg, args.feature_hash, args.num_hash_features, args.reweight_feature), np.array(prec_freq_list))
                    np.save('./save/fedavg_Eurlex_iid{}_ensemble{}_R{}_FeatureHash{}_H{}_reweight{}_tail.npy'.format(args.iid, args.model_avg, args.num_model_avg, args.feature_hash, args.num_hash_features, args.reweight_feature), np.array(prec_tail_list))
            else:
                if args.dim1 > 1000:
                    plt.savefig('./save/fedavg_{}_{}_{}_C{}_iid{}_wide_ensemble{}_R{}_reweight{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.model_avg, args.num_model_avg, args.reweight_feature))               
                    np.save('./save/fedavg_Eurlex_iid{}_wide_ensemble{}_R{}_reweight{}.npy'.format(args.iid, args.model_avg, args.num_model_avg, args.reweight_feature), np.array(prec_list))
                    np.save('./save/fedavg_Eurlex_iid{}_wide_ensemble{}_R{}_reweight{}_frequent.npy'.format(args.iid, args.model_avg, args.num_model_avg, args.reweight_feature), np.array(prec_freq_list))
                    np.save('./save/fedavg_Eurlex_iid{}_wide_ensemble{}_R{}_reweight{}_tail.npy'.format(args.iid, args.model_avg, args.num_model_avg, args.reweight_feature), np.array(prec_tail_list))
                else:
                    plt.savefig('./save/fedavg_{}_{}_{}_C{}_iid{}_ensemble{}_R{}_reweight{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.model_avg, args.num_model_avg, args.reweight_feature))               
                    np.save('./save/fedavg_Eurlex_iid{}_ensemble{}_R{}_reweight{}.npy'.format(args.iid, args.model_avg, args.num_model_avg, args.reweight_feature), np.array(prec_list))
                    np.save('./save/fedavg_Eurlex_iid{}_ensemble{}_R{}_reweight{}_frequent.npy'.format(args.iid, args.model_avg, args.num_model_avg, args.reweight_feature), np.array(prec_freq_list))
                    np.save('./save/fedavg_Eurlex_iid{}_ensemble{}_R{}_reweight{}_tail.npy'.format(args.iid, args.model_avg, args.num_model_avg, args.reweight_feature), np.array(prec_tail_list))


            plt.figure()
            plt.plot(range(len(prec_list)), np.array(prec_list)[:,0], 'r', label='@1')
            plt.plot(range(len(prec_list)), np.array(prec_list)[:,1], 'y', label='@2')
            plt.plot(range(len(prec_list)), np.array(prec_list)[:,2], 'b', label='@3')
            plt.ylabel('test_acc')
            plt.legend()
            if args.feature_hash:
                if args.dim1 > 1000:
                    plt.savefig('./save/fedavg_{}_{}_{}_C{}_iid{}_wide_ensemble{}_R{}_FeatureHash{}_H{}_reweight{}_test_acc.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.model_avg, args.num_model_avg, args.feature_hash, args.num_hash_features, args.reweight_feature))
                else:
                    plt.savefig('./save/fedavg_{}_{}_{}_C{}_iid{}_ensemble{}_R{}_FeatureHash{}_H{}_reweight{}_test_acc.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.model_avg, args.num_model_avg, args.feature_hash, args.num_hash_features, args.reweight_feature))
            else:
                if args.dim1 > 1000:
                    plt.savefig('./save/fedavg_{}_{}_{}_C{}_iid{}_wide_ensemble{}_R{}_reweight{}_test_acc.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.model_avg, args.num_model_avg, args.reweight_feature))
                else:
                    plt.savefig('./save/fedavg_{}_{}_{}_C{}_iid{}_ensemble{}_R{}_reweight{}_test_acc.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.model_avg, args.num_model_avg, args.reweight_feature))

        else:
            prec_list=np.array(prec_list)
            np.save('./save/standard_Eurlex_iid{}.npy'.format(args.iid), prec_list)
