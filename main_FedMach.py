#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from itertools import islice
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import sys
from torchvision import datasets, transforms
import torch
import argparse
from models.Update import LocalUpdate, topN_precision_test_hash, topN_precision_test_nohash
from models.Nets import MLP, get_feature_dim, get_hidden_dim1, get_hidden_dim2
from models.Fed import FedAvg
from models.test import test_img
from models.Options import args_parser_fedmach
from utils.sampling import wiki10_noniid, amazon_670k_noniid, Eurlex_noniid, Eurlex_iid
'''
    # !!Note!!: WHEN YOU WANT TO change dataset, try to grep "need to be changed when changing dataset" in codebase
'''

if __name__ == '__main__':    
    # parse args
    args = args_parser_fedmach()
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
    print("Total User Number: {:3d} ".format(len(dict_users)))
    print("Selected User Number: {:3d} ".format(max(int(args.frac * args.num_users), 1)))
    print("Total Number of Hash Table: {:3d} ".format(args.R))
    print("Size of one Hash Table: {:3d} ".format(args.B))
    print("Batch Size: {:3d}".format(args.local_bs))

    print('Number of features: {}'.format(get_feature_dim()))
    n_train = len(dict_users[0])
    #n_train = 490449
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
        net_glob = []
        for i in range(args.R):
            model = MLP(dim_in=get_feature_dim(), dim1_hidden=get_hidden_dim1(), dim2_hidden=get_hidden_dim2(), dim_out=args.B)
            model.to(args.device)
            net_glob.append(model)
    else:
        exit('Error: unrecognized model')

    for i in range(args.R):
        print(net_glob[i])
        net_glob[i].train()

    w_glob = []
    # copy weights
    for i in range(args.R):
        w_glob.append(net_glob[i].state_dict())

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    prec_list, prec_freq_list, prec_tail_list = [], [], []

    for iter in range(args.epochs):

        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        dict_models = {i: np.array([]) for i in range(args.R)}

        # Allocate the dict model for different hash tables in different users.
        # Avoid the behavior "push a tensor to array" because this will cause memory leak.
        for i in range(len(idxs_users)):
            for r in range(args.R):
                model = MLP(dim_in=get_feature_dim(), dim1_hidden=get_hidden_dim1(), dim2_hidden=get_hidden_dim2(), dim_out=args.B)
                dict_models[r] = np.append(dict_models[r], copy.deepcopy(model.state_dict()))
                del model

        for i in range(len(idxs_users)):
            for r in range(args.R):
                # print(dataset_train)

                local = LocalUpdate(args=args, dataset_path=dataset_train, idxs=dict_users[idxs_users[i]], r=r)
                import time
                start = time.time()
                w, loss = local.train(net=copy.deepcopy(net_glob[r]).to(args.device))
                end = time.time()
                elapsed_time = end - start
                print("\nSingle Epoch Elapsed time: %.2fsec" % elapsed_time + "\n")

                dict_models[r][i] = copy.deepcopy(w)
                loss_locals.append(copy.deepcopy(loss))
                del w

        # update global weights
        for r in range(args.R):
            w_glob[r] = FedAvg(dict_models[r])

        # copy weight to net_glob
            net_glob[r].load_state_dict(w_glob[r])
            #torch.save(net_glob[r].state_dict(), './saved_models/b_'+str(args.B)+'/hashTable_'+str(r)+'.pt')

        if args.algorithm == 'fedmach':
            prec, prec_freq, prec_tail = topN_precision_test_hash(net_glob, dataset_test, args)
            prec_list.append(prec)
            prec_freq_list.append(prec_freq)
            prec_tail_list.append(prec_tail)
        elif args.algorithm == 'fedavg':
            prec, prec_freq, prec_tail = topN_precision_test_nohash(net_glob, dataset_test, args)
            prec_list.append(prec)
            prec_freq_list.append(prec_freq)
            prec_tail_list.append(prec_tail)

        del dict_models
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
#         print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
    
        # plot loss curve
        if args.num_users > 1:
            plt.figure()
            plt.plot(range(len(loss_train)), loss_train)
            plt.ylabel('train_loss')
            if args.feature_hash:
                plt.savefig('./save/fedmach_{}_{}_{}_C{}_iid{}_b{}_FeatureHash{}_H{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid,args.B, args.feature_hash, args.num_hash_features))
                np.save('./save/fedmach_Eurlex_{}_{}_iid{}_FeatureHash{}_H{}.npy'.format(args.B, args.R, args.iid, args.feature_hash, args.num_hash_features),np.array(prec_list))
                np.save('./save/fedmach_Eurlex_{}_{}_iid{}_FeatureHash{}_H{}_frequent.npy'.format(args.B, args.R, args.iid, args.feature_hash, args.num_hash_features), np.array(prec_freq_list))
                np.save('./save/fedmach_Eurlex_{}_{}_iid{}_FeatureHash{}_H{}_tail.npy'.format(args.B, args.R, args.iid, args.feature_hash, args.num_hash_features), np.array(prec_tail_list))
            else:
                plt.savefig('./save/fedmach_{}_{}_{}_C{}_iid{}_b{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid,args.B))
                np.save('./save/fedmach_Eurlex_{}_{}_iid{}.npy'.format(args.B, args.R, args.iid),np.array(prec_list))
                np.save('./save/fedmach_Eurlex_{}_{}_iid{}_frequent.npy'.format(args.B, args.R, args.iid), np.array(prec_freq_list))
                np.save('./save/fedmach_Eurlex_{}_{}_iid{}_tail.npy'.format(args.B, args.R, args.iid), np.array(prec_tail_list))

            plt.figure()
            plt.plot(range(len(prec_list)), np.array(prec_list)[:,0], 'r', label='@1')
            plt.plot(range(len(prec_list)), np.array(prec_list)[:,1], 'y', label='@2')
            plt.plot(range(len(prec_list)), np.array(prec_list)[:,2], 'b', label='@3')
            plt.ylabel('test_acc')
            plt.legend()
            if args.feature_hash:
                plt.savefig('./save/fedmach_{}_{}_{}_C{}_iid{}_b{}_FeatureHash{}_H{}_test_acc.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid,args.B, args.feature_hash, args.num_hash_features))
            else:
                plt.savefig('./save/fedmach_{}_{}_{}_C{}_iid{}_b{}_test_acc.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid,args.B))
        else:
            prec_list=np.array(prec_list)
            np.save('./save/MACH_Eurlex_iid{}.npy'.format(args.iid), prec_list)



