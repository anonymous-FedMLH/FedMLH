#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import time
from itertools import islice
import torch
from torch import nn, autograd
from models.Nets import MLP,get_feature_dim,get_hidden_dim1,get_hidden_dim2
import numpy as np
from utils.util import gather_batch
import pdb

def load_lookup(n_class, repetition):
    return np.load('./lookup/b_'+str(n_class)+'/bucket_order_'+str(repetition)+'.npy')

def load_feature_lookup(n_feature):
    return np.load('./lookup/feature_hash/bucket_B_'+str(n_feature)+'.npy')

def train_data_generator_fulldata_hash(files, batch_size, data_index, n_classes, repetition):
    lookup = load_lookup(n_classes, repetition)
    # TODO: add @data_index to select data
    # TODO: check the implementation correctness of this function.
    while 1:
        lines = []
        with open(files, 'r', encoding='utf-8') as f:
            while True:
                temp = len(lines)
                lines += list(islice(f, batch_size - temp))
                if len(lines)!=batch_size:
                    break
                idxs = []
                vals = []
                ##
                y_idxs = []
                y_vals = []
                y_batch = np.zeros([batch_size,n_classes], dtype=float)
                count = 0
                for line in lines:
                    itms = line.strip().split()
                    ##
                    y_idxs = [int(itm) for itm in itms[0].split(',')]
                    y_vals = [1.0 for itm in range(len(y_idxs))]
                    for i in range(len(y_idxs)):
                        y_batch[count, lookup[y_idxs[i]]] = y_vals[i]
                    idxs += [(count,int(itm.split(':')[0])) for itm in itms[1:]]
                    vals += [float(itm.split(':')[1]) for itm in itms[1:]]
                    count += 1
                lines = []
                yield (idxs, vals, y_batch)
                
                

def train_data_generator_hash(files, batch_size, data_index, n_classes, repetition):
    lookup = load_lookup(n_classes, repetition)
    # TODO: add @data_index to select data
    # TODO: check the implementation correctness of this function.
    file_array = []
    lines = []
    with open(files, 'r', encoding='utf-8') as f:
        file_array = f.readlines()

    while 1:
        select_iterator = 0
        while True:
            # select the a batch of the input data
            temp = len(lines)
            more_lines = batch_size - temp
            start_index = select_iterator
            end_index = select_iterator
            if (select_iterator + more_lines) <= (len(data_index) - 1):
                end_index = select_iterator + more_lines
            else:
                end_index = len(data_index) - 1
            select_data_index = data_index[start_index: end_index]
            lines += [file_array[i] for i in select_data_index]
            select_iterator = end_index
            if len(lines) != batch_size:
                break
            idxs = []
            vals = []
            y_idxs = []
            y_vals = []
            y_batch = np.zeros([batch_size, n_classes], dtype=float)
            count = 0
            for line in lines:
                itms = line.strip().split()
                try:
                    y_idxs = [int(itm) for itm in itms[0].split(',')]
                except:
                    y_idxs=[0]
                y_vals = [1.0 for itm in range(len(y_idxs))]
                # print('y_idx: ', y_idxs, 'itms:', itms)
                for i in range(len(y_idxs)):
                    # pdb.set_trace()
                    y_batch[count, lookup[y_idxs[i]]] = y_vals[i]
                idxs += [(count, int(itm.split(':')[0])) for itm in itms[1:]]
                vals += [float(itm.split(':')[1]) for itm in itms[1:]]
                count += 1
            lines = []
            yield (idxs, vals, y_batch)  #

def train_data_generator_nohash(files, batch_size, data_index, n_classes):
    # TODO: add @data_index to select data
    # TODO: check the implementation correctness of this function.
    file_array = []
    lines = []
    with open(files, 'r', encoding='utf-8') as f:
        file_array = f.readlines()
    while 1:
        select_iterator = 0
        while True:
            # select the a batch of the input data
            temp = len(lines)
            more_lines = batch_size - temp
            start_index = select_iterator
            end_index = select_iterator
            if (select_iterator + more_lines) <= (len(data_index) - 1):
                end_index = select_iterator + more_lines
            else:
                end_index = len(data_index) - 1
            select_data_index = data_index[start_index : end_index]
            lines += [file_array[i] for i in select_data_index]
            select_iterator = end_index
            if len(lines) != batch_size:
                break
            idxs = []
            vals = []
            y_idxs = []
            y_vals = []
            y_batch = np.zeros([batch_size, n_classes], dtype=float)
            count = 0
            # pdb.set_trace()
            for line in lines:
                itms = line.strip().split()
                ##
                try:
                    y_idxs = [int(itm) for itm in itms[0].split(',')]
                except:
                    y_idxs=[0]
                # if len(y_idxs) == 0:
                #     print(y_idxs)
                #     exit('Error: No label for this sample')

                for i in range(len(y_idxs)):
                    y_batch[count, y_idxs[i]] = 1.0

                try:
                    idxs += [(count, int(itm.split(':')[0])) for itm in itms[1:]]
                    vals += [float(itm.split(':')[1]) for itm in itms[1:]]
                except:
                    idxs += [(count, 0)]
                    vals += [0.0]                    
                count += 1
            lines = []
            yield (idxs, vals, y_batch)  #


def test_data_generator(files, batch_size):
    while 1:
        lines = []
        with open(files, 'r', encoding='utf-8') as f:
            while True:
                temp = len(lines)
                lines += list(islice(f, batch_size - temp))
                if len(lines) != batch_size:
                    break
                idxs = []
                vals = []
                ##
                labels = []
                count = 0
                for line in lines:
                    ##
                    itms = line.strip().split(' ')
                    ##
                    try:
                        y_idxs = [int(itm) for itm in itms[0].split(',')]
                    except:
                        y_idxs=[0]
                    labels.append(y_idxs)
                    if len(itms) <= 1:
                        print(itms)
                        exit('Error: format of itms is not correct')
                    try:
                        idxs += [(count, int(itm.split(':')[0])) for itm in itms[1:]]
                        vals += [float(itm.split(':')[1]) for itm in itms[1:]]
                        if int(itm.split(':')[0])>= get_feature_dim():
                            exit('Error: index out of range')
                    except:
                        idxs += [(count, 0)]
                        vals += [0.0]
                    count += 1
                lines = []
                yield (idxs, vals, labels) #s
                
                
                
def topN_precision_test_ensemble(net_g, datatest, args):
    N = args.num_classes
    R = len(net_g)
    label_count = np.load('./lookup/label_frequency.npy')
    label_frequent = np.array(range(len(label_count)))[label_count>=args.freq_thres]
    label_tail = np.array(range(len(label_count)))[label_count<args.freq_thres]
    batch_size = args.local_bs
    score_sum = [0.0, 0.0, 0.0]
    score_sum_freq = [0.0, 0.0, 0.0]
    score_sum_tail = [0.0, 0.0, 0.0]
    ldr_test = test_data_generator(datatest, batch_size)

    for i in range(R):
        net_g[i].eval()

    for local_step in range(args.total_test_steps):
        begin_time = time.time()
        log_probs = []
        idxs_batch, vals_batch, labels_batch = next(ldr_test)
        idxs_batch = torch.from_numpy(np.asarray(idxs_batch))
        vals_batch = torch.from_numpy(np.asarray(vals_batch))
        input = torch.sparse.FloatTensor(idxs_batch.t(), vals_batch,
                                         torch.Size([args.local_bs, get_feature_dim()])).to_dense()
        input = input.float().to(args.device)
        for i in range(R):
            inference_result = net_g[i](input)
            log_probs.append(inference_result)

        preds = torch.stack(log_probs)
        # after stack the shape of preds should be: [R, batch_size, N]
        preds = preds.permute(1, 0, 2)
        # after permutation, the shape of preds should be: [batch_size, R, N]
        # average the predicted logits of global models
        preds = torch.mean(preds, 1)
        preds = preds.cpu().detach().numpy()
        top_lbls_1 = np.argmax(preds, axis=-1)
        top_lbls_3 = np.argpartition(preds, -3, axis=-1)[:, -3:]
        top_lbls_5 = np.argpartition(preds, -5, axis=-1)[:, -5:]

        for i in range(batch_size):
            labels_freq = np.intersect1d(labels_batch[i], label_frequent)
            labels_tail = np.intersect1d(labels_batch[i], label_tail)
            #### P@1
            if top_lbls_1[i] in labels_batch[i]:
                score_sum[0] += 1
            if top_lbls_1[i] in labels_freq:
                score_sum_freq[0] += 1 
            if top_lbls_1[i] in labels_tail:
                score_sum_tail[0] += 1
            #### P@3
            score_sum[1] += len(np.intersect1d(top_lbls_3[i], labels_batch[i])) / min(len(labels_batch[i]), 3)
            score_sum_freq[1] += len(np.intersect1d(top_lbls_3[i], labels_freq)) / min(len(labels_batch[i]), 3)
            score_sum_tail[1] += len(np.intersect1d(top_lbls_3[i], labels_tail)) / min(len(labels_batch[i]), 3)
            #### P@5
            score_sum[2] += len(np.intersect1d(top_lbls_5[i], labels_batch[i])) / min(len(labels_batch[i]), 5)
            score_sum_freq[2] += len(np.intersect1d(top_lbls_5[i], labels_freq)) / min(len(labels_batch[i]), 5)
            score_sum_tail[2] += len(np.intersect1d(top_lbls_5[i], labels_tail)) / min(len(labels_batch[i]), 5)

    print('precision@1 for', args.total_test_steps, 'points:', score_sum[0] / (args.total_test_steps * batch_size))
    print('precision@3 for', args.total_test_steps, 'points:', score_sum[1] / (args.total_test_steps * batch_size))
    print('precision@5 for', args.total_test_steps, 'points:', score_sum[2] / (args.total_test_steps * batch_size))
    print('tail precision@1 for', args.total_test_steps, 'points:', score_sum_tail[0] / (args.total_test_steps * batch_size))
    print('tail precision@3 for', args.total_test_steps, 'points:', score_sum_tail[1] / (args.total_test_steps * batch_size))
    print('tail precision@5 for', args.total_test_steps, 'points:', score_sum_tail[2] / (args.total_test_steps * batch_size))
    
    return (np.array(score_sum)/(args.total_test_steps * batch_size), 
            np.array(score_sum_freq)/(args.total_test_steps * batch_size), 
            np.array(score_sum_tail)/(args.total_test_steps * batch_size))



def topN_precision_test_hash(net_g, datatest, args):
    N = args.num_classes
    B = args.B
    R = args.R
    candidates = np.array(range(N))
    lookup = np.empty([R, N], dtype=int)
    for r in range(R):
        lookup[r] = np.load('./lookup/b_' + str(B) + '/bucket_order_' + str(r) + '.npy')
    label_count = np.load('./lookup/label_frequency.npy')
    label_frequent = np.array(range(len(label_count)))[label_count>=args.freq_thres]
    label_tail = np.array(range(len(label_count)))[label_count<args.freq_thres]

    candidate_indices = np.ascontiguousarray(lookup[:, candidates])
    batch_size = args.local_bs
    score_sum = [0.0, 0.0, 0.0]
    score_sum_freq = [0.0, 0.0, 0.0]
    score_sum_tail = [0.0, 0.0, 0.0]
    ldr_test = test_data_generator(datatest, batch_size)

    for i in range(args.R):
        net_g[i].eval()

    for local_step in range(args.total_test_steps):
        begin_time = time.time()
        log_probs = []
        scores = np.zeros((batch_size, N), dtype=np.float32)

        idxs_batch, vals_batch, labels_batch = next(ldr_test)
        idxs_batch = torch.from_numpy(np.asarray(idxs_batch))
        vals_batch = torch.from_numpy(np.asarray(vals_batch))
        input = torch.sparse.FloatTensor(idxs_batch.t(), vals_batch,
                                         torch.Size([args.local_bs, get_feature_dim()])).to_dense()
        input = input.float().to(args.device)
        for i in range(args.R):
            inference_result = net_g[i](input)
            log_probs.append(inference_result)

        preds = torch.stack(log_probs)
        # after stack the shape of preds should be: [R, batch_size, B]
        preds = preds.permute(1, 0, 2)
        preds = preds.cpu().detach().numpy()
        preds = np.ascontiguousarray(preds)
        gather_batch(preds, candidate_indices, scores, R, B, N, batch_size, args.n_threads)
        top_lbls_1 = np.argmax(scores, axis=-1)
        top_lbls_3 = np.argpartition(scores, -3, axis=-1)[:, -3:]
        top_lbls_5 = np.argpartition(scores, -5, axis=-1)[:, -5:]

        for i in range(batch_size):
            labels_freq = np.intersect1d(labels_batch[i], label_frequent)
            labels_tail = np.intersect1d(labels_batch[i], label_tail)
            #### P@1
            if top_lbls_1[i] in labels_batch[i]:
                score_sum[0] += 1
            if top_lbls_1[i] in labels_freq:
                score_sum_freq[0] += 1 
            if top_lbls_1[i] in labels_tail:
                score_sum_tail[0] += 1
            #### P@3
            score_sum[1] += len(np.intersect1d(top_lbls_3[i], labels_batch[i])) / min(len(labels_batch[i]), 3)
            score_sum_freq[1] += len(np.intersect1d(top_lbls_3[i], labels_freq)) / min(len(labels_batch[i]), 3)
            score_sum_tail[1] += len(np.intersect1d(top_lbls_3[i], labels_tail)) / min(len(labels_batch[i]), 3)
            #### P@5
            score_sum[2] += len(np.intersect1d(top_lbls_5[i], labels_batch[i])) / min(len(labels_batch[i]), 5)
            score_sum_freq[2] += len(np.intersect1d(top_lbls_5[i], labels_freq)) / min(len(labels_batch[i]), 5)
            score_sum_tail[2] += len(np.intersect1d(top_lbls_5[i], labels_tail)) / min(len(labels_batch[i]), 5)

    print('precision@1 for', args.total_test_steps, 'points:', score_sum[0] / (args.total_test_steps * batch_size))
    print('precision@3 for', args.total_test_steps, 'points:', score_sum[1] / (args.total_test_steps * batch_size))
    print('precision@5 for', args.total_test_steps, 'points:', score_sum[2] / (args.total_test_steps * batch_size))
    print('tail precision@1 for', args.total_test_steps, 'points:', score_sum_tail[0] / (args.total_test_steps * batch_size))
    print('tail precision@3 for', args.total_test_steps, 'points:', score_sum_tail[1] / (args.total_test_steps * batch_size))
    print('tail precision@5 for', args.total_test_steps, 'points:', score_sum_tail[2] / (args.total_test_steps * batch_size))
    
    return (np.array(score_sum)/(args.total_test_steps * batch_size), 
            np.array(score_sum_freq)/(args.total_test_steps * batch_size), 
            np.array(score_sum_tail)/(args.total_test_steps * batch_size))

def topN_precision_test_nohash(net_g, datatest, args):
    N = args.num_classes
    batch_size = args.local_bs
    score_sum = [0.0, 0.0, 0.0]
    score_sum_freq = [0.0, 0.0, 0.0]
    score_sum_tail = [0.0, 0.0, 0.0]
    label_count = np.load('./lookup/label_frequency.npy')
    label_frequent = np.array(range(len(label_count)))[label_count>=args.freq_thres]
    label_tail = np.array(range(len(label_count)))[label_count<args.freq_thres]
    ldr_test = test_data_generator(datatest, batch_size)
    net_g.eval()
    for local_step in range(args.total_test_steps):
        begin_time = time.time()
        #scores = np.zeros((batch_size, N), dtype=np.float32)
        idxs_batch, vals_batch, labels_batch = next(ldr_test)
        idxs_batch = torch.from_numpy(np.asarray(idxs_batch))
        vals_batch = torch.from_numpy(np.asarray(vals_batch))
        input = torch.sparse.FloatTensor(idxs_batch.t(), vals_batch,
                                         torch.Size([args.local_bs, get_feature_dim()])).to_dense()
        input = input.float().to(args.device)
        inference_result = net_g(input)
        inference_result = inference_result.cpu().detach().numpy()
        top_lbls_1 = np.argmax(inference_result, axis=-1)
        top_lbls_3 = np.argpartition(inference_result, -3, axis=-1)[:, -3:]
        top_lbls_5 = np.argpartition(inference_result, -5, axis=-1)[:, -5:]

        for i in range(batch_size):
            labels_freq = np.intersect1d(labels_batch[i], label_frequent)
            labels_tail = np.intersect1d(labels_batch[i], label_tail)
            #### P@1
            if top_lbls_1[i] in labels_batch[i]:
                score_sum[0] += 1
            if top_lbls_1[i] in labels_freq:
                score_sum_freq[0] += 1 
            if top_lbls_1[i] in labels_tail:
                score_sum_tail[0] += 1
            #### P@3
            score_sum[1] += len(np.intersect1d(top_lbls_3[i], labels_batch[i])) / min(len(labels_batch[i]), 3)
            score_sum_freq[1] += len(np.intersect1d(top_lbls_3[i], labels_freq)) / min(len(labels_batch[i]), 3)
            score_sum_tail[1] += len(np.intersect1d(top_lbls_3[i], labels_tail)) / min(len(labels_batch[i]), 3)
            #### P@5
            score_sum[2] += len(np.intersect1d(top_lbls_5[i], labels_batch[i])) / min(len(labels_batch[i]), 5)
            score_sum_freq[2] += len(np.intersect1d(top_lbls_5[i], labels_freq)) / min(len(labels_batch[i]), 5)
            score_sum_tail[2] += len(np.intersect1d(top_lbls_5[i], labels_tail)) / min(len(labels_batch[i]), 5)
        # print('Test time For step {}: {}'.format(local_step, time.time() - begin_time))

    print('precision@1 for', args.total_test_steps, 'points:', score_sum[0] / (args.total_test_steps * batch_size))
    print('precision@3 for', args.total_test_steps, 'points:', score_sum[1] / (args.total_test_steps * batch_size))
    print('precision@5 for', args.total_test_steps, 'points:', score_sum[2] / (args.total_test_steps * batch_size))
    print('tail precision@1 for', args.total_test_steps, 'points:', score_sum_tail[0] / (args.total_test_steps * batch_size))
    print('tail precision@3 for', args.total_test_steps, 'points:', score_sum_tail[1] / (args.total_test_steps * batch_size))
    print('tail precision@5 for', args.total_test_steps, 'points:', score_sum_tail[2] / (args.total_test_steps * batch_size))

    return (np.array(score_sum)/(args.total_test_steps * batch_size), 
            np.array(score_sum_freq)/(args.total_test_steps * batch_size), 
            np.array(score_sum_tail)/(args.total_test_steps * batch_size))

class LocalUpdate(object):
    def __init__(self, args, dataset_path=None, idxs=None, r=None):
        self.args = args
        if args.reweight_feature:
            feature_weight = torch.from_numpy(np.load(args.feature_weight_loc)).to(args.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight = feature_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.dataset_path = dataset_path
        self.idxs = idxs
        if args.algorithm == 'fedmach':
            self.r = r
            self.ldr_train = train_data_generator_hash(files=self.dataset_path, batch_size=self.args.local_bs, data_index=list(self.idxs),
                                                   n_classes = self.args.B, repetition=self.r)
        if args.algorithm == 'fedavg':
            self.ldr_train = train_data_generator_nohash(files=self.dataset_path, batch_size=self.args.local_bs,
                                                         data_index=list(self.idxs),
                                                         n_classes=self.args.num_classes)

    def train(self, net):
        #TODO: here, the sparse to dense operation may need to be changed
        net.train()
        # train and update
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.0)
        optimizer = torch.optim.Adam(net.parameters())
        epoch_loss = []
        for local_round in range(self.args.local_ep):
            batch_loss = []
            for local_step in range(self.args.step_per_epoch):
                idxs_batch, vals_batch, labels_batch = next(self.ldr_train)
                idxs_batch = torch.from_numpy(np.asarray(idxs_batch))
                vals_batch = torch.from_numpy(np.asarray(vals_batch))
                input = torch.sparse.FloatTensor(idxs_batch.t(), vals_batch, torch.Size([self.args.local_bs, get_feature_dim()])).to_dense()
                output = torch.from_numpy(labels_batch)
                input = input.float().to(self.args.device)
                output = output.float().to(self.args.device)
                net.zero_grad()
                log_probs = net(input)
                loss = self.criterion(log_probs, output)
                loss.backward()
                optimizer.step()
                if self.args.verbose:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, local_step * len(input), len(self.idxs),
                               100. * local_step / len(self.idxs), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)