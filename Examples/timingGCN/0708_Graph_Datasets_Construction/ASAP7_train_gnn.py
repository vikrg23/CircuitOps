#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# SPDX-License-Identifier: Apache-2.0

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.


# In[1]:


### This notebook reads the preprocessed data_train, data_test dataset and uses a modified code of 'train_gnn.py' for training the ASAP7-based datasets
import os
import time
import argparse
import tee
import pickle as pk

import random
import numpy as np
from sklearn.metrics import r2_score

import dgl

import torch
import torch.nn.functional as F
import sys

ROOT_DIR = '/home/vgopal18/Circuitops/examples/timingGCN/0708_Graph_Datasets_Construction/trial/'
sys.path.append(ROOT_DIR)

from model import TimingGCN


# In[2]:


args = argparse.Namespace()
args.netdelay = False
args.celldelay = False
args.groundtruth = False

model = TimingGCN()
model.cuda()

with open(f'{ROOT_DIR}/data_train.pk', 'rb') as pkf:
    data_train = pk.load(pkf)


# In[3]:


def test(model):    # tsrf
    model.eval()
    with torch.no_grad():
        def test_dict(data):
            for k, (g, ts) in data.items():
                torch.cuda.synchronize()
                time_s = time.time()
                pred = model(g, ts, groundtruth=False)[2]
                torch.cuda.synchronize()
                time_t = time.time()
                truth = g.ndata['n_tsrf']
                train_mask = g.ndata['train_mask'].type(torch.bool)
                r2 = r2_score(pred[train_mask].cpu().numpy().reshape(-1),
                              truth[train_mask].cpu().numpy().reshape(-1))
                # notice: there is a typo in the parameter order of r2 calculator.
                # please see https://github.com/TimingPredict/TimingPredict/issues/7.
                # for exact reproducibility of experiments in paper, we will not directly fix the typo here.
                # the experimental conclusions are not affected.
                # r2 = r2_score(pred.cpu().numpy().reshape(-1),
                #               truth.cpu().numpy().reshape(-1))
                print('{:15} r2 {:1.5f}, time {:2.5f}'.format(k, r2, time_t - time_s))
                # print('{}'.format(time_t - time_s + ts['topo_time']))
        print('======= Training dataset ======')
        test_dict(data_train)


#test(model)


# In[4]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
batch_size = 1

loss_tsrf_all = []
for e in range(5000):
    model.train()
    train_loss_tot_tsrf = 0
    optimizer.zero_grad()

    for k, (g, ts) in random.sample(sorted(data_train.items()), batch_size):

        if e < 100:
            _, _, pred_tsrf = model(g, ts, groundtruth=True)
        else:
            _, _, pred_tsrf = model(g, ts, groundtruth=False)

        train_mask = g.ndata['train_mask'].type(torch.bool)
        
        loss_tsrf = F.mse_loss(pred_tsrf[train_mask], g.ndata['n_tsrf'][train_mask])
        train_loss_tot_tsrf += loss_tsrf.item()
        loss_tsrf.backward()

    loss_tsrf_all.append(train_loss_tot_tsrf)
    if e % 10 == 9:
        print(f'Ep {e}, Trn Loss {np.mean(loss_tsrf_all[-10:])}')

    if e % 100 == 0:
        test(model)

    optimizer.step()


import matplotlib.pyplot as plt

plt.plot(loss_tsrf_all[:100], label='train loss (w/ ground truth)')
plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# In[ ]:


#batch_size = 1
#
#loss_tsrf_all_test = []
#for e in range(1):
#    model.train()
#    train_loss_tot_tsrf = 0
#
#    for k, (g, ts) in sorted(data_train.items()):
#        _, _, pred_tsrf = model(g, ts, groundtruth=False)
#        train_mask = g.ndata['train_mask']
#        
#        loss_tsrf = F.mse_loss(pred_tsrf[train_mask], g.ndata['n_tsrf'][train_mask])
#        train_loss_tot_tsrf += loss_tsrf.item()
#
#    loss_tsrf_all_test.append(train_loss_tot_tsrf)
#
#
## In[ ]:
#
#
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
#batch_size = 1
#
#loss_tsrf_all = []
#for e in range(100000):
#    model.train()
#    train_loss_tot_tsrf = 0
#    # train_loss_tot_net_delays, train_loss_tot_cell_delays, train_loss_tot_ats = 0, 0, 0
#    # train_loss_tot_cell_delays_prop, train_loss_tot_ats_prop = 0, 0
#    optimizer.zero_grad()
#
#    for k, (g, ts) in random.sample(sorted(data_train.items()), batch_size):
#        _, _, pred_tsrf = model(g, ts, groundtruth=False)
#
#        train_mask = g.ndata['train_mask']
#
#        # loss_net_delays, loss_cell_delays = 0, 0
#
#        # print(pred_atslew.shape, g)
#
#        # if args.netdelay:
#        #     loss_net_delays = F.mse_loss(pred_net_delays, g.ndata['n_net_delays_log'])
#        #     train_loss_tot_net_delays += loss_net_delays.item()
#
#        # if args.celldelay:
#        #     loss_cell_delays = F.mse_loss(pred_cell_delays, g.edges['cell_out'].data['e_cell_delays'])
#        #     train_loss_tot_cell_delays += loss_cell_delays.item()
#        # else:
#        #     # Workaround for a dgl bug...
#        #     # It seems that if some forward propagation channel is not used in backward graph, the GPU memory would BOOM. so we just create a fake gradient channel for this cell delay fork and make sure it does not contribute to gradient by *0.
#        #     loss_cell_delays = torch.sum(pred_cell_delays) * 0.0
#        
#        loss_tsrf = F.mse_loss(pred_tsrf[train_mask], g.ndata['n_tsrf'][train_mask])
#        train_loss_tot_tsrf += loss_tsrf.item()
#        loss_tsrf.backward()
#
#    loss_tsrf_all.append(train_loss_tot_tsrf)
#    if e % 10 == 9:
#        print(f'Ep {e}, Trn Loss {np.mean(loss_tsrf_all[-10:])}')
#
#    optimizer.step()
#
#
## In[ ]:
#
#
#import matplotlib.pyplot as plt
#
#plt.plot(loss_tsrf_all[:900], label='train loss (w/ ground truth)')
## plt.plot(loss_tsrf_all[:1000], 'train loss (w/ ground truth)')
#plt.yscale('log')
#plt.xlabel('epoch')
#plt.ylabel('loss')


# In[ ]:





# In[ ]:




