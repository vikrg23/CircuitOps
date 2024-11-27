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


### This notebook reads the dgl graph and does preprocessing according to data_graph.py for the model to read
import os, sys
ROOT_DIR = '/home/vgopal18/Circuitops/examples/timingGCN/0708_Graph_Datasets_Construction/trial/'
sys.path.append(ROOT_DIR)
import time

import pickle as pk

import numpy as np
import pandas as pd

import dgl
import networkx as nx
# import graph_tool as gt
# from graph_tool.all import *

import torch

import matplotlib.pyplot as plt


# In[2]:


## load all datasets in design_names = ['NV_NVDLA_partition_m', 'NV_NVDLA_partition_p', 'ariane136', 'mempool_tile_wrap']
design_names = ['gcd']
dataset_dir = '0820_v1'
testing_designs = ['gcd']

# read all the graph for all the designs
gs = {}
for design_name in design_names:
    design_dir = f'{ROOT_DIR}/'
    gs[design_name] = dgl.load_graphs(f'{design_dir}/graph.dgl')[0]

# identical function taken from data_graph.py
def gen_topo(g_hetero):
    torch.cuda.synchronize()
    time_s = time.time()
    na, nb = g_hetero.edges(etype='net_out', form='uv')
    ca, cb = g_hetero.edges(etype='cell_out', form='uv')
    g = dgl.graph((torch.cat([na, ca]).cpu(), torch.cat([nb, cb]).cpu()))
    topo = dgl.topological_nodes_generator(g)
    ### inspect the topography!
    g.ndata['fanout'] = g_hetero.ndata['nf'][:, 1].cpu()
    for li, nodes in enumerate(topo):
        # print(f'level {li}, # nodes = {len(nodes)}')
        # print(g.ndata['fanout'][nodes.numpy()])
        assert (li % 2 == 0 and (g.ndata['fanout'][nodes] == 0).sum() == 0) or (li % 2 == 1 and (g.ndata['fanout'][nodes] == 1).sum() == 0)
    assert len(topo) % 2 == 0
    ret = [t.cuda() for t in topo]
    torch.cuda.synchronize()
    time_e = time.time()
    return ret, time_e - time_s


# data preprocessing
new_gs = {}
for design_name, des_gs in gs.items():
    new_gs[design_name] = []
    for gi, g in enumerate(des_gs):
        g.nodes['node'].data['nf'] = g.nodes['node'].data['nf'].type(torch.float32)
        g.edges['cell_out'].data['ef'] = g.edges['cell_out'].data['ef'].type(torch.float32)
        g.edges['net_out'].data['ef'] = g.edges['net_out'].data['ef'].type(torch.float32)
        g.edges['net_in'].data['ef'] = g.edges['net_in'].data['ef'].type(torch.float32)
        g.ndata['pin_slack'][g.ndata['pin_slack'] > 1e10] = torch.nan
        g.ndata['pin_rise_arr'][g.ndata['pin_rise_arr'] < -1e10] = torch.nan
        g.ndata['pin_fall_arr'][g.ndata['pin_fall_arr'] < -1e10] = torch.nan
        g.ndata['n_tsrf'] = torch.stack([g.ndata['pin_tran'], g.ndata['pin_slack'], g.ndata['pin_rise_arr'], g.ndata['pin_fall_arr']], axis=1).type(torch.float32)
        # print(f'{design_name}, {gi+1}/{len(des_gs)}')
        topo, topo_time = gen_topo(g)
        ts = {'input_nodes': (g.ndata['nf'][:, 1] < 0.5).nonzero().flatten().type(torch.int32),
            'output_nodes': (g.ndata['nf'][:, 1] > 0.5).nonzero().flatten().type(torch.int32),
            'output_nodes_nonpi': torch.logical_and(g.ndata['nf'][:, 1] > 0.5, g.ndata['nf'][:, 0] < 0.5).nonzero().flatten().type(torch.int32),
            'pi_nodes': torch.logical_and(g.ndata['nf'][:, 1] > 0.5, g.ndata['nf'][:, 0] > 0.5).nonzero().flatten().type(torch.int32),
            # 'po_nodes': torch.logical_and(g.ndata['nf'][:, 1] < 0.5, g.ndata['nf'][:, 0] > 0.5).nonzero().flatten().type(torch.int32),
            # 'endpoints': (g.ndata['n_is_end'] > 0.5).nonzero().flatten().type(torch.long),
            'topo': [t.cpu() for t in topo]}
            # 'topo_time': topo_time}
        for key in ts.keys():
            if type(ts[key]) == torch.Tensor:
                ts[key] = ts[key].cpu().numpy()
            else:
                ts[key] = [item.cpu().numpy() for item in ts[key]]
        new_gs[design_name].append([g, ts])
        
# get means and stds
nf, n_tsrf, ef_pnop, ef_pcop = [], [], [], []
for design_name, des_gs in new_gs.items():
    if design_name in testing_designs:
        print(f'Skipped {design_name} for mean,std calculation')
    for gi, g_n_ts in enumerate(des_gs):
        g, ts = g_n_ts
        # get means and stds
        nf.append(g.ndata['nf'])
        # n_tsrf.append(g.ndata['n_tsrf'])
        ef_pnop.append(g.edata['ef'][('node', 'net_out', 'node')])
        ef_pcop.append(g.edata['ef'][('node', 'cell_out', 'node')])


nf = torch.cat(nf, axis=0)
# n_tsrf = torch.cat(n_tsrf, axis=0)
ef_pnop = torch.cat(ef_pnop, axis=0)
ef_pcop = torch.cat(ef_pcop, axis=0)

nf_mean = torch.nanmean(nf, dim=0)
nf_std = (torch.nanmean(nf ** 2, dim=0) - nf_mean ** 2) ** 0.5
# n_tsrf_mean = torch.nanmean(n_tsrf, dim=0)
# n_tsrf_std = (torch.nanmean(n_tsrf ** 2, dim=0) - n_tsrf_mean ** 2) ** 0.5
ef_pnop_mean = torch.nanmean(ef_pnop, dim=0)
ef_pnop_std = (torch.nanmean(ef_pnop ** 2, dim=0) - ef_pnop_mean ** 2) ** 0.5
ef_pcop_mean = torch.nanmean(ef_pcop, dim=0)
ef_pcop_std = (torch.nanmean(ef_pcop ** 2, dim=0) - ef_pcop_mean ** 2) ** 0.5
ef_pcop_std[torch.isnan(ef_pcop_std)] = 0
drop_ef_pcop_cols = ef_pcop_std == 0
print('There are', (ef_pcop_std == 0).sum(), 'all-the-same pin-cell-pin attributes')
ef_pcop_std[ef_pcop_std == 0] = ef_pcop_mean[ef_pcop_std == 0]

data = {}
for design_name, des_gs in new_gs.items():
    for gi, g_n_ts in enumerate(des_gs):
        g, ts = g_n_ts
        topo = ts['topo']
        ## normalize
        g.ndata['nf'][:, :2] = g.ndata['nf'][:, :2] / nf_std[:2].unsqueeze(0)
        g.ndata['nf'][:, 2:] = (g.ndata['nf'][:, 2:] - nf_mean[2:].unsqueeze(0)) / nf_std[2:].unsqueeze(0)

        # self normalize for the answers
        n_tsrf_mean = torch.nanmean(g.ndata['n_tsrf'], dim=0)
        n_tsrf_std = (torch.nanmean(g.ndata['n_tsrf'] ** 2, dim=0) - n_tsrf_mean ** 2) ** 0.5
        g.ndata['n_tsrf'] /= n_tsrf_std.unsqueeze(0)

        g.edata['ef'] = {
            ('node', 'net_out', 'node') : g.edata['ef'][('node', 'net_out', 'node')] / ef_pnop_std.unsqueeze(0),
            ('node', 'net_in', 'node')  : g.edata['ef'][('node', 'net_in', 'node')] / ef_pnop_std.unsqueeze(0),
            ('node', 'cell_out', 'node'): g.edata['ef'][('node', 'cell_out', 'node')] / ef_pcop_std.unsqueeze(0)
        }

        # train mask must not contain nans
        assert (torch.isnan(g.ndata['nf'].any(dim=-1)) & g.ndata['train_mask']).sum() == 0
        assert (torch.isnan(g.ndata['n_tsrf'].any(dim=-1)) & g.ndata['train_mask']).sum() == 0

        # set nans to zero
        g.ndata['nf'][torch.isnan(g.ndata['nf'])] = 0
        g.ndata['n_tsrf'][torch.isnan(g.ndata['n_tsrf'])] = 0
        
        data[(design_name, gi)] = g, ts

        # some assertions
        assert ts['input_nodes'].max() < len(g.ndata['nf']), f"{ts['input_nodes'].max()}, {g.ndata['nf'].shape}"
        assert ts['output_nodes'].max() < len(g.ndata['nf']), f"{ts['output_nodes'].max()}, {g.ndata['nf'].shape}"
        assert ts['output_nodes_nonpi'].max() < len(g.ndata['nf']), f"{ts['output_nodes_nonpi'].max()}, {g.ndata['nf'].shape}"
        #assert ts['pi_nodes'].max() < len(g.ndata['nf']), f"{ts['pi_nodes'].max()}, {g.ndata['nf'].shape}"

        # just for report
        print(gi, design_name, len(g.ndata['nf']), g.ndata['train_mask'].sum().item(), len(g.edata['ef'][('node', 'cell_out', 'node')]), len(g.edata['ef'][('node', 'net_out', 'node')]), len(topo))

data_train = {k: t for k, t in data.items() if k[0] not in testing_designs}
data_test = {k: t for k, t in data.items() if k[0] in testing_designs}

# normalize LUT props
with open(f'{ROOT_DIR}/LUT_prop.pk', 'rb') as pkf:
    LUT_prop = pk.load(pkf)
for key in LUT_prop.keys():
    LUT_prop[key] /= ef_pcop_std.numpy()
with open(f'{ROOT_DIR}/LUT_prop_normed.pk', 'wb') as pkf:
    pk.dump(LUT_prop, pkf)

# normalize rise fall cap props
rf_cap = pd.read_csv(f'{ROOT_DIR}/rise_fall_caps.csv', index_col=0)
rf_cap = (rf_cap - nf_mean[-2:].unsqueeze(0).numpy()) / nf_std[-2:].unsqueeze(0).numpy()
rf_cap.to_csv(f'{ROOT_DIR}/rise_fall_caps_normed.csv')


# In[ ]:





# In[3]:


print('[node data] ( = dstdata)')
for nkey, ndat in g.ndata.items():
    assert type(ndat) == torch.Tensor, 'Type must be torch.Tensor'
    print(f'{nkey:22s} {ndat.shape}')
    if nkey == 'nf':
        nf = ndat
        for fkey, frange in [('is_prim IO', [0,1]), ('fanout(1) or in(0)', [1,2]), ('dis to tlrb', [2,6]), ('RF cap', [6,8])]:
            print(f'  {fkey:20s} {ndat[:, frange[0]:frange[1]].shape}')
print()

print('[edge data]')
for ekey, edat in g.edata.items():
    assert type(edat) == dict, 'Type must be dict'
    print(f'{ekey}:')
    for edat_key, edat_dat in edat.items():
        print(f'  {f"{edat_key}":30s} {edat_dat.shape}')


# In[4]:


with open(f'{ROOT_DIR}/project/TimingGCN_pl/data/{dataset_dir}/data_train.pk', 'wb') as pkf:
    pk.dump(data_train, pkf)
with open(f'{ROOT_DIR}/project/TimingGCN_pl/data/{dataset_dir}/data_test.pk',  'wb') as pkf:
    pk.dump(data_test, pkf)


# In[5]:





# In[14]:


# distribution of the variables in data train
for des, (g, ts) in data_test.items():
    print([len(topo )for topo in ts['topo']])


# In[ ]:





# In[ ]:





# In[ ]:




