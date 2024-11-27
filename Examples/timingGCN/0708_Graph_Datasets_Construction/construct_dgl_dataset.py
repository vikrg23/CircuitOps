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


### This notebook constructs TimingGCN compatible dgl-based datasets from the ASAP7/CircuitOps design datasets
import os, sys
ROOT_DIR = '/home/vgopal18/Circuitops/examples/timingGCN/0708_Graph_Datasets_Construction/trial/'
sys.path.append(ROOT_DIR)

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
dataset_dir = '0709_v1'

#load datasets
for design_name in design_names:
    design_dir = f'{ROOT_DIR}/'
    with open(f'{design_dir}/LUTs.pk', 'rb') as pkf:
        LUTs = pk.load(pkf)
    with open(f'{ROOT_DIR}/BUFF_LUTs.pk', 'rb') as pkf:
        BUFF_LUTs = pk.load(pkf)
    pin_df = pd.read_csv(f'{design_dir}/pin_df.csv', index_col=0)
    pin_pin_df = pd.read_csv(f'{design_dir}/pin_pin_df.csv', index_col=0)
    net_df = pd.read_csv(f'{design_dir}/net_df.csv', index_col=0)
    fo4_df = pd.read_csv(f'{ROOT_DIR}/fo4_df.csv', index_col=0)
    # assert there are no isolated pins
    edge_pin_ids = np.array(sorted(list(set(pin_pin_df['src_id']).union(set(pin_pin_df['tar_id'])))))
    assert edge_pin_ids.shape[0] == len(pin_df) and edge_pin_ids[-1] + 1 == len(pin_df)

### create the features for the graph
# pin input variables
is_prim_IO = pin_df[['is_port']].values.astype(float)
is_fanout = (pin_df[['dir']] == 0).astype(float)
rel_pos = pin_df[['to_top', 'to_left', 'to_right', 'to_bottom']].values.astype(float)
cap_info = pin_df[['rise_cap', 'fall_cap']].values.astype(float)
node_features = np.concatenate((is_prim_IO, is_fanout, rel_pos, cap_info), axis=1)

# create LUTs and corresponding key for mapping the cell variables
LUTs_flat  = {}
for key, LUT in LUTs.items():
    LUT_flat = LUT['LUTidx'].reshape(4, -1)
    # LUT_flat = np.log(LUT['LUTidx']).reshape(4, -1)                                     
    LUT_flat = np.concatenate((np.ones((4, 1)), LUT_flat), axis=1).flatten()
    LUT_flat = np.concatenate((LUT_flat, LUT['LUTmat'].flatten()))
    # LUT_flat = np.concatenate((LUT_flat, np.log(LUT['LUTmat']).flatten()))              
    LUTs_flat[key] = LUT_flat


pin_cell_pin_df = pin_pin_df[pin_pin_df['is_net'] == 0]
cell_edge_LUT_keys = [(ref, src, tar) for ref, src, tar in zip(pin_cell_pin_df['libcell_name'], pin_cell_pin_df['src_pin_name'], pin_cell_pin_df['tar_pin_name'])]
assert False not in [key in LUTs_flat.keys() for key in cell_edge_LUT_keys]

# cell edge input variables
LUTs_map = pd.Series([val for _, val in LUTs_flat.items()], LUTs_flat.keys())
cell_edge_info = np.array(list(pd.Series(cell_edge_LUT_keys).map(LUTs_map)))

# net edge input variables
pin_net_pin_df = pin_pin_df[pin_pin_df['is_net'] == 1].copy()
pin_net_pin_df['net_name'] = pin_net_pin_df['src_id'].map(pd.Series(pin_df['net_name'], index=pin_df['id']))
pin_x_map = pd.Series(pin_df['x'], pin_df['id'])
pin_y_map = pd.Series(pin_df['y'], pin_df['id'])
src_x = pin_net_pin_df['src_id'].map(pin_x_map)
src_y = pin_net_pin_df['src_id'].map(pin_y_map)
tar_x = pin_net_pin_df['tar_id'].map(pin_x_map)
tar_y = pin_net_pin_df['tar_id'].map(pin_y_map)

# append total_cap, net_cap, net_res to net info
net_indiv_info = []
for net_attrs in ['total_cap', 'net_cap', 'net_res']:
    net_map = pd.Series(net_df[net_attrs].values, index=net_df['net_name'])
    net_indiv_info.append(pin_net_pin_df['net_name'].map(net_map).values)


net_indiv_info = np.stack(net_indiv_info, axis=1)

net_out_edge_info = np.stack(((tar_x - src_x).values, (tar_y - src_y).values), axis=1)
net_out_edge_info = np.concatenate((net_indiv_info, net_out_edge_info), axis=1)

# individually make some values reasonable
net_out_edge_info[:, 0] *= 1e15
net_out_edge_info[:, 2] *= 1e-3
net_out_edge_info[:, 3:] *= 1e-3

#net_out_edge_info has total cap, net cap, net res, net length x, net length y
net_out_edge_info.min(axis=0), net_out_edge_info.max(axis=0), net_out_edge_info.std(axis=0)

### create the tasks for the graph
task_targets = ['pin_tran', 'pin_slack', 'pin_rise_arr', 'pin_fall_arr', 'is_endpoint']
task_df = pin_df[task_targets].copy()
task_df[(pin_df['is_macro'] == 1) | (pin_df['is_seq'] == 1)] = np.nan

# create the training mask for non macro and non sequential pins
train_mask = (pin_df['is_macro'].values == 0) & (pin_df['is_seq'].values == 0)


### 8/8 add -- cell-graph to pin-graph transfer
# generate look-up table
cell_graph_dset_name = '0726_v1'
cell_name_to_id = pd.read_csv(f'{ROOT_DIR}/cell_df.csv', index_col=0)[['cell_name', 'id']]
cell_name_to_id = pd.Series(cell_name_to_id['id'].values, index=cell_name_to_id['cell_name'])

# create cell ids for 'cell_out' edges and 'pin' nodes
# 'pin'
cell_id_of_pins = pin_df['cell_name'].map(cell_name_to_id).values
cell_id_of_pins[np.isnan(cell_id_of_pins)] = -1e8
cell_id_of_pins = cell_id_of_pins.astype(int)
cell_id_of_cell_out = pin_cell_pin_df['cell_name'].map(cell_name_to_id).values
cell_id_of_cell_out[np.isnan(cell_id_of_cell_out)] = -1e8
cell_id_of_cell_out = cell_id_of_cell_out.astype(int)

# create BUFF_LUTs and corresponding key for mapping the cell variables
BUFF_LUTs_flat  = {}
for key, LUT in BUFF_LUTs.items():
    LUT_flat = LUT['LUTidx'].reshape(4, -1)
    # LUT_flat = np.log(LUT['LUTidx']).reshape(4, -1)                                     #### TAKE LOG
    LUT_flat = np.concatenate((np.ones((4, 1)), LUT_flat), axis=1).flatten()
    LUT_flat = np.concatenate((LUT_flat, LUT['LUTmat'].flatten()))
    # LUT_flat = np.concatenate((LUT_flat, np.log(LUT['LUTmat']).flatten()))              #### TAKE LOG
    BUFF_LUTs_flat[key] = LUT_flat


# create mapping from buffer ref to lab id
BUF_ref_to_id = fo4_df.loc[fo4_df['func_id'] == 35, ['libcell_name']]
BUF_ref_to_id.reset_index(inplace=True, drop=True)
BUF_ref_to_id = pd.Series(BUF_ref_to_id.index, index=BUF_ref_to_id['libcell_name'])
    
# cell edge input variables (cell property)
BUFF_LUTs_map = pd.Series([val for _, val in BUFF_LUTs_flat.items()], BUFF_LUTs_flat.keys())
BUFF_LUTs_map.index = BUFF_LUTs_map.index.get_level_values(0).map(BUF_ref_to_id)
assert False not in (BUFF_LUTs_map.index.values == np.arange(len(BUFF_LUTs_map.index)))
BUFF_cell_props = np.array(list(BUFF_LUTs_map))

# rise/fall caps (pin property)
BUFF_rise_fall_caps = pd.read_csv(f'{ROOT_DIR}/BUFF_rise_fall_caps.csv', index_col=0)
BUFF_rise_fall_caps.index = BUFF_rise_fall_caps.index.get_level_values(0).map(BUF_ref_to_id)
assert False not in (BUFF_rise_fall_caps.index.values == np.arange(len(BUFF_rise_fall_caps.index)))
BUFF_pin_rf_caps = BUFF_rise_fall_caps.values

with open(f'{ROOT_DIR}/BUFF_props.pk', 'wb') as pkf:
    pk.dump({'BUFF_cell_props': BUFF_cell_props, 'BUFF_pin_rf_caps': BUFF_pin_rf_caps}, pkf)


# Create dictionary for graph 
# Two edge types: cell_out and net_out

data_dict = {
    ('node', 'cell_out', 'node'): (torch.from_numpy(pin_cell_pin_df['src_id'].values), torch.from_numpy(pin_cell_pin_df['tar_id'].values)),
    ('node', 'net_out', 'node'): (torch.from_numpy(pin_net_pin_df['src_id'].values), torch.from_numpy(pin_net_pin_df['tar_id'].values)),
    # ('node', 'net_in', 'node'): (torch.from_numpy(pin_net_pin_df['tar_id'].values), torch.from_numpy(pin_net_pin_df['src_id'].values))
}

g = dgl.heterograph(data_dict, idtype=torch.int32)

#Add node features IO, FO, rel pos, cap to graph
g.ndata['nf'] = torch.from_numpy(node_features)

#Add node targets slack, tran, rise arr, fall arr to graph
for task_target in task_targets:
    g.ndata[f'{task_target}'] = torch.from_numpy(task_df[task_target].values)
# train mask
g.ndata['train_mask'] = torch.from_numpy(train_mask)

# Add edge features : LUT and net props to graph
g.edata['ef'] = {
    ('node', 'cell_out', 'node'): torch.from_numpy(cell_edge_info),
    ('node', 'net_out', 'node'): torch.from_numpy(net_out_edge_info),
    # ('node', 'net_in', 'node'): torch.from_numpy(net_in_edge_info)
}

# ids
g.ndata['cell_id'] = torch.from_numpy(cell_id_of_pins)
g.edata['cell_id'] = {
    ('node', 'cell_out', 'node'): torch.from_numpy(cell_id_of_cell_out)
}

### construct pseudo nodes for fanin at level 1, because all nodes must start from fanout!!
# first sort the nodes by topological order
na, nb = g.edges(etype='net_out', form='uv')
ca, cb = g.edges(etype='cell_out', form='uv')
g_homo = dgl.graph((torch.cat([na, ca]).cpu(), torch.cat([nb, cb]).cpu()))
topo = dgl.topological_nodes_generator(g_homo)

# find the level 1 fanin nodes
lv1_fanin_nodes = topo[0][g.ndata['nf'][topo[0], 1] == 0]

# add pseudo nodes for fanout
g.add_nodes(len(lv1_fanin_nodes))
for feat in g.ndata.keys():
    g.ndata[feat][-len(lv1_fanin_nodes):] = g.ndata[feat][lv1_fanin_nodes]
    if feat == 'nf':
        g.ndata[feat][-len(lv1_fanin_nodes):, 1] = 1

# add edge from the pseudo nodes to the fanin nodes
ef_feats = g.edata['ef'][('node', 'net_out', 'node')].shape[1]
pseudo_node_ids = np.arange(len(is_fanout), len(is_fanout)+len(lv1_fanin_nodes))
g.add_edges(pseudo_node_ids, lv1_fanin_nodes.numpy(), etype=('node', 'net_out', 'node'), data={'ef': torch.zeros(len(lv1_fanin_nodes), ef_feats, dtype=torch.float64)})
# g.add_edges(lv1_fanin_nodes.numpy(), pseudo_node_ids, etype=('node', 'net_in', 'node'), data={'ef': torch.zeros(len(lv1_fanin_nodes), ef_feats, dtype=torch.float64)})

# Reports the node and edge data
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


# Create singular graph with combined edge type
na, nb = g.edges(etype='net_out', form='uv')
ca, cb = g.edges(etype='cell_out', form='uv')
g_homo = dgl.graph((torch.cat([na, ca]).cpu(), torch.cat([nb, cb]).cpu()))
topo = dgl.topological_nodes_generator(g_homo)
g_homo.ndata['fanout'] = g.ndata['nf'][:, 1].cpu()

# assert that even/odd levels -> only fanout/fanin nodes
for li, nodes in enumerate(topo):
    fanout = g_homo.ndata['fanout'][nodes].numpy()
    assert (li % 2 == 0 and (fanout == 0).sum() == 0) or (li % 2 == 1 and (fanout == 1).sum() == 0)

# add pseudo node to last level if odd number of levels
if len(topo) % 2 == 1:
    # get the fanout nodes at the last level
    fin_lv_fanout_nodes = topo[-1]

    # add pseudo nodes for fanin
    g.add_nodes(len(fin_lv_fanout_nodes))
    for feat in g.ndata.keys():
        g.ndata[feat][-len(fin_lv_fanout_nodes):] = g.ndata[feat][fin_lv_fanout_nodes]
        if feat == 'nf':
            g.ndata[feat][-len(fin_lv_fanout_nodes):, 1] = 0

    # add edge from the fanout nodes to the pseudo nodes
    ef_feats = g.edata['ef'][('node', 'net_out', 'node')].shape[1]
    pseudo_node_ids = np.arange(len(g.ndata['nf'])-len(fin_lv_fanout_nodes), len(g.ndata['nf']))
    g.add_edges(fin_lv_fanout_nodes.numpy(), pseudo_node_ids, etype=('node', 'net_out', 'node'), data={'ef': torch.zeros(len(fin_lv_fanout_nodes), ef_feats, dtype=torch.float64)})
    # g.add_edges(pseudo_node_ids, fin_lv_fanout_nodes.numpy(), etype=('node', 'net_in', 'node'), data={'ef': torch.zeros(len(fin_lv_fanout_nodes), ef_feats, dtype=torch.float64)})

na, nb = g.edges(etype='net_out', form='uv')
ca, cb = g.edges(etype='cell_out', form='uv')
g_homo = dgl.graph((torch.cat([na, ca]).cpu(), torch.cat([nb, cb]).cpu()))
topo = dgl.topological_nodes_generator(g_homo)
g_homo.ndata['fanout'] = g.ndata['nf'][:, 1].cpu()

assert len(topo) % 2 == 0

### inspect the topography!
for li, nodes in enumerate(topo):
    fanout = g_homo.ndata['fanout'][nodes].numpy()
    print(f'level {li}, # nodes = {len(nodes)}, # fanout = {(fanout == 1).sum()}, # fanin = {(fanout == 0).sum()}')


# Extract subgraphs
MIN_SUB_GRAPH_NODES = 100
na, nb = g.edges(etype='net_out', form='uv')
ca, cb = g.edges(etype='cell_out', form='uv')
g_homo = dgl.graph((torch.cat([na, ca]).cpu(), torch.cat([nb, cb]).cpu()))
nx_g = g_homo.to_networkx().to_undirected()
comps = nx.connected_components(nx_g)
comps = [np.array(sorted(list(comp))) for comp in comps]
sub_gs = [g.subgraph(nodes) for nodes in comps if len(nodes) >= MIN_SUB_GRAPH_NODES]

# Changes graph to bi directional
# Add net_in edge with values opposite from net_out edges for each subgraph
bidir_gs = []
for sub_g in sub_gs:
    net_out_src, net_out_dst = sub_g.edges(etype='net_out')
    data_dict = {
        ('node', 'cell_out', 'node'): sub_g.edges(etype='cell_out'),
        ('node', 'net_out', 'node'): sub_g.edges(etype='net_out'),
        ('node', 'net_in', 'node'): (net_out_dst, net_out_src)
    }
    bidir_g = dgl.heterograph(data_dict, idtype=torch.int32)
    for key in sub_g.ndata.keys():
        bidir_g.ndata[key] = sub_g.ndata[key]
    bidir_g.edata['ef'] = {
        ('node', 'cell_out', 'node'): sub_g.edata['ef'][('node', 'cell_out', 'node')],
        ('node', 'net_out', 'node'): sub_g.edata['ef'][('node', 'net_out', 'node')],
        ('node', 'net_in', 'node'): -sub_g.edata['ef'][('node', 'net_out', 'node')]
    }
    bidir_g.edata['cell_id'] = {
        ('node', 'cell_out', 'node'): sub_g.edata['cell_id'][('node', 'cell_out', 'node')]
    }
    bidir_gs.append(bidir_g)


# save dgl graph to original dataset location
dgl.save_graphs(f'{design_dir}/graph.dgl', bidir_gs)



