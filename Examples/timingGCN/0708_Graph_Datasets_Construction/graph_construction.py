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


# ### Extract from CircuitOps Tables and Raw EDA Files and Construct TimingGCN-compatible Graph

# In[1]:


import os, sys
ROOT_DIR = '/home/vgopal18/Circuitops/examples/timingGCN/0708_Graph_Datasets_Construction/trial/'
sys.path.append(ROOT_DIR)
sys.path.append("/home/vgopal18/Circuitops/CircuitOps/src/python/")
#sys.path.append(f'{ROOT_DIR}/project')

import numpy as np
import pandas as pd
import pickle as pk

import graph_tool as gt

import matplotlib.pyplot as plt

from libertyParser.libertyParser import libertyParser                           # https://github.com/liyanqing1987/libertyParser
#from utils.generate_LPG_from_tables import read_tables_OpenROAD_v2
from circuitops_api import *

# In[2]:


#design_names = ['NV_NVDLA_partition_m', 'NV_NVDLA_partition_p', 'ariane136', 'mempool_tile_wrap']
design_name = 'gcd'
dset_name = '0709_v1'
#pin_df, cell_df, net_df, design_prop_df, pin_pin_df, _, _, _, _, fo4_df = read_IR_tables('/home/vgopal18/Circuitops/CircuitOps/IRs/nangate45/gcd/')
# pin_df, cell_df, net_df, pin_pin_df, cell_pin_df, net_pin_df, net_cell_df, cell_cell_df, fo4_df = read_tables_OpenROAD_v2(f'{ROOT_DIR}/IR_Tables/{design_name}/')

g, pin_df, cell_df, net_df, fo4_df, pin_pin_df, _, _, _, _, edge_df, _, _ = \
  generate_LPG_from_tables("/home/vgopal18/Circuitops/CircuitOps/IRs/asap7/gcd/")
# In[3]:


# read the size files and save to pin_df
#with open(f'{ROOT_DIR}/design/{design_name}/{design_name}.size', 'r') as file:
#    lines = file.readlines()
#sizes = []
#for line in lines:
#    sizes.append(line.replace('\n','').split(' '))
#sizes = pd.DataFrame(sizes, columns=['cellname', 'ref'])
#sizes = pd.Series(sizes['ref'].values, index=sizes['cellname'])

#pin_df['ref_opt'] = pin_df['cellname'].map(sizes)
#assert pin_df['ref_opt'].isna().sum() == 0

# take away three specific pin-net-pin connections in ariane136 to make it a DAG
if design_name == 'ariane136':
    drop_idx = []
    for src_name, tar_name in [['g1407995/Y', 'g1410303/A'], ['g1405562/Y', 'g1406560/B'], ['g1404610/Y', 'g1405522/B1']]:
        src_id = pin_df['id'][pin_df['name'] == src_name].values
        tar_id = pin_df['id'][pin_df['name'] == tar_name].values
        assert len(src_id) == 1 and len(tar_id) == 1
        drop_idx.append(pin_pin_df.index[(pin_pin_df['src_id'] == src_id[0]) & (pin_pin_df['tar_id'] == tar_id[0])].values[0])
    drop_idx = pd.Index(drop_idx)
    pin_pin_df.drop(index=drop_idx, inplace=True)

# # remove all the macros pins + cells + nets
# macro_ids = pin_df[pin_df['is_macro'] == 1]['id'].values
# assert False not in ((macro_ids[1:] - macro_ids[:-1]) == 1)
# pin_pin_df.drop(index = pin_pin_df.index[pin_pin_df['src_id'].isin(macro_ids) | pin_pin_df['tar_id'].isin(macro_ids)], inplace=True)
# pin_df.drop(index = pin_df.index[pin_df['is_macro'] == 1], inplace=True)


# discard pins that are not connected to anything
isolate_ids = set(pin_df['id']).difference(set(pin_pin_df['src_id']).union(set(pin_pin_df['tar_id'])))
pin_df.drop(index = pin_df.index[pin_df['id'].isin(isolate_ids)], inplace=True)


# In[ ]:


# reset indices
pin_df.reset_index(inplace=True, drop=True)
pin_pin_df.reset_index(inplace=True, drop=True)

# create new columns for original ids
pin_df['org_id'] = pin_df['id']
pin_pin_df['org_src_id'] = pin_pin_df['src_id']
pin_pin_df['org_tar_id'] = pin_pin_df['tar_id']

# create mapping from original id to new id
map_pin_id = pd.Series(pin_df.index, pin_df['org_id'])
pin_df['id'] = pin_df['org_id'].map(map_pin_id)
pin_pin_df['src_id'] = pin_pin_df['src_id'].map(map_pin_id)
pin_pin_df['tar_id'] = pin_pin_df['tar_id'].map(map_pin_id)


# In[5]:


### modify is_port
# A quick parser to extract all I/O netnames from verilog netlist
def get_port_nets(fullpath: str):
    start_reading_IO = False
    with open(fullpath, 'r') as file:
        # read until first IO line
        while not start_reading_IO:
            line = file.readline().replace('\n', '')
            if 'input' in line.split(' ') or 'output'in line.split(' '):
                start_reading_IO = True
        IOnets = [line.split(' ')[-1].replace(';','')]
        # read until first wire line
        while True:
            line = file.readline().replace('\n', '')
            if 'input' in line.split(' ') or 'output'in line.split(' '):
                IOnets += [line.split(' ')[-1].replace(';','')]
            elif 'wire' in line.split(' '):
                break
    return IOnets


IOnets = get_port_nets('/home/vgopal18/OpenROAD/OpenROAD-flow-scripts/flow/results/asap7/gcd/base/1_synth.v')

print('[s_port] property in pin_df is invalid. Extract from .v file')
print('split [netname] by \'[\' and take first item')
pin_df['netname_prefix'] = pin_df['net_name'].str.replace('\\', '').str.split('[', expand=True)[0]

pin_df['is_port'] = pin_df['netname_prefix'].isin(IOnets)
print('There are', pin_df['is_port'].sum(), 'port pins')


# In[6]:


### add to-boundary distances
# A quick parser to extract die boundaries
def get_die_boundaries(fullpath: str):
    with open(fullpath, 'r') as file:
        while True:
            line = file.readline()
            if 'DIEAREA' in line:
                lx, by, rx, ty = np.array(line.split(' '))[[2,3,6,7]]
                return float(lx), float(by), float(rx), float(ty)


lx, by, rx, ty = get_die_boundaries('/home/vgopal18/OpenROAD/OpenROAD-flow-scripts/flow/results/asap7/gcd/base/6_final.def')
print(f'Boundary ({lx} {by}) - ({rx} {ty})')

pin_df['to_top'] = ty - pin_df['y']
pin_df['to_left'] = pin_df['x'] - lx
pin_df['to_right'] = rx - pin_df['x']
pin_df['to_bottom'] = pin_df['y'] - by


# In[7]:


# LUT extraction for edges
lib_dir = '/home/vgopal18/OpenROAD/OpenROAD-flow-scripts/flow/platforms/asap7/lib/'
libfiles = [libfile for libfile in sorted(os.listdir(lib_dir)) if libfile[:5] != 'sram_'] # discard macros
print(libfiles)
libs = []
for libfile in libfiles:
    if libfile[-4:] == '.lib':
        print(f'Parsing {libfile}')
        libs.append(libertyParser('/home/vgopal18/OpenROAD/OpenROAD-flow-scripts/flow/platforms/asap7/lib/'+libfile))


print(libs)

cell_edges = pin_pin_df[pin_pin_df['is_net'] == 0].copy()
cell_edges['src_pin_name'] = cell_edges['src'].str.split('/').str[-1]
cell_edges['tar_pin_name'] = cell_edges['tar'].str.split('/').str[-1]
cell_edges['cell_name'] = cell_edges['src'].str.split('/').str[:-1].str.join('/')
libcell_lookup = pd.Series(cell_df['libcell_name'].values, index=cell_df['cell_name'])
cell_edges['libcell_name'] = cell_edges['cell_name'].map(libcell_lookup)
pin_df['libcell_name'] = pin_df['cell_name'].map(libcell_lookup)

# save cell_edge additional information to pin_pin_df
for col in cell_edges.columns:
    if col not in pin_pin_df.columns:
        pin_pin_df[col] = None
        pin_pin_df.loc[pin_pin_df['is_net'] == 0, col] = cell_edges[col]


# In[8]:
print(pin_df.loc[0])

# get cells used in this design
used_refs = set(pin_df['libcell_name'][pin_df['is_macro'] == 0].values)
info_cells = []
for lib in libs:
    for cell in lib.getCellList():
        if cell in used_refs:
            info_cells.append(cell)


info_cells = set(info_cells)
no_info_refs = used_refs.difference(info_cells)
#assert len(no_info_refs) == 0, f'No info: {no_info_refs}'


# In[30]:


# extract the LUTs for BUFFER!!!
print('BUFFER BUFFER BUFFER BUFFER BUFFER BUFFER BUFFER BUFFER')

arc_types = ['cell_rise', 'cell_fall', 'rise_transition', 'fall_transition']
parseLUT = lambda tab : np.array([row.split(',') for row in tab.replace(' ', '').replace('(\"','').replace('\")','').split('\",\"')]).astype(float)
parseLUTidx = lambda idxs : np.array(idxs.replace(' ','').replace('(\"','').replace('\")','').split(',')).astype(float)
BUFF_LUTs = {}
for lib in libs:
    # create a mapping from group name to index in lib dict for faster query
    group_names = [group['name' ]for group in lib.libDic['group']]
    group_map = pd.Series(np.arange(len(group_names)), index=group_names)
    for cell in lib.getCellList():
        found = False
        for pref in ['BUFx', 'HB1x', 'HB2x', 'HB3x', 'HB4x']:
            if pref == cell[:4]:
                found = True
                break
        if not found:
            continue
        ### Section for extracting LUTs
        # extract the src and tar pin names, and get pin info from liberty
        src_tar = cell_edges[['src_pin_name', 'tar_pin_name']][cell_edges['libcell_name'] == cell]
        pin_pairs = set([*zip(src_tar['src_pin_name'], src_tar['tar_pin_name'])])
        pins = lib.getLibPinInfo(cellList=[cell])['cell'][cell]['pin']
        # for each src/tar pin pair, extract the LUT for timing
        pin_pairs = {('A', 'Y')}
        for src_pin, tar_pin in pin_pairs:
            pin_stat = pins[tar_pin]['timing']
            LUTmat, LUTidx = [], []
            for one_arc in pin_stat:
                # there might be multiple timing arcs for one src/tar pair
                # try to extract the arc where the 'when' key is not specified
                if one_arc['related_pin'] != f'\"{src_pin}\"':
                    continue
                for arc_type in arc_types:
                    LUTidx.append(parseLUTidx(one_arc['table_type'][arc_type]['index_1']))
                    LUTidx.append(parseLUTidx(one_arc['table_type'][arc_type]['index_2']))
                    LUTmat.append(parseLUT(one_arc['table_type'][arc_type]['values']))
            LUTidx = np.array(LUTidx).reshape(-1, 4, 2, 7)
            LUTmat = np.array(LUTmat).reshape(-1, 4, 7, 7)
            # assert lengths are the same
            assert LUTmat.shape[0] == LUTidx.shape[0]
            # all the indices must be the same!
            assert True not in (LUTidx.max(axis=0) != LUTidx.min(axis=0))
            LUTidx = LUTidx[0]
            # take the values of the matrix with the worst average
            worst_ids = LUTmat.mean(axis=(2, 3)).argmax(axis=0)
            LUTmat = np.array([LUTmat[idx1, idx2] for idx1, idx2 in zip(worst_ids, np.arange(len(arc_types)))])
            # save to dict that is indexed with (cell, src_pin, tar_pin)
            BUFF_LUTs[(cell, src_pin, tar_pin)] = {'LUTidx': LUTidx, 'LUTmat': LUTmat}


print(f'{len(BUFF_LUTs.keys())} different BUFF_LUTs, indexed by (cell, src, tar), for [{design_name}], are constructed')

with open(f'{ROOT_DIR}/BUFF_LUTs.pk', 'wb') as pkf:
    pk.dump(BUFF_LUTs, pkf)


# In[53]:


# extract rise/fall capacitances for BUFFER!!!
print('BUFFER BUFFER BUFFER BUFFER BUFFER BUFFER BUFFER BUFFER')

# pin_df['pinname'] = [lst[-1] for lst in pin_df['name'].str.split('/')]
RiseCaps, FallCaps = pd.Series(), pd.Series()
for lib in libs:
    # create a mapping from group name to index in lib dict for faster query
    group_names = [group['name' ]for group in lib.libDic['group']]
    group_map = pd.Series(np.arange(len(group_names)), index=group_names)
    for cell in lib.getCellList():
        found = False
        for pref in ['BUFx', 'HB1x', 'HB2x', 'HB3x', 'HB4x']:
            if pref == cell[:4]:
                found = True
                break
        if not found:
            continue
        ### Section for extracting rise/fall capacitances for input pins (dir = 1)
        # extract the src pin names, and get pin info from liberty
        # src_pins = set(pin_df['pinname'][(pin_df['ref'] == cell) & (pin_df['dir'] == 1)])
        src_pins = {'A'}
        cell_group = lib.libDic['group'][group_map[cell]]['group']
        found_src_pins = []
        for cell_attr in cell_group:
            if cell_attr['name'] in src_pins:
                src_pin = cell_attr['name']
                found_src_pins.append(src_pin)
                RiseCaps[cell] = cell_attr['rise_capacitance']
                FallCaps[cell] = cell_attr['fall_capacitance']
        # assert that we can find all input pins
        assert set(found_src_pins) == src_pins


BUFF_rise_fall_caps = pd.concat([pd.DataFrame(RiseCaps, columns=['rise_cap']), pd.DataFrame(FallCaps, columns=['fall_cap'])], axis=1)
BUFF_rise_fall_caps.to_csv(f'{ROOT_DIR}/BUFF_rise_fall_caps.csv')


# In[14]:


# extract the LUTs
arc_types = ['cell_rise', 'cell_fall', 'rise_transition', 'fall_transition']
parseLUT = lambda tab : np.array([row.split(',') for row in tab.replace(' ', '').replace('(\"','').replace('\")','').split('\",\"')]).astype(float)
parseLUTidx = lambda idxs : np.array(idxs.replace(' ','').replace('(\"','').replace('\")','').split(',')).astype(float)
LUTs = {}
for lib in libs:
    # create a mapping from group name to index in lib dict for faster query
    group_names = [group['name' ]for group in lib.libDic['group']]
    group_map = pd.Series(np.arange(len(group_names)), index=group_names)
    for cell in lib.getCellList():
        if cell not in info_cells:
            continue
        ### Section for extracting LUTs
        # extract the src and tar pin names, and get pin info from liberty
        src_tar = cell_edges[['src_pin_name', 'tar_pin_name']][cell_edges['libcell_name'] == cell]
        pin_pairs = set([*zip(src_tar['src_pin_name'], src_tar['tar_pin_name'])])
        pins = lib.getLibPinInfo(cellList=[cell])['cell'][cell]['pin']
        # for each src/tar pin pair, extract the LUT for timing
        for src_pin, tar_pin in pin_pairs:
            pin_stat = pins[tar_pin]['timing']
            LUTmat, LUTidx = [], []
            for one_arc in pin_stat:
                # there might be multiple timing arcs for one src/tar pair
                # try to extract the arc where the 'when' key is not specified
                if one_arc['related_pin'] != f'\"{src_pin}\"':
                    continue
                for arc_type in arc_types:
                    LUTidx.append(parseLUTidx(one_arc['table_type'][arc_type]['index_1']))
                    LUTidx.append(parseLUTidx(one_arc['table_type'][arc_type]['index_2']))
                    LUTmat.append(parseLUT(one_arc['table_type'][arc_type]['values']))
            LUTidx = np.array(LUTidx).reshape(-1, 4, 2, 7)
            LUTmat = np.array(LUTmat).reshape(-1, 4, 7, 7)
            # assert lengths are the same
            assert LUTmat.shape[0] == LUTidx.shape[0]
            # all the indices must be the same!
            assert True not in (LUTidx.max(axis=0) != LUTidx.min(axis=0))
            LUTidx = LUTidx[0]
            # take the values of the matrix with the worst average
            worst_ids = LUTmat.mean(axis=(2, 3)).argmax(axis=0)
            LUTmat = np.array([LUTmat[idx1, idx2] for idx1, idx2 in zip(worst_ids, np.arange(len(arc_types)))])
            # save to dict that is indexed with (cell, src_pin, tar_pin)
            LUTs[(cell, src_pin, tar_pin)] = {'LUTidx': LUTidx, 'LUTmat': LUTmat}

print(f'{len(LUTs.keys())} different LUTs, indexed by (cell, src, tar), for [{design_name}], are constructed')


# In[12]:


# extract rise/fall capacitances
pin_df['pin_name'] = [lst[-1] for lst in pin_df['pin_name'].str.split('/')]
RiseCaps, FallCaps = pd.Series(), pd.Series()

for lib in libs:
    # create a mapping from group name to index in lib dict for faster query
    group_names = [group['name' ]for group in lib.libDic['group']]
    group_map = pd.Series(np.arange(len(group_names)), index=group_names)
    for cell in lib.getCellList():
        if cell not in info_cells:
            continue
        ### Section for extracting rise/fall capacitances for input pins (dir = 1)
        # extract the src pin names, and get pin info from liberty
        src_pins = set(pin_df['pin_name'][(pin_df['libcell_name'] == cell) & (pin_df['dir'] == 1)])
        cell_group = lib.libDic['group'][group_map[cell]]['group']
        found_src_pins = []
        for cell_attr in cell_group:
            if cell_attr['name'] in src_pins:
                src_pin = cell_attr['name']
                found_src_pins.append(src_pin)
                RiseCaps[f'{cell}/{src_pin}'] = cell_attr['rise_capacitance']
                FallCaps[f'{cell}/{src_pin}'] = cell_attr['fall_capacitance']
        # assert that we can find all input pins
        assert set(found_src_pins) == src_pins

# ### set the rise/fall capacitances by mapping
pin_df['rise_cap'] = pin_df['libcell_name'].str.cat(pin_df['pin_name'], sep='/').map(RiseCaps)
pin_df['fall_cap'] = pin_df['libcell_name'].str.cat(pin_df['pin_name'], sep='/').map(FallCaps)
assert False not in (pin_df['rise_cap'].isna() == pin_df['fall_cap'].isna())
#assert pin_df['rise_cap'][(pin_df['is_macro'] == 0) & (pin_df['dir'] == 1)].isna().sum() == 0


# In[13]:


# save pin_df, pin_pin_df, LUTs
print(f'Saving to dataset {dset_name}/{design_name}')

if not os.path.isdir(f'{ROOT_DIR}/'):
    os.mkdir(f'{ROOT_DIR}/')
if not os.path.isdir(f'{ROOT_DIR}/'):
    os.mkdir(f'{ROOT_DIR}/')

with open(f'{ROOT_DIR}/LUTs.pk', 'wb') as pkf:
    pk.dump(LUTs, pkf)

pin_df.to_csv(f'{ROOT_DIR}/pin_df.csv')
pin_pin_df.to_csv(f'{ROOT_DIR}/pin_pin_df.csv')
net_df.to_csv(f'{ROOT_DIR}/net_df.csv')
cell_df.to_csv(f'{ROOT_DIR}/cell_df.csv')
fo4_df.to_csv(f'{ROOT_DIR}/fo4_df.csv')

# In[ ]:




