import os, sys
sys.path.append("/home/vgopal18/Circuitops/CircuitOps/src/python/")
from circuitops_api import *

def create_singular_graph(g):
    na, nb = g.edges(etype='net_out', form='uv')
    ca, cb = g.edges(etype='cell_out', form='uv')
    g_homo = dgl.graph((torch.cat([na, ca]).cpu(), torch.cat([nb, cb]).cpu()))
    return g

def add_pseudo_nodes(g,level,dir='fo'):
    """
    Adds pseudo fan-in or fan-out nodes at required level

    :param g: Graph in which nodes are to be added
    :type g: DGL graph object
    :param level: Level at which pseudo nodes have to be inserted
    :type level: int
    :param dir: Either 'fo' for fan-out or 'fi' for fan-in
    :type dir: str

    :return: Graph with pseudo nodes added
    :rtype: DGL graph object

    Example:
        graph_mod = add_pseudo_nodes(og_graph)
    """
    topo = dgl.topological_nodes_generator(g_homo)
    fanin_nodes = topo[level-1][g.ndata['nf'][topo[0], 1] == 0]
    
    # add pseudo nodes for fanout
    g.add_nodes(len(fanin_nodes))
    for feat in g.ndata.keys():
        g.ndata[feat][-len(fanin_nodes):] = g.ndata[feat][fanin_nodes]
        if feat == 'nf':
            g.ndata[feat][-len(fanin_nodes):, 1] = 1
    
    # add edge from the pseudo nodes to the fanin nodes
    ef_feats = g.edata['ef'][('node', 'net_out', 'node')].shape[1]
    pseudo_node_ids = np.arange(len(is_fanout), len(is_fanout)+len(fanin_nodes))
    g.add_edges(pseudo_node_ids, lv1_fanin_nodes.numpy(), etype=('node', 'net_out', 'node'), data={'ef': torch.zeros(len(lv1_fanin_nodes), ef_feats, dtype=torch.float64)})
    return g

def change_graph_bidirectional(g):
    """
    Changes the cell to cell edges as bi-directional

    :param g: Graph which has to be changed to bi directional
    :type g: DGL graph object
    :return: bidir_g, graph with bi-directional edges
    :rtype: DGL graph object

    Example:
        bidir_graph = change_graph_bidirectional(og_graph)
    """
    net_out_src, net_out_dst = g.edges(etype='net_out')
    data_dict = {
        ('node', 'cell_out', 'node'): g.edges(etype='cell_out'),
        ('node', 'net_out', 'node'): g.edges(etype='net_out'),
        ('node', 'net_in', 'node'): (net_out_dst, net_out_src)
    }
    bidir_g = dgl.heterograph(data_dict, idtype=torch.int32)
    for key in g.ndata.keys():
        bidir_g.ndata[key] = g.ndata[key]
    bidir_g.edata['ef'] = {
        ('node', 'cell_out', 'node'): g.edata['ef'][('node', 'cell_out', 'node')],
        ('node', 'net_out', 'node'): g.edata['ef'][('node', 'net_out', 'node')],
        ('node', 'net_in', 'node'): -g.edata['ef'][('node', 'net_out', 'node')]
    }
    bidir_g.edata['cell_id'] = {
        ('node', 'cell_out', 'node'): g.edata['cell_id'][('node', 'cell_out', 'node')]
    }
    return bidir_g


