# coding=gbk
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import degree
from torch_geometric.data import DataLoader
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List

from torch import nn
import os.path as osp
import random
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from loader_v1 import MoleculeDataset


class S2VGraph(object):
    def __init__(self, g, y, LG_x_attr, LG_edge_attr):
        self.y = y
        self.x = LG_x_attr
        self.edge_index = [[], []]
        self.edge_attr = LG_edge_attr
        self.g = g
        self.LG_edge_attr = LG_edge_attr

def Line_Graph_v1(data):
    LG_node_joint = []
    LG_node_joint_idx = []
    LG_node_num = 0

    label = data.y

    G = nx.DiGraph()
    data_edge_index = data.edge_index

    for i in range(len(data_edge_index[0])):

        up_index_tag = int(data_edge_index[0][i])
        down_index_tag = int(data_edge_index[1][i])
        if [up_index_tag, down_index_tag] in LG_node_joint or [down_index_tag, up_index_tag] in LG_node_joint:
            continue
        else:
            LG_node_joint.append([up_index_tag, down_index_tag])
            LG_node_joint_idx.append(i)
            LG_node_num = LG_node_num + 1

    LG_x_attr = []
    LG_edge_attr = []

    for i in range(LG_node_num):
        G.add_node(i)
        tag1, tag2 = [w for w in LG_node_joint[i]]
        tag1_atom_type1 = data.x[tag1][0]
        tag1_atom_type2 = data.x[tag1][1]
        tag2_atom_type1 = data.x[tag2][0]
        tag2_atom_type2 = data.x[tag2][1]
        up_x_attr = 120 * tag1_atom_type2 + tag1_atom_type1
        down_x_attr = 120 * tag2_atom_type2 + tag2_atom_type1
        LG_x_attr.append(list([up_x_attr, down_x_attr]))
        up_bond_type1 = data.edge_attr[LG_node_joint_idx[i]][0]
        up_bond_type2 = data.edge_attr[LG_node_joint_idx[i]][1]

        for j in range(LG_node_num):
            tag3, tag4 = [w for w in LG_node_joint[j]]
            if i==j:
                continue
            elif tag1 == tag3 or tag1 == tag4 or tag2 == tag3 or tag2 == tag4:
                G.add_edge(i, j)
                down_bond_type1 = data.edge_attr[LG_node_joint_idx[j]][0]
                down_bond_type2 = data.edge_attr[LG_node_joint_idx[j]][1]
                #print(up_bond_type1, down_bond_type1, up_bond_type2, down_bond_type2)
                if tag1 == tag3:
                    central_atom_type1 = data.x[ tag1][0]
                    central_atom_type2 = data.x[ tag1][1]
                elif tag1 == tag4:
                    central_atom_type1 = data.x[ tag1][0]
                    central_atom_type2 = data.x[ tag1][1]
                elif tag2 == tag3:
                    central_atom_type1 = data.x[ tag2][0]
                    central_atom_type2 = data.x[ tag2][1]
                elif tag2 == tag4:
                    central_atom_type1 = data.x[ tag2][0]
                    central_atom_type2 = data.x[ tag2][1]
                up_index_attr = up_bond_type1 * 720 + down_bond_type1 * 120 + central_atom_type1
                down_index_attr = up_bond_type2 * 9 + down_bond_type2 * 3 + central_atom_type2
                LG_edge_attr1 = list([up_index_attr, down_index_attr])
                LG_edge_attr.append(LG_edge_attr1)
    G_lg = S2VGraph(G, label, LG_x_attr, LG_edge_attr)

    for i, j in G_lg.g.edges():
        G_lg.edge_index[0].append(i)
        G_lg.edge_index[1].append(j)

    data.x = torch.LongTensor(G_lg.x)
    data.edge_index = torch.LongTensor(G_lg.edge_index)
    data.edge_attr = torch.LongTensor(G_lg.edge_attr)

    return data

def Line_Graph_v2(data):
    LG_node_joint = []
    LG_node_joint_idx = []
    LG_node_num = 0

    label = data.y
    G = nx.DiGraph()
    data_edge_index = data.edge_index

    for i in range(len(data_edge_index[0])):

        up_index_tag = int(data_edge_index[0][i])
        down_index_tag = int(data_edge_index[1][i])
        if [up_index_tag, down_index_tag] in LG_node_joint or [down_index_tag, up_index_tag] in LG_node_joint:
            continue
        else:
            LG_node_joint.append([up_index_tag, down_index_tag])
            LG_node_joint_idx.append(i)
            LG_node_num = LG_node_num + 1

    LG_x_attr = []
    LG_edge_attr = []
    for i in range(LG_node_num):
        G.add_node(i)
        LG_x_attr.append(list(data.edge_attr[LG_node_joint_idx[i]]))
        tag1, tag2 = [w for w in LG_node_joint[i]]
        up_bond_type1 = data.edge_attr[LG_node_joint_idx[i]][0]
        up_bond_type2 = data.edge_attr[LG_node_joint_idx[i]][1]

        for j in range(LG_node_num):
            tag3, tag4 = [w for w in LG_node_joint[j]]
            if i==j:
                continue
            elif tag1 == tag3 or tag1 == tag4 or tag2 == tag3 or tag2 == tag4:
                G.add_edge(i, j)
                down_bond_type1 = data.edge_attr[LG_node_joint_idx[j]][0]
                down_bond_type2 = data.edge_attr[LG_node_joint_idx[j]][1]
                #print(up_bond_type1, down_bond_type1, up_bond_type2, down_bond_type2)
                if tag1 == tag3:
                    central_atom_type1 = data.x[ tag1][0]
                    central_atom_type2 = data.x[ tag1][1]
                elif tag1 == tag4:
                    central_atom_type1 = data.x[ tag1][0]
                    central_atom_type2 = data.x[ tag1][1]
                elif tag2 == tag3:
                    central_atom_type1 = data.x[ tag2][0]
                    central_atom_type2 = data.x[ tag2][1]
                elif tag2 == tag4:
                    central_atom_type1 = data.x[ tag2][0]
                    central_atom_type2 = data.x[ tag2][1]
                #up_index_attr = [up_bond_type1, central_atom_type1, down_bond_type1]
                #down_index_attr = [up_bond_type2, central_atom_type2, down_bond_type2]
                LG_edge_attr1 = [up_bond_type1, central_atom_type1, down_bond_type1, up_bond_type2, central_atom_type2, down_bond_type2]
                LG_edge_attr.append(LG_edge_attr1)
    G_lg = S2VGraph(G, label, LG_x_attr, LG_edge_attr)

    for i, j in G_lg.g.edges():
        G_lg.edge_index[0].append(i)
        G_lg.edge_index[1].append(j)

    data.x = torch.LongTensor(G_lg.x)
    data.edge_index = torch.LongTensor(G_lg.edge_index)
    data.edge_attr = torch.LongTensor(G_lg.edge_attr)

    return data 
if __name__ == '__main__':

    path = osp.expanduser('~/datasets')
    path = osp.join(path, "zinc_standard_agent")
    dataset = MoleculeDataset(path, dataset="zinc_standard_agent")
    print(dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for data in loader:
      print(data.x)
      print(data.edge_index)
      print(data.edge_attr)
      data1 = Line_Graph_v1(data)
      print(data1.x)
      print(data1.edge_index)
      print(data1.edge_attr)
      break
