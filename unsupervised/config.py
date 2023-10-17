# coding=gbk
import argparse
import torch
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List
import networkx as nx
import numpy as np
from torch_geometric.utils import degree

from torch import nn
import os.path as osp
import random
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

"""
SGN mode : mode
1: Contrastive Learning Based on Single-Order Subgraph Networks from G(ori) vs G(SGN1) (SGNCL);
2: Contrastive Learning Based on Single-Order Subgraph Networks from G(ori) vs G(SGN1) (SGNCL-v1);
3: Contrastive learning based on fused Multi-Order Subgraph Networks with p =0.5 ( SGNCL-fu);
4: Contrastive learning based on fused Multi-Order Subgraph Networks with p-Tunability ( SGNCL-fu);

related configuration:
|  dataset  |    layers  |     lr    |  batch_size  | pool   |   epoch  |
|   mutag   |      3     |     0.01  |     128      |  mean  |    40    |
|   PTC     |      3     |     0.01  |     128      |  mean  |    40    |
|   NCI1    |      3     |     0.01  |     128      |  sun   |    40    |
| PROTEINS  |      4     |    0.001  |     128      |  max   |    40    |
|    DD     |      4     |    0.001  |     128      |  sum   |   100    |
|  IMDB-B   |      4     |    0.001  |      64      |  sum   |    40    |
|  IMDB-M   |      4     |    0.001  |      64      |  sum   |    40    |
|   RDT-B   |      3     |    0.001  |      32      |  mean  |    40    |
"""


def arg_parse():
    parser = argparse.ArgumentParser(description='SGNCL: subgraph network-based contrastive learning model')
    parser.add_argument('--dataset', type=str, default="MUTAG", help='name of dataset (default: MUTAG)')
    parser.add_argument('--base_model', type=str, default="GINConv",
                        help='Selecting the GNN model that configures the encoder in graph contrastive learning.')
    parser.add_argument('--activation', type=str, default="relu", help='we can choose prelu, relu and elu')
    parser.add_argument('--mode', type=int, default=1, help='we can choose mode: 1,2,3,4')
    parser.add_argument('--pool', type=str, default="mean", help='mean, sum, max')
    parser.add_argument('--hidden_dim', type=int, default=32, help='input hidden_dim in GNN model')
    parser.add_argument('--epoch', type=int, default=40, help='input epoch for training (default: 50)')
    parser.add_argument('--num_layers', type=int, default=3, help='num_layers of GCN (default: 2)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--tau', type=float, default=0.2, help='L.InfoNCE(tau=0.2)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--p', type=float, default=0,
                        help="Hyper-parameter of contrastive learning based on fused multilevel subgraph network")
    parser.add_argument('--gpu', type=str, default='1', help='gpu')

    return parser.parse_args()


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, batch=None, device='cuda'):
        self.y = label
        self.x = []
        self.edge_index = [[], []]
        self.device = device
        self.batch = batch
        self.g = g
        self.node_tags = node_tags


def Line_Graph(dataloader, device):
    G_list = []

    for data in dataloader:
        batch = []
        label = data.y
        G = nx.Graph()

        # Node property settings for social networks
        if data.x == None:
            deg = degree(data.edge_index[1])

        # Relabel the edges in the original graph as nodes of SGN
        LG_node_joint = []
        LG_node_num = 0
        for i in range(len(data.edge_index[0])):
            up_index_tag = int(data.edge_index[0][i])
            down_index_tag = int(data.edge_index[1][i])
            if [up_index_tag, down_index_tag] in LG_node_joint or [down_index_tag, up_index_tag] in LG_node_joint:
                continue
            else:
                LG_node_joint.append([up_index_tag, down_index_tag])
                LG_node_num = LG_node_num + 1
                batch.append(int(data.batch[up_index_tag]))
        # Define the node attributes in the SGN
        LG_x_attr = []
        for i in range(LG_node_num):
            G.add_node(i)
            up_index_i = LG_node_joint[i][0]
            down_index_i = LG_node_joint[i][1]
            if data.x == None:
                up_index_attr = int(deg[up_index_i])
                down_index_attr = int(deg[down_index_i])
                LG_x_attr1 = [up_index_attr, down_index_attr]
            else:
                up_index_attr = data.x[up_index_i]
                down_index_attr = data.x[down_index_i]
                LG_x_attr1 = list(torch.cat([up_index_attr, down_index_attr], -1))
            LG_x_attr.append(LG_x_attr1)

            for j in range(LG_node_num):
                if i == j:
                    continue
                else:
                    up_index_j = LG_node_joint[j][0]
                    down_index_j = LG_node_joint[j][1]
                    if up_index_i == up_index_j or up_index_i == down_index_j or down_index_i == up_index_j or down_index_i == down_index_j:
                        G.add_edge(i, j)
        G_list.append(S2VGraph(G, label, LG_x_attr, batch, device))

    for G_lg in G_list:
        for i, j in G_lg.g.edges():
            G_lg.edge_index[0].append(i)
            G_lg.edge_index[1].append(j)

        for i in range(len(G_lg.g)):
            G_lg.x.append(G_lg.node_tags[i])

        G_lg.x = torch.FloatTensor(G_lg.x)
        G_lg.batch = torch.LongTensor(G_lg.batch)
        G_lg.edge_index = torch.LongTensor(G_lg.edge_index)

    return G_list


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = osp.expanduser('~/datasets')
    dataset = 'MUTAG'
    path = osp.join(path, dataset)
    dataset = TUDataset(path, name=dataset)
    dataloader = DataLoader(dataset, batch_size=5)
    for data in dataloader:
        print(data.batch)
