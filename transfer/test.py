# coding=gbk
from loader import MoleculeDataset_aug
from torch_geometric.data import DataLoader
import os.path as osp
from copy import deepcopy
from aug import Line_Graph
if __name__ == '__main__':

    path = osp.expanduser('~/datasets')
    path = osp.join(path, "zinc_standard_agent")
    dataset = MoleculeDataset_aug(path, dataset="zinc_standard_agent")
    dataset.aug = 0
    dataset1 = deepcopy(dataset)
    dataset1.aug = 1
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    loader1 = DataLoader(dataset1, batch_size=2, shuffle=False)
    for step, batch in enumerate(zip(loader, loader1)):
        batch1,batch2 = batch
        print(batch1.x)
        print(batch1.edge_index)
        print(batch1.edge_attr)
        print("_____________________________")
        print(batch2.x)
        print(batch2.edge_index)
        print(batch2.edge_attr)
        break