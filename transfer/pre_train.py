from config import arg_phase
args = arg_phase()
import os.path as osp
from loader_aug import MoleculeDataset_aug

from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import DataLoader
import torch_scatter
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import GNN

from copy import deepcopy
import gc


class SGNCL_TRANS(nn.Module):
    def __init__(self, gnn1,gnn2, pool):
        super(SGNCL_TRANS, self).__init__()
        self.encoder_ori = gnn1
        self.encoder_aug = gnn2
        if pool == "mean":
            self.pool = global_mean_pool
        elif pool == "sum":
            self.pool = global_add_pool
        elif pool == "max":
            self.pool = global_max_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_ori(self, x, edge_index, edge_attr, batch):
        x = self.encoder_ori(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x
    
    def forward_aug(self, x, edge_index, edge_attr, batch):
        x = self.encoder_aug(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x
        
    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def loss_cl_v(self, x1, x2, x3):
        T = 0.1
        batch_size, _ = x1.size()

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        x3_abs = x3.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)

        sim_matrix_SGN = torch.einsum('ik,jk->ij', x1, x3) / torch.einsum('i,j->ij', x1_abs, x3_abs)
        sim_matrix_SGN = torch.exp(sim_matrix_SGN / T)

        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss1 = pos_sim / (sim_matrix.sum(dim=1)-pos_sim)
        pos_sim_sgn = sim_matrix_SGN[range(batch_size), range(batch_size)]
        loss2 = pos_sim_sgn/ (sim_matrix_SGN.sum(dim=1)-pos_sim_sgn)
        loss = - torch.log((loss1 + loss2)/2).mean()
        return loss

def train(args, model, device, dataset, optimizer):
    dataset.aug = 0
    dataset1 = dataset.shuffle()
    dataset2 = deepcopy(dataset1)

    dataset1.aug = args.mode1
    dataset2.aug = args.mode2

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers=8, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers=8, shuffle=False)
    torch.set_grad_enabled(True)
    model.train()

    train_loss_accum = 0

    for step, batch in enumerate(zip(loader1, loader2)):
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()

        x1 = model.forward_ori(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        x2 = model.forward_aug(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)

        loss = model.loss_cl(x1, x2)

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())

    return train_loss_accum/(step+1)

def main():
    # Training settings
    print(args)
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # set up dataset
    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    dataset = MoleculeDataset_aug(path, dataset=args.dataset)
    print(dataset)
   
    # set up model
    encoder_ori = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type, mode =0)
    encoder_aug = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type, mode =1)
    model = SGNCL_TRANS(encoder_ori, encoder_aug, args.pool)
    model.to(device)

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        loss = train(args, model, device, dataset, optimizer)
        print(loss)


        if epoch % 10 == 0:
            torch.save(model.encoder_ori.state_dict(), args.save_model_path + "_seed" + str(args.seed) + "_" + str(epoch) +"_" + str(args.pool) + "_gnn.pth")


if __name__ == "__main__":
    main()
