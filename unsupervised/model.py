import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_add_pool, global_mean_pool, global_max_pool#
import numpy as np
from torch_geometric.utils import degree

def get_base_model(name: str):
    def gat_wrapper(in_channels, out_channels):
        return GATConv(in_channels=in_channels, out_channels=out_channels, heads=1)

    def gin_wrapper(in_channels, out_channels):
        return GINConv(
            nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels)))

    base_model = {
        'GCNConv': GCNConv,
        'SAGEConv': SAGEConv,
        'GATConv': gat_wrapper,
        'GINConv': gin_wrapper
    }
    return base_model[name]


def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'elu': F.elu,
        'prelu': torch.nn.PReLU(),
    }
    return activations[name]

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, base_model, activation, pool):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.base_model = get_base_model(base_model)
        self.activation = get_activation(activation)
        self.pool = pool
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.layers.append(self.base_model(input_dim, hidden_dim))
            else:
                self.layers.append(self.base_model(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
    def forward(self, x, edge_index, batch):
        if x == None:
             x = degree(edge_index[1]).reshape(-1,1)
        x = x.cuda()
        edge_index = edge_index.cuda()
        batch = batch.cuda()
        
        zs = []
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)
            x = self.activation(x)
            x = self.batch_norms[i](x)
            zs.append(x)
        # 获取节点表示
        if self.pool == "mean":
            g_per_dim_cat = [global_mean_pool(z, batch) for z in zs]   
        elif self.pool == "sum":
            g_per_dim_cat = [global_add_pool(z, batch) for z in zs]   
        elif self.pool == "max":
            g_per_dim_cat = [global_max_pool(z, batch) for z in zs]   
        h, G2 = [torch.cat(z, dim=1) for z in [zs, g_per_dim_cat]]
        return h, G2
    def get_embedding(self, dataloader):
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in dataloader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch     
                if x == None:
                    x = degree(edge_index[1]).reshape(-1,1)
                _, x = self.forward(x, edge_index, batch)
                ret.append(x.detach().cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y
    
class SGN_CL(nn.Module):
    def __init__(self, encoder1, encoder2, encoder3, hidden_dim, num_layers, tau, mode, p=0):
        super(SGN_CL, self).__init__()
        self.ori_encoder = encoder1
        self.SGN1_encoder = encoder2
        self.SGN2_encoder = encoder3
        self.linear_shortcut = nn.Linear(hidden_dim * num_layers, hidden_dim * num_layers)
        self.tau = tau
        self.mode = mode
        self.hidden_dim = hidden_dim 
        self.p = p
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim * num_layers),
            nn.PReLU(),
            nn.Linear(hidden_dim * num_layers, hidden_dim * num_layers),
            nn.PReLU(),
            nn.Linear(hidden_dim * num_layers, hidden_dim * num_layers)
        )

    def forward(self, data_ori, SGN1, SGN2):
        # 原始图编码
        h1, g1 = self.ori_encoder(data_ori.x, data_ori.edge_index, data_ori.batch)
        m1 = self.fc(g1) + self.linear_shortcut(g1)
        # 一阶图编码
        if self.mode == 1 or self.mode == 3 or self.mode == 4:
            h2, g2 = self.SGN1_encoder(SGN1.x, SGN1.edge_index, SGN1.batch)
            m2 = self.fc(g2)+ self.linear_shortcut(g2)
        # 二阶图编码
        if self.mode == 2 or self.mode == 3 or self.mode == 4:
            h3, g3 = self.SGN2_encoder(SGN2.x, SGN2.edge_index, SGN2.batch)
            m3 = self.fc(g3)+ self.linear_shortcut(g3)
        # 返回值判断：
        if self.mode == 1:
            return m1, m2
        elif self.mode == 2:
            return m1, m3
        elif self.mode == 3 or self.mode == 4:
            return m1, m2, m3

    def loss(self, m1, m2):
        T = self.tau
        batch_size, _ = m1.size()
        m1_abs = m1.norm(dim=1)
        m2_aug_abs = m2.norm(dim=1)
        
        sim_matrix = torch.einsum('ik,jk->ij', m1, m2) / torch.einsum('i,j->ij', m1_abs, m2_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1)-pos_sim)    
        loss = - torch.log(loss).mean()
        return loss
        
        
    def loss_fu(self, m1, m2, m3):
        T = self.tau
        batch_size, _ = m1.size()
        m1_abs = m1.norm(dim=1)
        m2_aug_abs = m2.norm(dim=1)
        m3_aug_abs = m3.norm(dim=1)
        
        sim_matrix = torch.einsum('ik,jk->ij', m1, m2) / torch.einsum('i,j->ij', m1_abs, m2_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        
        sim_matrix_SGN = torch.einsum('ik,jk->ij', m1, m3) / torch.einsum('i,j->ij', m1_abs, m3_aug_abs)
        sim_matrix_SGN = torch.exp(sim_matrix_SGN / T)
        
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss1 = pos_sim / (sim_matrix.sum(dim=1)-pos_sim)
        pos_sim_sgn = sim_matrix_SGN[range(batch_size), range(batch_size)]
        loss2 = pos_sim_sgn/ (sim_matrix_SGN.sum(dim=1)-pos_sim_sgn)
        loss = - torch.log((loss1 + loss2)/2).mean()
        return loss
    
    def loss_bias(self, m1, m2, m3):
        T = self.tau
        batch_size, _ = m1.size()
        m1_abs = m1.norm(dim=1)
        m2_aug_abs = m2.norm(dim=1)
        m3_aug_abs = m3.norm(dim=1)
        
        sim_matrix = torch.einsum('ik,jk->ij', m1, m2) / torch.einsum('i,j->ij', m1_abs, m2_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        
        sim_matrix_SGN = torch.einsum('ik,jk->ij', m1, m3) / torch.einsum('i,j->ij', m1_abs, m3_aug_abs)
        sim_matrix_SGN = torch.exp(sim_matrix_SGN / T)
        
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss1 = pos_sim / (sim_matrix.sum(dim=1)-pos_sim)
        pos_sim_sgn = sim_matrix_SGN[range(batch_size), range(batch_size)]
        loss2 = pos_sim_sgn/ (sim_matrix_SGN.sum(dim=1)-pos_sim_sgn)
        loss = - torch.log(self.p * loss1 + (1-self.p) * loss2).mean()
        return loss
