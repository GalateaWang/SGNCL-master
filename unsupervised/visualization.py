# coding=gbk
from torch import nn
import os
import numpy as np
import os.path as osp
import torch
import random

from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from model import Encoder, SGN_CL
from evaluate_embedding import evaluate_embedding
from config import arg_parse, Line_Graph
import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
args = arg_parse()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    
def train(dataloader, SGN1, SGN2, model, optimizer):
    
    loss_all = 0
    data_SGN1 = None
    data_SGN2 = None
    i = 0
    model.train()

    for data in dataloader:
        data_ori = data.cuda()
        if args.mode == 1 or args.mode == 3 or args.mode == 4:
            data_SGN1 = SGN1[i]
        if args.mode == 2  or args.mode ==3 or args.mode == 4:
            data_SGN2 = SGN2[i]
        i = i+1    
        optimizer.zero_grad()   
        if  args.mode == 1 or args.mode == 2:
            m1, m2 = model(data_ori, data_SGN1, data_SGN2)
            loss = model.loss(m1, m2)
        elif args.mode == 3:
            m1, m2, m3 = model(data_ori, data_SGN1, data_SGN2)
            loss = model.loss_fu(m1, m2, m3)
        elif args.mode == 4:
            m1, m2, m3 = model(data_ori, data_SGN1, data_SGN2)
            loss = model.loss_bias(m1, m2, m3)
            
        loss_all += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return loss_all /len(dataloader)
    
def similarity_visualization(dataloader, SGN1, SGN2, model, log, sgn,seed,epoch):
    i = 0
    data_SGN1 = None
    data_SGN2 = None
    model.eval()
    for data in dataloader:
        data_ori = data.cuda()
        if args.mode == 1 or args.mode == 3 or args.mode == 4:
            data_SGN1 = SGN1[i]
        if args.mode == 2  or args.mode ==3 or args.mode == 4:
            data_SGN2 = SGN2[i]
        i = i+1  
        if  args.mode == 1 or args.mode == 2:
            m1, m2 = model(data_ori, data_SGN1, data_SGN2)
            batch_size = m1.size()
            m1_abs = m1.norm(dim=1)
            m2_aug_abs = m2.norm(dim=1)
        
            sim_matrix = torch.einsum('ik,jk->ij', m1, m2) / torch.einsum('i,j->ij', m1_abs, m2_aug_abs)
            sim_matrix = np.float32(sim_matrix.tolist())
            sim_matrix = sim_matrix[0:4,0:4]
            data_y = data_ori.y.tolist()
            data_y = data_y[0:4]
            y_index = []
            ii_index = []
            sim_matrix_x = np.empty([len(data_y), len(data_y)], dtype = float)
            for index, i in enumerate(data_y):
                if i == 1:
                    y_index.append(index)
                    ii_index.append(i)
            
            for index, i in enumerate(data_y):
                if i == 0:
                    y_index.append(index)
                    ii_index.append(i)

            
            for i_index, i in enumerate(y_index):
                for j_index, j in enumerate(y_index):
                    sim_matrix_x[i_index][j_index] = sim_matrix[i][j]
            
            batch = data_ori.batch.tolist()
            batch_i = 0
            batch_j = 0
            for index, ss in enumerate(batch):
                if ss == 0 and batch_i == 0:
                    batch_i = index
                if ss == 4:
                    batch_j = index
            xxx = data_ori.x.tolist() 
            xxx = xxx[0:69] 
            with open(log, 'a+') as f:
                f.write('{}\n'.format(data_ori.batch.tolist()))
                f.write('{}\n'.format(batch_i))
                f.write('{}\n'.format(batch_j))
                
                f.write('{}\n'.format(xxx[0:100]))
                f.write('{}\n'.format(data_ori.edge_index.tolist()))
                
                
                f.write('#######i:{}############\n'.format(i))
                f.write('Positive and negative pair similarity\n')
                f.write('sim_matrix\n')
                f.write('{}\n'.format(sim_matrix))
                f.write('sim_matrix_x\n')
                f.write('{}\n'.format(sim_matrix_x))
                f.write('y\n')
                f.write('{}\n'.format(data_ori.y))
                f.write('y_index\n')
                f.write('{}\n'.format(y_index))
                f.write('ii_index\n')
                f.write('{}\n'.format(ii_index))
                
                f.closed
                
            sns.set()
            sns.set_style('whitegrid')
            #norm1 = mpl.colors.Normalize(vmin=-0.5, vmax=0.2)
            df = pd.DataFrame(sim_matrix_x, index =ii_index, columns = ii_index)
            plt.figure(figsize=(10,8))
            sns.heatmap(df,square=False, annot=False,fmt=".2f",cmap="Blues")#, xticklabels=[], yticklabels=[])
            #plt.show()
            plt.savefig("pic/SGN{}_seed{}_epoch{}_small_pic.png".format(sgn,seed,epoch),dpi=300) 
            break

def test(args, dataloader, model, epoch, accuracies):
    model.eval()
    emb, y=  model.ori_encoder.get_embedding(dataloader)
    res = evaluate_embedding(emb, y)
    
    accuracies['epoch'].append(epoch)
    accuracies['val_acc'].append(res[0])
    accuracies['test_acc'].append(res[1])
    accuracies['classifier'].append(res[2])
    print('epoch:{}'.format(epoch), 'val_acc: {},'.format(res[0]), 'test_acc: {},'.format(res[1]), 'classifier:{}'.format(res[2]))

    return res[1],res[0]
    

def main(dataset, mode, epoch, seed, pool, p, log):
    args.dataset = dataset
    args.mode = mode
    args.epoch = epoch
    args.seed = seed
    args.pool = pool
    accuracies = {'epoch': [], 'val_acc' : [], 'test_acc' : [], 'classifier' : []}
    loss_log = {'epoch': [], 'loss': []}
    test_log = []
    
    print("======================================")
    information = "dataset:{},".format(args.dataset) + "base_model:{},".format(args.base_model) + "activation:{},".format(args.activation) + \
          "SGN_mode:{},".format(args.mode) + "hidden_dim:{},".format(args.hidden_dim) +\
          "epoch:{},".format(args.epoch) + "num_layers:{},".format(args.num_layers) +\
          "lr:{},".format(args.lr) + "tau:{},".format(args.tau) + "seed:{},".format(args.seed) + "batch_size:{}".format(args.batch_size)
    print(information)
    with open(log, 'a+') as f:
        f.write('{}\n'.format(information))
        f.closed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = osp.expanduser('~/datasets')
    path = osp.join(path,args.dataset)
    dataset = TUDataset(path, name=args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    if dataset[0].x == None:
        input_dim = 1
    else:
        input_dim = max(dataset.num_features, 1)


    SGN1 = Line_Graph(dataloader, device)
    input_dim1 = 2 * input_dim 
    SGN2, input_dim2 = None, None

    if args.mode == 2 or args.mode == 3 or args.mode == 4:
        SGN2 = Line_Graph(SGN1, device)
        input_dim2 = 2 * input_dim1
        
    print(input_dim,'/',input_dim1,'/',input_dim2)
    encoder_ori = Encoder(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                          base_model = args.base_model, activation = args.activation, pool = args.pool).to(device)
                          
    encoder_SGN1 = Encoder(input_dim=input_dim1, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                          base_model=args.base_model, activation=args.activation, pool = args.pool).to(device)
    encoder_SGN2 = None
    if args.mode == 2  or args.mode == 3 or args.mode == 4:
        encoder_SGN2 = Encoder(input_dim=input_dim2, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                               base_model=args.base_model, activation=args.activation, pool = args.pool).to(device)
    print("mode:{},p:{}".format(args.mode,p))                  
    model = SGN_CL(encoder1=encoder_ori, encoder2=encoder_SGN1, encoder3=encoder_SGN2,
                   hidden_dim=args.hidden_dim, num_layers=args.num_layers, tau=args.tau, mode=args.mode, p=p).to(device)                   
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    next_loss = 1e5
    count = 0
    similarity_visualization(dataloader, SGN1, SGN2, model, log, args.mode, args.seed, 0)
    print('===== after training =====')
    for i in range(1, args.epoch + 1):
        loss = train(dataloader, SGN1, SGN2, model, optimizer)
        loss_log['epoch'].append(i)
        loss_log['loss'].append(loss)
        print('Epoch {}, Loss {}'.format(i, loss))
        
        if i % 10 == 0:
            test_acc, val_acc = test(args, dataloader, model, i, accuracies)
            test_log.append(test_acc) 
            similarity_visualization(dataloader, SGN1, SGN2, model, log, args.mode, args.seed, i)

    return test_acc, test_log

if __name__ == '__main__':
  for SGN in [1,2,3]:
      for epoch in [100]: 
        dataset = 'MUTAG'
        pool = "mean"
        for seed in [4]:
            log = './result/visualization_{}_'.format(dataset)+'{}_seed_{}.log'.format(SGN,seed)
            with open(log, 'a+') as f:
                f.write('#######DATASET:{}############\n'.format(dataset))
                f.write("########SGN is {} #############\n".format(SGN))
                f.closed
            setup_seed(seed)
            test_acc, test_log = main(dataset, SGN, epoch, seed,pool, 0, log)
            with open(log, 'a+') as f:
                f.write("########seed:{}, test acc is {}#############\n".format(seed, test_acc))
                f.closed


