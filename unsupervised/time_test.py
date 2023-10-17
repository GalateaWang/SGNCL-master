# coding=gbk
from torch import nn
import os
import numpy as np
import os.path as osp
import torch
import random
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from torch.optim import Adam, lr_scheduler
from model import Encoder, SGN_CL
from evaluate_embedding import evaluate_embedding
from config import arg_parse, Line_Graph
from t_SNE import draw_plot
args = arg_parse()
import time
from torch_geometric.utils import degree
from torch.optim import Adam, lr_scheduler

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

def test(args, dataloader, model, epoch, accuracies):
    model.eval()
    emb, y=  model.ori_encoder.get_embedding(dataloader)
    draw_plot(emb, y, epoch)
    res = evaluate_embedding(emb, y)
    
    accuracies['epoch'].append(epoch)
    accuracies['val_acc'].append(res[0])
    accuracies['test_acc'].append(res[1])
    accuracies['classifier'].append(res[2])
    print('epoch:{}'.format(epoch), 'val_acc: {},'.format(res[0]), 'test_acc: {},'.format(res[1]), 'classifier:{}'.format(res[2]))

    return res[1],res[0]
    

def main(dataset, mode, epoch, seed, pool, p,lr,layer,batch_size):
    args.dataset = dataset
    args.mode = mode
    args.epoch = epoch
    args.seed = seed
    args.pool = pool
    args.lr = lr
    args.num_layers = layer
    args.batch_size = batch_size
    accuracies = {'epoch': [], 'val_acc' : [], 'test_acc' : [], 'classifier' : []}
    loss_log = {'epoch': [], 'loss': []}
    test_log = []
    
    print("======================================")
    information = "dataset:{},".format(args.dataset) + "base_model:{},".format(args.base_model) + "activation:{},".format(args.activation) + \
          "SGN_mode:{},".format(args.mode) + "hidden_dim:{},".format(args.hidden_dim) +\
          "epoch:{},".format(args.epoch) + "num_layers:{},".format(args.num_layers) +\
          "lr:{},".format(args.lr) + "tau:{},".format(args.tau) + "seed:{},".format(args.seed) + "batch_size:{}".format(args.batch_size)
    print(information)
    with open('./result/{}_{}.log'.format(args.dataset,args.mode), 'a+') as f:
        f.write('{}\n'.format(information))
        f.closed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = osp.expanduser('~/datasets')
    path = osp.join(path,args.dataset)
    dataset = TUDataset(path, name=args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    dataset_eval = TUDataset(path, name=args.dataset).shuffle()
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size)
    train_start_time = time.time()
    if dataset[0].x == None:
        input_dim = 1
    else:
        input_dim = max(dataset.num_features, 1)
    SGN1_start_time = time.time()
    SGN1 = Line_Graph(dataloader, device)
    input_dim1 = 2 * input_dim 
    SGN1_stop_time = time.time()
    SGN2, input_dim2 = None, None
    SGN2_start_time = time.time()
    if args.mode == 2 or args.mode == 3 or args.mode == 4:
        SGN2 = Line_Graph(SGN1, device)
        input_dim2 = 2 * input_dim1
    SGN2_stop_time = time.time()
    encoder_ori = Encoder(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                          base_model = args.base_model, activation = args.activation, pool = args.pool).to(device)
                          
    encoder_SGN1 = Encoder(input_dim=input_dim1, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                          base_model=args.base_model, activation=args.activation, pool = args.pool).to(device)
    encoder_SGN2 = None
    if args.mode == 2  or args.mode == 3 or args.mode == 4:
        encoder_SGN2 = Encoder(input_dim=input_dim2, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                               base_model=args.base_model, activation=args.activation, pool = args.pool).to(device)               
    model = SGN_CL(encoder1=encoder_ori, encoder2=encoder_SGN1, encoder3=encoder_SGN2,
                   hidden_dim=args.hidden_dim, num_layers=args.num_layers, tau=args.tau, mode=args.mode, p=p).to(device)                   
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print('===== after training =====')
    for i in range(1, args.epoch + 1):
        loss = train(dataloader, SGN1, SGN2, model, optimizer)
    train_stop_time = time.time()
        

    with open('./result/text_time.log', 'a+') as f:
        f.write('SGN1_gen_time:{},SGN2_gen_time:{},train_time:{}\n'.format(SGN1_stop_time - SGN1_start_time, SGN2_stop_time - SGN2_start_time, train_stop_time - train_start_time))

if __name__ == '__main__':
  for swt in [2]: 
    for dataset in ["IMDB-BINARY"]:
      for epoch in [40]: 
        pool = "mean"
        log = './result/text_time2.log'
        lr = 0.001
        layer = 4
        batch_size = 64
        with open(log, 'a+') as f:
            f.write('#DATASET:{}\n'.format(dataset))
            f.write("#SGN is {}\n".format(swt))
            f.write("#epoch is {}\n".format(epoch))
            f.write("#######pool is {} #############\n".format(pool))
            f.closed
        ac = 0
        max_acc= 0
        min_acc = 1
        for seed in [0]:
            setup_seed(seed)
            main(dataset, swt, epoch, seed,pool, 0, lr,layer,batch_size)
