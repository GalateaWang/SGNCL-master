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
        if args.mode == 2 or args.mode ==3 or args.mode == 4:
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

def test(dataloader, model, epoch, accuracies):
    model.eval()
    emb, y=  model.ori_encoder.get_embedding(dataloader)
    res = evaluate_embedding(emb, y)
    
    accuracies['epoch'].append(epoch)
    accuracies['val_acc'].append(res[0])
    accuracies['test_acc'].append(res[1])
    accuracies['classifier'].append(res[2])
    print('epoch:{}'.format(epoch), 'val_acc: {},'.format(res[0]), 'test_acc: {},'.format(res[1]), 'classifier:{}'.format(res[2]))

    return res[1],res[0]
    

def main():
    setup_seed(args.seed)
    accuracies = {'epoch': [], 'val_acc' : [], 'test_acc' : [], 'classifier' : []}
    loss_log = {'epoch': [], 'loss': []}
    test_log = []

    information = "dataset:{},".format(args.dataset) + "base_model:{},".format(args.base_model) + "activation:{},".format(args.activation) + \
          "SGN_mode:{},".format(args.mode) + "hidden_dim:{},".format(args.hidden_dim) +\
          "epoch:{},".format(args.epoch) + "num_layers:{},".format(args.num_layers) +\
          "lr:{},".format(args.lr) + "tau:{},".format(args.tau) + "seed:{},".format(args.seed) + "batch_size:{}".format(args.batch_size)
    # print(information)
    with open('./result/{}'.format(args.dataset)+'_{}'.format(args.mode)+'_seed{}.log'.format(args.seed), 'a+') as f:
        f.write('{}\n'.format(information))
        f.closed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = osp.expanduser('~/datasets')
    path = osp.join(path,args.dataset)

    dataset = TUDataset(path, name=args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    dataset_eval = TUDataset(path, name=args.dataset).shuffle()
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size)

    if dataset[0].x == None:
        input_dim = 1
    else:
        input_dim = max(dataset.num_features, 1)
    input_dim1 = 2 * input_dim
    SGN2, input_dim2 = None, None
       
    if os.path.isfile("./param/{}_SGN1_{}.pth".format(dataset, args.seed)):
        SGN1 = torch.load("./param/{}_SGN1_{}.pth".format(dataset, args.seed))
    else:
        SGN1 = Line_Graph(dataloader, device)
        torch.save(SGN1,"./param/{}_SGN1_{}.pth".format(dataset, args.seed))

    if args.mode == 2 or args.mode == 3 or args.mode == 4:
        if os.path.isfile("./param/{}_SGN2_{}.pth".format(dataset, args.seed)):
            SGN2 = torch.load("./param/{}_SGN2_{}.pth".format(dataset, args.seed))
        else:
            SGN2 = Line_Graph(SGN1, device)
            torch.save(SGN2,"./param/{}_SGN2_{}.pth".format(dataset, args.seed))
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

    print("mode:{},p:{}".format(args.mode,args.p))

    model = SGN_CL(encoder1=encoder_ori, encoder2=encoder_SGN1, encoder3=encoder_SGN2,
                   hidden_dim=args.hidden_dim, num_layers=args.num_layers, tau=args.tau, mode=args.mode, p=args.p).to(device)                   

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    '''
    model.eval()
    emb, y =  model.ori_encoder.get_embedding(dataloader_eval)
    print('===== Before training =====')
    res = evaluate_embedding(emb, y)  
    accuracies['epoch'].append(0)
    accuracies['val_acc'].append(res[0])
    accuracies['test_acc'].append(res[1])
    accuracies['classifier'].append(res[2])
    print('val_acc: {},'.format(res[0]), 'test_acc: {},'.format(res[1]), 'classifier:{}'.format(res[2]))
    '''

    print('===== after training =====')
    for i in range(1, args.epoch + 1):
        loss = train(dataloader, SGN1, SGN2, model, optimizer)
        print('Epoch {}, Loss {}'.format(i, loss))
        loss_log['epoch'].append(i)
        loss_log['loss'].append(loss)

        if i % 10 == 0:
            test_acc, val_acc = test(dataloader_eval, model, i, accuracies)
            test_log.append(test_acc) 

    with open('./result/{}'.format(args.dataset)+'_{}'.format(args.mode)+'_seed{}.log'.format(args.seed), 'a+') as f:
        f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(args.dataset, args.base_model, args.p,
                                                      args.mode, args.lr, args.num_layers,
                                                      args.epoch, args.seed, accuracies, loss_log))
        f.write('=====##test_acc: {}\n'.format(test_acc))
    return test_acc, test_log

if __name__ == '__main__':
    log ='./result/{}'.format(args.dataset)+'_{}'.format(args.mode)+'_seed{}.log'
    with open(log, 'a+') as f:
        f.write('#######DATASET:{}############\n'.format(args.dataset))
        f.write("########SGN is {} #############\n".format(args.mode))
        f.write("########epoch is {} #############\n".format(args.epoch))
        f.write("#######pool is {} #############\n".format(args.pool))
        f.write("#######p is {} #############\n".format(args.p)) # When sgn_mode is 4, adjusting p is valid
        f.closed

    setup_seed(args.seed)
    test_acc, test_log = main()




