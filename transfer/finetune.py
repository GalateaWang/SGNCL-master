from config import arg_phase_v1
args = arg_phase_v1()
import os.path as osp
from loader_v1 import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

import os
import shutil


criterion = nn.BCEWithLogitsLoss(reduction="none")

def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]


def main():

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    # device
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    if args.dataset == "tox21":
        num_tasks = 12
        epoch_num = 100
    elif args.dataset == "hiv":
        num_tasks = 1
        epoch_num = 100
    elif args.dataset == "pcba":
        num_tasks = 128
        epoch_num = 100
    elif args.dataset == "muv":
        num_tasks = 17
        epoch_num = 50
    elif args.dataset == "bace":
        num_tasks = 1
        epoch_num = 200
    elif args.dataset == "bbbp":
        num_tasks = 1
        epoch_num = 100
    elif args.dataset == "toxcast":
        num_tasks = 617
        epoch_num = 100
    elif args.dataset == "sider":
        num_tasks = 27
        epoch_num = 100
    elif args.dataset == "clintox":
        num_tasks = 2
        epoch_num = 300
    else:
        raise ValueError("Invalid dataset name.")

    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    dataset = MoleculeDataset(path, dataset=args.dataset)

    if args.split == "scaffold":
        smiles_list = pd.read_csv( path + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv(path + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
                          graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    model.to(device)

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    for epoch in range(1, epoch_num + 1):
        print("====epoch " + str(epoch))

        train(model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(model, device, val_loader)
        test_acc = eval(model, device, test_loader)

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        print("train: %.4f val: %.4f test: %.4f" % (train_acc, val_acc, test_acc))
    max_test_acc = max(test_acc_list)
    print("test_max: %.4f" % max_test_acc)
    with open('{}_result.log'.format(args.dataset), 'a+') as f:
        f.write(args.dataset)
        f.write('\n')
        f.write("{}".format(test_acc_list))
        f.write('\n')
        f.write(args.dataset + ' ' + str(args.runseed) + ' ' + str(np.array(test_acc_list)[-1]) + ' ' + str(max_test_acc)+"({})".format(test_acc_list.index(max_test_acc)))
        f.write('\n')
    return np.array(test_acc_list)[-1], max_test_acc

if __name__ == "__main__":
    last_acc, max_acc = main()

