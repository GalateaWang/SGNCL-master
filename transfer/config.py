import argparse

def arg_phase():
    parser = argparse.ArgumentParser(description='pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=4, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 80)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0, help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last", help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',  help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help="Random Seed")
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument('--mode1', type=int, default=0, help='we can choose mode: 1,2,3')
    parser.add_argument('--mode2', type=int, default=1, help='we can choose mode: 1,2,3')
    parser.add_argument('--pool', type=str, default="mean", help='we can choose mode: mean, sum, max')
    parser.add_argument('--save_model_path', type=str, default="./models_sgncl/sgngcl")
    parser.add_argument('--loss_temp', type=float, default=0.1)
    parser.add_argument('--lamda', type=float, default=0.1)

    args = parser.parse_args()
    return args
    
    
def arg_phase_v1():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=1, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1, help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean", help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last", help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default='bbbp', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default="models_sgncl/sgngcl_seed0_10", help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')

    args = parser.parse_args()  
    return args 
