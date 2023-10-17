import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_add
import torch.nn.functional as F
num_atom_type = 120
num_chirality_tag = 3
num_bond_type = 6
num_bond_direction = 3

#sgn1_x_type1 = 6
#sgn1_x_type2 = 3
sgn1_x_type1 = 360
sgn1_x_type2 = 360
sgn1_edge_type1 = 4320
sgn1_edge_type2 = 27

class GINConv(MessagePassing):
    def __init__(self, emb_dim, mode = 0):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.mode = mode                               
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)
        
        self.sgn1_edge_embedding1 = torch.nn.Embedding(sgn1_edge_type1, emb_dim)
        self.sgn1_edge_embedding2 = torch.nn.Embedding(sgn1_edge_type2, emb_dim)
        
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.sgn1_edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.sgn1_edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))
        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        if self.mode == 0:
            self_loop_attr[:,0] = 4 #bond type for self-loop edge
        elif self.mode == 1:
            self_loop_attr[:,0] = 3360
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        if self.mode == 0:
            edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
            #print("edge_embeddings_mode=0:/n", edge_embeddings.size())
            #print("x=0:\n", x.size())
        elif self.mode == 1:
            edge_embeddings = self.sgn1_edge_embedding1(edge_attr[:, 0]) + self.sgn1_edge_embedding2(edge_attr[:, 1])
            #print("edge_embeddings_mode=1:\n", edge_embeddings.size())
            #print("x=1:\n", x.size())
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin", mode = 0):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.mode = mode
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        
        self.x_sgn1_embedding1 = torch.nn.Embedding(sgn1_x_type1, emb_dim)
        self.x_sgn1_embedding2 = torch.nn.Embedding(sgn1_x_type2, emb_dim)
        
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.x_sgn1_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_sgn1_embedding2.weight.data)
        
        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, mode = self.mode))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")
        if self.mode == 0:
            x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        elif self.mode == 1:
            x = self.x_sgn1_embedding1(x[:, 0]) + self.x_sgn1_embedding2(x[:, 1])
        h_list = [x]

        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation
        
class GNN_graphpred(torch.nn.Module):

    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        ## GNN参数：num_layer, emb_dim, JK, drop_ratio, gnn_type="gin", mode = 0
        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type, mode = 0)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")    
        node_representation = self.gnn(x, edge_index, edge_attr)
        return self.graph_pred_linear(self.pool(node_representation, batch))
