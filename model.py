import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn.conv import HypergraphConv
from torch_geometric.nn.conv import GCNConv
import math
from utils import clones, device


class HyperNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, phi, gcn_dim):
        super(HyperNet, self).__init__()
        self.visit = nn.Linear(in_dim, 1)

        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=False)
        self.h_0 = nn.Parameter(torch.zeros(1, 1, hidden_dim), requires_grad=True).to(device)

        in_dim = hidden_dim
        self.hgc1 = HypergraphConv(in_dim, in_dim)
        self.hgc2 = HypergraphConv(in_dim, in_dim)
        self.hgc3 = HypergraphConv(in_dim, in_dim)

        self.phi = nn.Parameter(torch.tensor(phi).float(), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)
        self.gcn = GCNConv(in_dim, gcn_dim)
        self.dropout = nn.Dropout(p = 0.3)

    def forward(self, x, hyperedge_index, sorted_length):
        batch_size = x.size(0)
        x = self.dropout(x)
        h_0_contig = self.h_0.expand(1, batch_size, self.gru.hidden_size).contiguous()

        gru_out, _ = self.gru(x, h_0_contig)

        out_list = []
        for i in range(gru_out.shape[0]):
            idx = sorted_length[i] - 1
            out_list.append(gru_out[i ,idx ,:])
        input = torch.stack(out_list)

        output1 = self.hgc1(input ,hyperedge_index)
        output1 = output1 + input
        output2 = self.hgc2(output1 ,hyperedge_index)
        output2 = output2 + output1
        output3 = self.hgc3(output2 ,hyperedge_index)
        output3 = output3 + output2

        adj_matrix = torch.mm(output3, output3.T) / torch.pow(torch.tensor(output3.shape[1]).to(device) ,2)
        adj_matrix_out = torch.where(adj_matrix >= self.phi, torch.ones(adj_matrix.shape).to(device)
                                     ,torch.zeros(adj_matrix.shape).to(device))
        edge_index = torch.nonzero(adj_matrix_out == 1).T

        out = self.gcn(output3, edge_index)

        return out


class MLP(nn.Module):
    def __init__(self, in_dim, MLP_dims, drop_prob):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.MLP_dims = MLP_dims

        if MLP_dims == "-":
            middle_layers = []
        else:
            middle_layers = MLP_dims.split("-")
        all_MLP_dimensions = [in_dim]

        for i in middle_layers:
            all_MLP_dimensions.append(int(i))

        all_MLP_dimensions.append(2)
        self.lin_layers_nn = nn.ModuleList()
        for i in range(len(all_MLP_dimensions) - 1):
            self.lin_layers_nn.append(nn.Linear(all_MLP_dimensions[i], all_MLP_dimensions[i + 1]))

        self.dropout = nn.Dropout(p=drop_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)
        input_MLP = x
        for i in range(len(self.lin_layers_nn) - 1):
            input_MLP = self.relu(self.lin_layers_nn[i](input_MLP))

        out = self.lin_layers_nn[-1](input_MLP)

        return out


class MultiMLP(nn.Module):
    def __init__(self, in_dim, MLP_dims, N, drop_prob):
        super(MultiMLP, self).__init__()
        self.MLPs = clones(MLP(in_dim, MLP_dims, drop_prob), N)

    def forward(self, x):
        out = [mlp(x) for mlp in self.MLPs]
        return out


class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, phi, gcn_dim, MLP_dims, N, drop_prob):
        super(Model, self).__init__()
        self.hnet = HyperNet(in_dim, hidden_dim, phi, gcn_dim)

        self.proj = nn.Linear(gcn_dim, N)
        self.mlps = MultiMLP(gcn_dim, MLP_dims, N, drop_prob)
        self.dropout = nn.Dropout(p=drop_prob)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, input, hyperedge_index, sorted_length):
        input = self.dropout(input)
        gcn_out = self.hnet(input, hyperedge_index, sorted_length)
        att = self.softmax(torch.sum(self.proj(gcn_out), 0))
        out_list = self.mlps(gcn_out)

        return out_list, att.cpu().detach().numpy()

