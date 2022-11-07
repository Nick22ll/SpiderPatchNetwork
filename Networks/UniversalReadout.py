import os

import torch
import torch.nn as nn


class UniversalReadout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(UniversalReadout, self).__init__()
        self.readout_dim = out_dim

        ####  Layers  ####
        self.phi = Phi(in_dim, hidden_dim, dropout)
        self.rho = Rho(hidden_dim, out_dim, dropout)

    def forward(self, graph, features_name):
        num_nodes = graph.batch_num_nodes()
        readout = torch.empty((0, self.readout_dim), device=graph.ndata[features_name][0].device)
        last_idx = 0
        for i in range(graph.batch_size):
            phis = self.phi(graph.ndata[features_name][last_idx: last_idx + num_nodes[i]])
            sums = torch.sum(phis, dim=0).view((1, -1))
            readout = torch.vstack((readout, self.rho(sums)))
            last_idx += num_nodes[i]
        return readout

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class Phi(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(Phi, self).__init__()

        ####  Layers  ####
        self.linear1 = nn.Linear(in_dim, in_dim * 2, bias=False)
        self.linear2 = nn.Linear(in_dim * 2, out_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.LinNorm1 = nn.InstanceNorm1d(1, eps=1e-05, momentum=0.1)
        self.LinNorm2 = nn.InstanceNorm1d(1, eps=1e-05, momentum=0.1)

        ####  Weights Initialization  ####
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        updated_feats = self.linear1(features)
        updated_feats = self.LinNorm1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        updated_feats = self.linear2(updated_feats)
        updated_feats = self.LinNorm2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        updated_feats = self.dropout(updated_feats)

        return updated_feats

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class Rho(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(Rho, self).__init__()

        ####  Layers  ####
        self.linear1 = nn.Linear(in_dim, in_dim * 2, bias=False)
        self.linear2 = nn.Linear(in_dim * 2, out_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.LinNorm1 = nn.InstanceNorm1d(1, eps=1e-05, momentum=0.1)
        self.LinNorm2 = nn.InstanceNorm1d(1, eps=1e-05, momentum=0.1)

        ####  Weights Initialization  ####
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        updated_feats = self.linear1(features)
        updated_feats = self.LinNorm1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        updated_feats = self.linear2(updated_feats)
        updated_feats = self.LinNorm2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        updated_feats = self.dropout(updated_feats)

        return updated_feats

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
