import os

import dgl
import torch
import torch.nn as nn


class UniversalReadout(nn.Module):
    def __init__(self, in_dim):
        super(UniversalReadout, self).__init__()

        self.readout_dim = in_dim // 4

        ####  Layers  ####
        self.phi = Phi(in_dim, dropout=0.3)
        self.rho = Rho(in_dim // 2, dropout=0.3)

    def forward(self, graph, features_name, weights=None):
        graph.ndata["phis"] = self.phi(graph.ndata[features_name])  # With normalization put self.phi(graph.ndata[features_name][:, None, :])
        nodes_sum = dgl.sum_nodes(graph, "phis", weight=weights)
        return self.rho(nodes_sum)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class Phi(nn.Module):
    def __init__(self, in_dim, dropout):
        super(Phi, self).__init__()

        ####  Layers  ####
        self.linear1 = nn.Linear(in_dim, in_dim // 2, bias=False)
        self.linear2 = nn.Linear(in_dim // 2, in_dim // 2, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1)

        ####  Normalization Layers  ####
        # self.LinNorm1 = nn.InstanceNorm1d(int(in_dim/2), eps=1e-05, momentum=0.1)
        # self.LinNorm2 = nn.InstanceNorm1d(int(in_dim/4), eps=1e-05, momentum=0.1)

        ####  Weights Initialization  ####
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        updated_feats = self.linear1(features)
        # updated_feats = self.LinNorm1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.dropout(updated_feats)

        updated_feats = self.linear2(updated_feats)
        # updated_feats = self.LinNorm2(updated_feats)
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
    def __init__(self, in_dim, dropout):
        super(Rho, self).__init__()

        ####  Layers  ####
        self.linear1 = nn.Linear(in_dim, in_dim // 2, bias=False)
        self.linear2 = nn.Linear(in_dim // 2, in_dim // 2, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1)

        ####  Normalization Layers  ####
        # self.LinNorm1 = nn.InstanceNorm1d(1, eps=1e-05, momentum=0.1)
        # self.LinNorm2 = nn.InstanceNorm1d(1, eps=1e-05, momentum=0.1)

        ####  Weights Initialization  ####
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        updated_feats = self.linear1(features)
        # updated_feats = self.LinNorm1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.dropout(updated_feats)

        updated_feats = self.linear2(updated_feats)
        # updated_feats = self.LinNorm2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.dropout(updated_feats)

        return updated_feats

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
