import os

import dgl
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_dim, dropout):
        super(MLP, self).__init__()

        ####  Layers  ####
        self.in_linear = nn.Linear(in_feats, hidden_dim, bias=False)
        self.hidden_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim, out_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.LinNorm1 = nn.InstanceNorm1d(hidden_dim, eps=1e-05, momentum=0.1)
        self.LinNorm2 = nn.InstanceNorm1d(int(hidden_dim / 4), eps=1e-05, momentum=0.1)

        ####  Weights Initialization  ####
        torch.nn.init.xavier_uniform_(self.in_linear.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.hidden_linear.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, edge_weight):
        # Select the node features of seed_point as updated feats
        num_nodes = g.batch_num_nodes()
        features_number = len(g.ndata["aggregated_feats"][0])
        updated_feats = torch.empty((0, int(features_number * num_nodes[0])), device=g.ndata["aggregated_feats"][0].device)
        last_idx = 0
        for i in range(g.batch_size):
            unbatched_graph = g.ndata["aggregated_feats"][last_idx: last_idx + num_nodes[i]].reshape((1, -1))
            updated_feats = torch.vstack((updated_feats, unbatched_graph))
            last_idx += num_nodes[i]
        # updated_feats = dgl.mean_nodes(g, "aggregated_feats", "weight")

        # FIRST LAYER
        updated_feats = self.in_linear(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.LinNorm1(updated_feats)
        updated_feats = self.dropout(updated_feats)

        # SECOND LAYER
        updated_feats = self.hidden_linear(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.LinNorm2(updated_feats)
        updated_feats = self.dropout(updated_feats)

        # CLASSIFIER
        return self.classifier(updated_feats)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
