import os

import dgl
import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm
from dgl.nn.pytorch import GraphConv


class SpiralMeshReader(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SpiralMeshReader, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_dim, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim, out_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(hidden_dim, eps=1e-5)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

    def forward(self, mesh_graph, edge_weights):
        updated_feats = self.conv1(mesh_graph, mesh_graph.ndata["aggregated_feats"], edge_weight=edge_weights)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.conv2(mesh_graph, updated_feats, edge_weight=edge_weights)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormConv2(updated_feats)
        with mesh_graph.local_scope():
            mesh_graph.ndata['updated_feats'] = updated_feats
            # Calculate graph representation by average readout.
            updated_feats = dgl.mean_nodes(mesh_graph, 'updated_feats')
            return self.classifier(updated_feats)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
