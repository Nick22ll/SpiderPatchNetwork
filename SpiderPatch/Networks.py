import os

import dgl
import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GraphConv


class PatchConv1LayerClassifier(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout):
        super(PatchConv1LayerClassifier, self).__init__()

        ####  Layers  ####
        self.patchConv = PatchConv1Layer(in_feats, hidden_dim)
        self.linear = nn.Linear(hidden_dim, int(hidden_dim / 2), bias=False)
        self.classifier = nn.Linear(int(hidden_dim / 2), out_feats, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Weights Initialization  ####
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, node_feats, edge_weight):
        updated_feats = self.patchConv(g, node_feats, edge_weight)
        updated_feats = self.linear(updated_feats)
        updated_feats = self.dropout(updated_feats)
        return self.classifier(updated_feats)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class PatchConv2LayerClassifier(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout):
        super(PatchConv2LayerClassifier, self).__init__()

        ####  Layers  ####
        self.patchConv = PatchConv2Layer(in_feats, hidden_dim, int(hidden_dim / 2))
        self.linear = nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4), bias=False)
        self.classifier = nn.Linear(int(hidden_dim / 4), out_feats, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Weights Initialization  ####
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, node_feats, edge_weight):
        updated_feats = self.patchConv(g, node_feats, edge_weight)
        updated_feats = self.linear(updated_feats)
        updated_feats = self.dropout(updated_feats)
        return self.classifier(updated_feats)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class PatchConv1Layer(nn.Module):
    def __init__(self, in_feats, readout_dim):
        super(PatchConv1Layer, self).__init__()

        ####  Layers  ####
        self.conv = GraphConv(in_feats, readout_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, g, updated_feats, edge_weight):
        updated_feats = self.conv(g, updated_feats, edge_weight=edge_weight)
        updated_feats = self.LeakyReLU(updated_feats)
        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            # Calculate graph representation by average readout.
            return dgl.mean_nodes(g, 'updated_feats')

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class PatchConv2Layer(nn.Module):
    def __init__(self, in_feats, hidden_dim, readout_dim):
        super(PatchConv2Layer, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_feats, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, readout_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, g, updated_feats, edge_weight):
        updated_feats = self.conv1(g, updated_feats, edge_weight=edge_weight)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.conv2(g, updated_feats, edge_weight=edge_weight)
        updated_feats = self.LeakyReLU(updated_feats)
        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            # Calculate graph representation by average readout.
            return dgl.mean_nodes(g, 'updated_feats')

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class MeshNetwork(nn.Module):
    def __init__(self, patch_feat_dim=18, internal_hidden_dim=128, readout_dim=32, hidden_dim=64, out_feats=15):
        super(MeshNetwork, self).__init__()
        ####  Layers  ####
        self.patch_reader = PatchConv2LayerClassifier(patch_feat_dim, hidden_dim=internal_hidden_dim, out_feats=readout_dim, dropout=0.5)
        self.mesh_reader = GraphMesh2ConvClassifier(in_dim=readout_dim, hidden_dim=hidden_dim, out_dim=out_feats, dropout=0.5)
        # self.patch_reader = PatchConv1LayerClassifier(patch_feat_dim, hidden_dim=internal_hidden_dim, out_feats=readout_dim, dropout=0.5)
        # self.mesh_reader = GraphMesh1ConvClassifier(in_dim=readout_dim, hidden_dim=hidden_dim, out_dim=out_feats, dropout=0.5)

        ####  Variables  ####
        self.readout_dim = readout_dim

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.readout_dim), device=device)
        dataloader = GraphDataLoader(patches, batch_size=10, drop_last=False)
        for patch in dataloader:
            readouts = torch.vstack((readouts, self.patch_reader(patch, patch.ndata["aggregated_feats"], patch.edata["weights"])))
        return self.mesh_reader(mesh_graph, readouts), readouts

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class GraphMesh1ConvClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(GraphMesh1ConvClassifier, self).__init__()

        ####  Layers  ####
        self.conv = GraphConv(in_dim, hidden_dim, bias=False)
        self.linear = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.classifier = nn.Linear(int(hidden_dim / 2), out_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, mesh_graph, features):
        updated_feats = self.conv(mesh_graph, features)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.linear(updated_feats)
        updated_feats = self.dropout(updated_feats)
        with mesh_graph.local_scope():
            mesh_graph.ndata['updated_feats'] = updated_feats
            # Calculate graph representation by average readout.
            return self.classifier(dgl.mean_nodes(mesh_graph, 'updated_feats'))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class GraphMesh2ConvClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(GraphMesh2ConvClassifier, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_dim, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, int(hidden_dim / 2), bias=False)
        self.linear = nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4))
        self.classifier = nn.Linear(int(hidden_dim / 4), out_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, mesh_graph, features):
        updated_feats = self.conv1(mesh_graph, features)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.conv2(mesh_graph, updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.linear(updated_feats)
        updated_feats = self.dropout(updated_feats)
        with mesh_graph.local_scope():
            mesh_graph.ndata['updated_feats'] = updated_feats
            # Calculate graph representation by average readout.
            return self.classifier(dgl.mean_nodes(mesh_graph, 'updated_feats'))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
