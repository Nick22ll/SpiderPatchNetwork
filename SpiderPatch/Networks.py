import os

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GraphConv


class PatchConv2LayerClassifier(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout):
        super(PatchConv2LayerClassifier, self).__init__()
        self.patchConv = PatchConv2Layer(in_feats, hidden_dim, int(hidden_dim / 2))
        self.linear = nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4), bias=False)
        self.classifier = nn.Linear(int(hidden_dim / 4), out_feats, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, node_feats, edge_weight):
        updated_feats = F.relu(self.patchConv(g, node_feats, edge_weight))
        updated_feats = F.relu(self.linear(updated_feats))
        updated_feats = self.dropout(updated_feats)
        return self.classifier(updated_feats)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class PatchConv2Layer(nn.Module):
    def __init__(self, in_feats, hidden_dim, readout_dim):
        super(PatchConv2Layer, self).__init__()

        self.leaky_slope = 0.01

        self.conv1 = GraphConv(in_feats, hidden_dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.leaky_slope, mode='fan_in', nonlinearity='leaky_relu')

        self.conv2 = GraphConv(hidden_dim, readout_dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.leaky_slope, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, g, updated_feats, edge_weight):
        updated_feats = F.leaky_relu(self.conv1(g, updated_feats, edge_weight=edge_weight), self.leaky_slope)
        updated_feats = F.leaky_relu(self.conv2(g, updated_feats, edge_weight=edge_weight), self.leaky_slope)
        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            # Calculate graph representation by average readout.
            return dgl.mean_nodes(g, 'updated_feats')

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class PatchConv3LayerClassifier(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout):
        super(PatchConv3LayerClassifier, self).__init__()
        self.patchConv = PatchConv3Layer(in_feats, hidden_dim, int(hidden_dim / 2))
        self.linear = nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4), bias=False)
        self.classifier = nn.Linear(int(hidden_dim / 4), out_feats, bias=False)

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


class PatchConv3Layer(nn.Module):
    def __init__(self, in_feats, hidden_dim, readout_dim):
        super(PatchConv3Layer, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, int(hidden_dim / 2), bias=False)
        self.conv3 = GraphConv(int(hidden_dim / 2), readout_dim, bias=False)

    def forward(self, g, updated_feats, edge_weight):
        updated_feats = F.lerelu(self.conv1(g, updated_feats, edge_weight=edge_weight))
        updated_feats = F.relu(self.conv2(g, updated_feats, edge_weight=edge_weight))
        updated_feats = F.relu(self.conv3(g, updated_feats, edge_weight=edge_weight))
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
    def __init__(self, patch_feat_dim=18, internal_hidden_dim=128, readout_dim=16, hidden_dim=32, out_feats=15):
        super(MeshNetwork, self).__init__()
        self.patch_reader = PatchConv2LayerClassifier(patch_feat_dim, hidden_dim=internal_hidden_dim, out_feats=readout_dim, dropout=0.5)
        self.mesh_reader = GraphMeshConvolution(in_dim=readout_dim, hidden_dim=hidden_dim, out_dim=out_feats, dropout=0.5)

        self.readout_dim = readout_dim

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.readout_dim), device=device)
        dataloader = GraphDataLoader(patches, batch_size=5, drop_last=False)  # int(len(patches) / 2)
        for patch in dataloader:
            readouts = torch.vstack((readouts, self.patch_reader(patch, patch.ndata["aggregated_feats"], patch.edata["weights"])))
        # readout_mean = torch.mean(readouts, dim=0)
        return self.mesh_reader(mesh_graph, readouts), readouts  # readout_mean

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class GraphMeshConvolution(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(GraphMeshConvolution, self).__init__()
        self.leaky_slope = 0.01

        self.conv1 = GraphConv(in_dim, hidden_dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.leaky_slope, mode='fan_in', nonlinearity='leaky_relu')

        self.conv2 = GraphConv(hidden_dim, int(hidden_dim / 2), bias=False)
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.leaky_slope, mode='fan_in', nonlinearity='leaky_relu')

        self.classify = nn.Linear(int(hidden_dim / 2), out_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, mesh_graph, features):
        updated_feats = F.leaky_relu(self.conv1(mesh_graph, features), self.leaky_slope)
        updated_feats = F.leaky_relu(self.conv2(mesh_graph, updated_feats), self.leaky_slope)
        updated_feats = self.dropout(updated_feats)
        with mesh_graph.local_scope():
            mesh_graph.ndata['updated_feats'] = updated_feats
            # Calculate graph representation by average readout.
            return self.classify(dgl.mean_nodes(mesh_graph, 'updated_feats'))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class MeshNetworkSimilGAN:
    def __init__(self, patch_feat_dim, readout_dim, hidden_dim, out_feats, device):
        self.device = device
        self.patch_reader = PatchConv2LayerClassifier(patch_feat_dim, hidden_dim=100, out_feats=readout_dim).to(device)
        self.mesh_reader = GraphMeshConvolution(in_dim=readout_dim, hidden_dim=hidden_dim, out_dim=out_feats).to(device)

        self.readout_dim = readout_dim

        self.patch_optimizer = torch.optim.Adam(self.patch_reader.parameters(), lr=0.01)
        self.mesh_optimizer = torch.optim.Adam(self.mesh_reader.parameters(), lr=0.0001)

    def patch_train_step(self, mesh_graph, patches, label, device):
        self.patch_reader.zero_grad(set_to_none=True)
        readouts = torch.empty((0, self.readout_dim), device=device)
        dataloader = GraphDataLoader(patches, batch_size=20, drop_last=False)
        for patch in dataloader:
            readouts = torch.vstack((readouts, self.patch_reader(patch, patch.ndata["aggregated_feats"], patch.edata["weights"])))
        classifications = self.mesh_reader(mesh_graph, readouts)
        loss = F.cross_entropy(classifications, label)
        loss.backward()
        self.patch_optimizer.step()
        return loss.item()

    def mesh_train_step(self, mesh_graph, patches, label, device):
        self.mesh_reader.zero_grad(set_to_none=True)
        readouts = torch.empty((0, self.readout_dim), device=device)
        dataloader = GraphDataLoader(patches, batch_size=20, drop_last=False)
        for patch in dataloader:
            readouts = torch.vstack((readouts, self.patch_reader(patch, patch.ndata["aggregated_feats"], patch.edata["weights"])))
        classifications = self.mesh_reader(mesh_graph, readouts)
        loss = F.cross_entropy(classifications, label)
        loss.backward()
        self.mesh_optimizer.step()
        return loss.item()

    def train_step(self, mesh_graph, patches, label, device):
        loss_mesh = self.mesh_train_step(mesh_graph, patches, label, device)
        loss_patch = self.patch_train_step(mesh_graph, patches, label, device)
        return loss_patch, loss_mesh

    def evaluate(self, mesh_graph, patches):
        self.patch_reader.zero_grad()
        readouts = torch.empty((0, self.readout_dim), device=self.device)
        dataloader = GraphDataLoader(patches, batch_size=20, drop_last=False)
        for patch in dataloader:
            readouts = torch.vstack((readouts, self.patch_reader(patch, patch.ndata["aggregated_feats"], patch.edata["weights"])))
        return self.mesh_reader(mesh_graph, readouts)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.patch_reader.state_dict(), path + "/patch_reader.pt")
        os.makedirs(path, exist_ok=True)
        torch.save(self.mesh_reader.state_dict(), path + "/mesh_reader.pt")

    def load(self, path):
        self.patch_reader.load_state_dict(torch.load(path + "/patch_reader.pt"))
        self.patch_reader.eval()
        self.mesh_reader.load_state_dict(torch.load(path + "/mesh_reader.pt"))
        self.mesh_reader.eval()


class MeshReadoutNetwork(nn.Module):
    def __init__(self, readout_dim, hidden_dim, num_classes):
        super(MeshReadoutNetwork, self).__init__()
        self.conv1 = GraphConv(readout_dim, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, bias=False)
        self.classify = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(self, MG, feats):
        updated_feats = F.relu(self.conv1(MG, feats))
        updated_feats = F.relu(self.conv2(MG, updated_feats))
        with MG.local_scope():
            MG.ndata['updated_feats'] = updated_feats
            # Calculate graph representation by average readout.
            return self.classify(dgl.mean_nodes(MG, 'updated_feats'))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
