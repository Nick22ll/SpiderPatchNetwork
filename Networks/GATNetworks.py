import os

import dgl
import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GATConv, SortPooling

from Networks.MLP import GenericMLP
from Networks.SpiralReadout import SpiralReadout
from Networks.UniversalReadout import UniversalReadout


class GMReader2GATSortPoolReadout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, pool_dim, dropout):
        super(GMReader2GATSortPoolReadout, self).__init__()

        ####  Layers  ####
        self.GAT1 = GATConv(in_feats=in_dim, out_feats=hidden_dim, num_heads=4, feat_drop=dropout, attn_drop=dropout, residual=False, bias=False)
        self.GAT2 = GATConv(in_feats=hidden_dim * 4, out_feats=hidden_dim, num_heads=2, feat_drop=dropout, attn_drop=dropout, residual=False, bias=False)
        self.classifier = GenericMLP(pool_dim * 2 * hidden_dim, pool_dim * hidden_dim, out_dim, dropout)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormGAT1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormGAT2 = GraphNorm(hidden_dim, eps=1e-5)

        #### Readout Layer ####
        self.readout = SortPooling(k=pool_dim)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, mesh_graph, features):
        updated_feats = self.GAT1(mesh_graph, features)
        updated_feats = self.GraphNormGAT1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        readouts = self.readout(mesh_graph, updated_feats.mean(1))

        updated_feats = updated_feats.flatten(1)

        updated_feats = self.GAT2(mesh_graph, updated_feats)
        updated_feats = self.GraphNormGAT2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        readouts = torch.hstack((readouts, self.readout(mesh_graph, updated_feats.mean(1))))

        return self.classifier(readouts)


class GMReader2GATUniversalReadout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(GMReader2GATUniversalReadout, self).__init__()

        ####  Layers  ####
        self.GAT1 = GATConv(in_feats=in_dim, out_feats=hidden_dim, num_heads=4, feat_drop=dropout, attn_drop=dropout, residual=False, bias=False)
        self.GAT2 = GATConv(in_feats=hidden_dim * 4, out_feats=hidden_dim, num_heads=2, feat_drop=dropout, attn_drop=dropout, residual=False, bias=False)
        self.classifier = GenericMLP(hidden_dim, hidden_dim * 2, out_dim, dropout)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormGAT1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormGAT2 = GraphNorm(hidden_dim, eps=1e-5)

        #### Readout Layer ####
        self.readout1 = UniversalReadout(hidden_dim, hidden_dim * 2, int(hidden_dim / 2), dropout=0.05)
        self.readout2 = UniversalReadout(hidden_dim, hidden_dim * 2, int(hidden_dim / 2), dropout=0.05)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, mesh_graph, features):
        updated_feats = self.GAT1(mesh_graph, features)
        updated_feats = self.GraphNormGAT1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with mesh_graph.local_scope():
            mesh_graph.ndata["readout"] = updated_feats.mean(1)
            readouts = self.readout1(mesh_graph, "readout")

        updated_feats = updated_feats.flatten(1)

        updated_feats = self.GAT2(mesh_graph, updated_feats)
        updated_feats = self.GraphNormGAT2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with mesh_graph.local_scope():
            mesh_graph.ndata["readout"] = updated_feats.mean(1)
            readouts = torch.hstack((readouts, self.readout2(mesh_graph, "readout")))

        return self.classifier(readouts)


def save(self, path):
    torch.save(self.state_dict(), path)


def load(self, path):
    self.load_state_dict(torch.load(path))
    self.eval()


class GATMeshNetworkSRPR(nn.Module):
    def __init__(self, readout_dim, hidden_dim, out_feats, dropout, mesh_graph_edge_weights=True):
        super(GATMeshNetworkSRPR, self).__init__()

        ####  Variables  ####
        self.name = "GATMeshNetworkSRPR"
        self.readout_dim = readout_dim
        self.mesh_graph_edge_weights = mesh_graph_edge_weights

        ####  Layers  ####
        self.readout = SpiralReadout(readout_dim)
        self.mesh_reader = GMReader2GATSortPoolReadout(in_dim=self.readout_dim, hidden_dim=hidden_dim, out_dim=out_feats, dropout=dropout, pool_dim=10)

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.readout_dim), device=device)
        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(patches, batch_size=int(mesh_graph.num_nodes() / 2), drop_last=False)
        for spider_patch in dataloader:
            readouts = torch.vstack((readouts, self.readout(spider_patch, "aggregated_feats")))
        return self.mesh_reader(mesh_graph, readouts)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class GATMeshNetworkSRUR(nn.Module):
    def __init__(self, readout_dim, hidden_dim, out_feats, dropout, mesh_graph_edge_weights=True):
        super(GATMeshNetworkSRUR, self).__init__()

        ####  Variables  ####
        self.name = "GATMeshNetworkSRUR"
        self.readout_dim = readout_dim
        self.mesh_graph_edge_weights = mesh_graph_edge_weights

        ####  Layers  ####
        self.readout = SpiralReadout(readout_dim)
        self.mesh_reader = GMReader2GATUniversalReadout(in_dim=self.readout_dim, hidden_dim=hidden_dim, out_dim=out_feats, dropout=dropout)

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.readout_dim), device=device)
        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(patches, batch_size=int(mesh_graph.num_nodes() / 2), drop_last=False)
        for spider_patch in dataloader:
            readouts = torch.vstack((readouts, self.readout(spider_patch, "aggregated_feats")))

        return self.mesh_reader(mesh_graph, readouts)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
