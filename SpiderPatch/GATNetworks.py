import os

import dgl
import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GATConv


class PatchReader2GATLayer(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout):
        super(PatchReader2GATLayer, self).__init__()

        ####  Layers  ####
        self.GAT1 = GATConv(in_feats=in_feats, out_feats=hidden_dim, num_heads=4, feat_drop=dropout, attn_drop=dropout, residual=False, bias=False)
        self.GAT2 = GATConv(in_feats=hidden_dim * 4, out_feats=64, num_heads=3, feat_drop=dropout, attn_drop=dropout, residual=False, bias=False)
        self.GATclassifier = GATConv(in_feats=64 * 3, out_feats=32, num_heads=2, feat_drop=dropout, attn_drop=dropout, residual=False, bias=False)
        self.classifier = nn.Linear(32, out_feats, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormGAT1 = GraphNorm(hidden_dim * 4, eps=1e-5)
        self.GraphNormGAT2 = GraphNorm(64 * 3, eps=1e-5)

        ####  Dropout  ####
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, node_feats):
        updated_feats = self.GAT1(g, node_feats)
        updated_feats = updated_feats.flatten(1)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormGAT1(updated_feats)
        updated_feats = self.GAT2(g, updated_feats)
        updated_feats = updated_feats.flatten(1)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormGAT2(updated_feats)
        updated_feats = self.GATclassifier(g, updated_feats)
        updated_feats = updated_feats.mean(1)
        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            # Calculate graph representation by average readout.
            updated_feats = dgl.mean_nodes(g, 'updated_feats')
            updated_feats = self.dropout(updated_feats)
            return self.classifier(updated_feats)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class GraphMeshReader2GATLayer(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout):
        super(GraphMeshReader2GATLayer, self).__init__()

        ####  Layers  ####
        self.GAT1 = GATConv(in_feats=in_feats, out_feats=hidden_dim, num_heads=4, feat_drop=dropout, attn_drop=dropout, residual=False, bias=False)
        self.GAT2 = GATConv(in_feats=hidden_dim * 4, out_feats=32, num_heads=3, feat_drop=dropout, attn_drop=dropout, residual=False, bias=False)
        self.GATclassifier = GATConv(in_feats=32 * 3, out_feats=32, num_heads=2, feat_drop=dropout, attn_drop=dropout, residual=False, bias=False)
        self.classifier = nn.Linear(32, out_feats, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormGAT1 = GraphNorm(hidden_dim * 4, eps=1e-5)
        self.GraphNormGAT2 = GraphNorm(32 * 3, eps=1e-5)

        ####  Dropout  ####
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, node_feats):
        updated_feats = self.GAT1(g, node_feats)
        updated_feats = updated_feats.flatten(1)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormGAT1(updated_feats)
        updated_feats = self.GAT2(g, updated_feats)
        updated_feats = updated_feats.flatten(1)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormGAT2(updated_feats)
        updated_feats = self.GATclassifier(g, updated_feats)
        updated_feats = updated_feats.mean(1)
        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            # Calculate graph representation by average readout.
            updated_feats = dgl.mean_nodes(g, 'updated_feats')
            updated_feats = self.dropout(updated_feats)
            return self.classifier(updated_feats)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class GATMeshNetwork(nn.Module):
    def __init__(self, patch_feat_dim=5, internal_hidden_dim=256, readout_dim=32, hidden_dim=256, out_feats=15, dropout=0.5):
        super(GATMeshNetwork, self).__init__()

        ####  Layers  ####
        self.patch_reader = PatchReader2GATLayer(patch_feat_dim, hidden_dim=internal_hidden_dim, out_feats=readout_dim, dropout=dropout)
        self.mesh_reader = GraphMeshReader2GATLayer(readout_dim, hidden_dim=hidden_dim, out_feats=out_feats, dropout=dropout)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormPatchReader = GraphNorm(readout_dim, eps=1e-5)

        ####  Variables  ####
        self.readout_dim = readout_dim

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.readout_dim), device=device)
        dataloader = GraphDataLoader(patches, {"batch_size": 25, "drop_last": False})
        for patch in dataloader:
            readouts = torch.vstack((readouts, self.patch_reader(patch, patch.ndata["aggregated_feats"])))
        readouts = self.LeakyReLU(readouts)
        readouts = self.GraphNormPatchReader(readouts)
        return self.mesh_reader(mesh_graph, readouts), readouts

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
