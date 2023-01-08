import os

import dgl
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm
from dgl.nn.pytorch import GraphConv, SortPooling

from Networks.UniversalReadout import UniversalReadout


class JumpGMEmbedder(nn.Module):
    def __init__(self, in_channels, layers_num, dropout):
        super(JumpGMEmbedder, self).__init__()

        ####  Layers  ####
        # out_channels = in_channels * 4
        # self.chn_fixer = nn.Linear(in_channels, out_channels, bias=False)
        out_channels = in_channels
        self.conLayers = nn.ModuleList()
        for _ in range(layers_num):
            self.conLayers.append(GraphConv(out_channels, out_channels, bias=False))

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        #### Readout Layer ####
        self.readouts = nn.ModuleList()
        for _ in range(layers_num):
            self.readouts.append(UniversalReadout(out_channels, dropout))

        ####  Normalization Layers  ####
        self.normalizations = nn.ModuleList()
        for _ in range(layers_num):
            self.normalizations.append(GraphNorm(out_channels, eps=1e-5))

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = self.readouts[0].readout_dim * layers_num

    def forward(self, g, node_feats):
        # node_feats = self.chn_fixer(node_feats)
        updated_feats = node_feats
        readout = torch.empty(0, device=node_feats.device)
        for idx, conv_layer in enumerate(self.conLayers):
            updated_feats = conv_layer(g, updated_feats)
            updated_feats = self.normalizations[idx](updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            with g.local_scope():
                g.ndata['updated_feats'] = updated_feats
                readout = torch.hstack((readout, self.readouts[idx](g, 'updated_feats')))

        return self.LeakyReLU(readout)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class JumpResGMEmbedder(nn.Module):
    def __init__(self, in_channels, layers_num, dropout):
        super(JumpResGMEmbedder, self).__init__()

        ####  Layers  ####
        # out_channels = in_channels * 4
        # self.chn_fixer = nn.Linear(in_channels, out_channels, bias=False)
        out_channels = in_channels
        self.conLayers = nn.ModuleList()
        for _ in range(layers_num):
            self.conLayers.append(GraphConv(out_channels, out_channels, bias=False))

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        #### Readout Layer ####
        self.readouts = nn.ModuleList()
        for _ in range(layers_num):
            self.readouts.append(UniversalReadout(out_channels, dropout))

        ####  Normalization Layers  ####
        self.normalizations = nn.ModuleList()
        for _ in range(layers_num):
            self.normalizations.append(GraphNorm(out_channels, eps=1e-5))

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = self.readouts[0].readout_dim * layers_num

    def forward(self, g, node_feats):
        # node_feats = self.chn_fixer(node_feats)
        updated_feats = self.conLayers[0](g, node_feats)
        updated_feats = self.normalizations[0](updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = self.readouts[0](g, 'updated_feats')

        prev0 = torch.clone(updated_feats)
        updated_feats = updated_feats + node_feats
        for idx, conv_layer in enumerate(self.conLayers[1:], start=1):
            updated_feats = conv_layer(g, updated_feats)
            updated_feats = self.normalizations[idx](updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)

            prev1 = torch.clone(updated_feats)
            updated_feats = updated_feats + prev0
            prev0 = torch.clone(prev1)

            with g.local_scope():
                g.ndata['updated_feats'] = updated_feats
                readout = torch.hstack((readout, self.readouts[idx](g, 'updated_feats')))

        return self.LeakyReLU(readout)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class GMEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super(GMEmbedder, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_dim, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(hidden_dim, eps=1e-5)

        #### Readout Layer ####
        self.readout1 = UniversalReadout(hidden_dim, dropout=0)
        self.readout2 = UniversalReadout(hidden_dim, dropout=0)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = self.readout1.readout_dim + self.readout2.readout_dim + (hidden_dim * 2)

    def forward(self, mesh_graph, features, edge_weights):
        updated_feats = self.conv1(mesh_graph, features, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        updated_feats = self.dropout(updated_feats)

        with mesh_graph.local_scope():
            mesh_graph.ndata["readout"] = updated_feats
            readouts = self.readout1(mesh_graph, "readout")
            readouts = torch.hstack((readouts, dgl.mean_nodes(mesh_graph, "readout")))

        updated_feats = self.conv2(mesh_graph, updated_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        updated_feats = self.dropout(updated_feats)

        with mesh_graph.local_scope():
            mesh_graph.ndata["readout"] = updated_feats
            readouts = torch.hstack((readouts, self.readout2(mesh_graph, "readout")))
            readouts = torch.hstack((readouts, dgl.mean_nodes(mesh_graph, "readout")))

        return self.LeakyReLU(readouts)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class GMEmbedder2ConvAR(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super(GMEmbedder2ConvAR, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_dim, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(hidden_dim, eps=1e-5)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = hidden_dim * 2

    def forward(self, mesh_graph, features, edge_weights):
        updated_feats = self.conv1(mesh_graph, features, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with mesh_graph.local_scope():
            mesh_graph.ndata["readout"] = updated_feats
            readouts = dgl.mean_nodes(mesh_graph, "readout")

        updated_feats = self.conv2(mesh_graph, updated_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with mesh_graph.local_scope():
            mesh_graph.ndata["readout"] = updated_feats
            readouts = torch.hstack((readouts, dgl.mean_nodes(mesh_graph, "readout")))

        return self.LeakyReLU(readouts)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class GMEmbedder2ConvUR(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super(GMEmbedder2ConvUR, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_dim, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(hidden_dim, eps=1e-5)

        #### Readout Layer ####
        self.readout1 = UniversalReadout(hidden_dim, dropout=0)
        self.readout2 = UniversalReadout(hidden_dim, dropout=0)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = self.readout1.readout_dim + self.readout2.readout_dim

    def forward(self, mesh_graph, features, edge_weights):
        updated_feats = self.conv1(mesh_graph, features, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with mesh_graph.local_scope():
            mesh_graph.ndata["readout"] = updated_feats
            readouts = self.readout1(mesh_graph, "readout")

        updated_feats = self.conv2(mesh_graph, updated_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with mesh_graph.local_scope():
            mesh_graph.ndata["readout"] = updated_feats
            readouts = torch.hstack((readouts, self.readout2(mesh_graph, "readout")))

        return self.LeakyReLU(readouts)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class GMEmbedder2ConvSortPoolReadout(nn.Module):
    def __init__(self, in_dim, hidden_dim, pool_dim, dropout):
        super(GMEmbedder2ConvSortPoolReadout, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_dim, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, int(hidden_dim / 4), bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(int(hidden_dim / 4), eps=1e-5)

        #### Readout Layer ####
        self.readout = SortPooling(k=pool_dim)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = pool_dim * int(hidden_dim / 4) + (pool_dim * hidden_dim)

    def forward(self, mesh_graph, features, edge_weights):
        updated_feats = self.conv1(mesh_graph, features, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        readouts = self.readout(mesh_graph, updated_feats)

        updated_feats = self.conv2(mesh_graph, updated_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        readouts = torch.hstack((readouts, self.readout(mesh_graph, updated_feats)))

        return self.LeakyReLU(readouts)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
