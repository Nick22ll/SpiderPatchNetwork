import os

import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, SortPooling, JumpingKnowledge
from tqdm import tqdm

from Networks.NormalizationModules import UnitedNormCommon
from Networks.UniversalReadout import UniversalReadout


class CONVMGEmbedder(nn.Module):
    def __init__(self, feat_in_channels, readout_function="AR", jumping_mode=None, layers_num=3, dropout=0, *args, **kwargs):
        super(CONVMGEmbedder, self).__init__()

        self.JUMPING_MODE = jumping_mode

        ####  Layers  ####
        self.conLayers = nn.ModuleList()
        for _ in range(layers_num):
            self.conLayers.append(GraphConv(feat_in_channels, feat_in_channels, bias=False))

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)

        ####  Normalization Layers  ####
        self.normalizations = nn.ModuleList()
        self.normalizations.append(UnitedNormCommon(feat_in_channels))
        for i in range(1, layers_num):
            self.normalizations.append(UnitedNormCommon(feat_in_channels))

        ####  Readout Function  ####
        if readout_function == "UR":
            self.readout_list = nn.ModuleList()
            for i in range(layers_num):
                self.readout_list.append(UniversalReadout(feat_in_channels))
        elif readout_function == "AR":
            self.readout_list = []
            for i in range(layers_num):
                self.readout_list.append(dgl.mean_nodes)
        else:
            raise "Unknown Readout Function"

        ####   JumpKnowledge Modules  ####
        if self.JUMPING_MODE == "lstm" or self.JUMPING_MODE == "max":
            if readout_function == "UR":
                self.embed_dim = self.readout_list[0].readout_dim
            else:
                self.embed_dim = feat_in_channels
            self.jumping = JumpingKnowledge(mode=self.JUMPING_MODE, in_feats=self.embed_dim, num_layers=layers_num)
        elif self.JUMPING_MODE == "cat":
            self.jumping = JumpingKnowledge(mode=self.JUMPING_MODE)
            if readout_function == "UR":
                self.embed_dim = sum(r.readout_dim for r in self.readout_list)
            else:
                self.embed_dim = feat_in_channels + sum([feat_in_channels for i in range(1, layers_num)])
        elif self.JUMPING_MODE is None:
            if readout_function == "UR":
                self.embed_dim = self.readout_list[0].readout_dim
            else:
                self.embed_dim = feat_in_channels

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, mesh_graph, node_feats):
        # node_feats = self.chn_fixer(node_feats)
        updated_feats = node_feats
        embeddings = []
        for idx, conv_layer in enumerate(self.conLayers):
            updated_feats = conv_layer(mesh_graph, updated_feats)
            updated_feats = self.normalizations[idx](mesh_graph, updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            updated_feats = self.dropout(updated_feats)
            mesh_graph.ndata['updated_feats'] = updated_feats
            embeddings.append(self.readout_list[idx](mesh_graph, 'updated_feats'))

        if self.JUMPING_MODE is None:
            return self.LeakyReLU(embeddings[-1])

        return self.LeakyReLU(self.jumping(embeddings))

    def extractBestNormLayers(self):
        bests = []
        for idx, norm_layer in enumerate(self.normalizations):
            norm_powers = []
            for lambda_type in norm_layer.lambdas:
                norm_powers.append(torch.norm(lambda_type).detach().cpu().numpy())
            bests.append(norm_layer.norm_names[np.argmax(norm_powers)])
            tqdm.write(f"Best Norm for MESH embedder normalizer {idx} is {bests[-1]}")

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
