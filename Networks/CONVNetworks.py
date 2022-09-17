import os

import dgl
import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm


class PatchReader1ConvLayer(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout):
        super(PatchReader1ConvLayer, self).__init__()

        ####  Layers  ####
        self.conv = GraphConv(in_feats, hidden_dim, bias=False)
        self.linear = nn.Linear(hidden_dim, int(hidden_dim / 2), bias=False)
        self.classifier = nn.Linear(int(hidden_dim / 2), out_feats, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv = GraphNorm(hidden_dim, eps=1e-5)
        self.LinNorm = nn.InstanceNorm1d(int(hidden_dim / 2), eps=1e-05, momentum=0.1)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, edge_weight):
        updated_feats = self.conv(g, g.ndata["aggregated_feats"], edge_weight=edge_weight)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormConv(updated_feats)
        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            # Calculate graph representation by average readout.
            updated_feats = dgl.mean_nodes(g, 'updated_feats', 'weight')
            # updated_feats = torch.hstack((updated_feats, dgl.max_nodes(g, 'updated_feats', 'weight')))
            # updated_feats = torch.hstack((updated_feats, dgl.sum_nodes(g, 'updated_feats', 'weight')))
            updated_feats = self.linear(updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            updated_feats = self.LinNorm(updated_feats)
            updated_feats = self.dropout(updated_feats)
            return self.classifier(updated_feats)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class PatchReader2ConvLayerRetrieve(nn.Module):
    def __init__(self, in_feats, conv_dim, out_conv, in_feats_mlp, hidden_dim_mlp, out_feats, dropout):
        super(PatchReader2ConvLayerRetrieve, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_feats, conv_dim, bias=False)
        self.conv2 = GraphConv(conv_dim, out_conv, bias=False)
        self.in_linear = nn.Linear(in_feats_mlp, hidden_dim_mlp, bias=False)
        self.hidden_linear = nn.Linear(hidden_dim_mlp, int(hidden_dim_mlp / 4), bias=False)
        self.classifier = nn.Linear(int(hidden_dim_mlp / 4), out_feats, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(conv_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(out_conv, eps=1e-5)
        self.LinNorm = nn.InstanceNorm1d(out_conv * 3, eps=1e-05, momentum=0.1)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.xavier_uniform_(self.in_linear.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.hidden_linear.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, edge_weight):
        updated_feats = self.conv1(g, g.ndata["aggregated_feats"], edge_weight=edge_weight)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormConv1(updated_feats)

        updated_feats = self.conv2(g, updated_feats, edge_weight=edge_weight)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormConv2(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            # Calculate graph representation by average READOUT.
            # updated_feats = dgl.mean_nodes(g, 'updated_feats', 'weight')
            # updated_feats = torch.hstack((updated_feats, dgl.max_nodes(g, 'updated_feats', 'weight')))
            # updated_feats = torch.hstack((updated_feats, dgl.sum_nodes(g, 'updated_feats', 'weight')))

            # Select the node features of seed_point as updated feats
            num_nodes = g.batch_num_nodes()
            features_number = len(g.ndata["updated_feats"][0])
            updated_feats = torch.empty((0, int(features_number * num_nodes[0])), device=g.ndata["updated_feats"][0].device)
            last_idx = 0
            for i in range(g.batch_size):
                unbatched_graph = g.ndata["updated_feats"][last_idx: last_idx + num_nodes[i]].reshape((1, -1))
                updated_feats = torch.vstack((updated_feats, unbatched_graph))
                last_idx += num_nodes[i]

            updated_feats = self.in_linear(updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            updated_feats = self.LinNorm(updated_feats)
            updated_feats = self.dropout(updated_feats)

            updated_feats = self.hidden_linear(updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            updated_feats = self.LinNorm(updated_feats)
            updated_feats = self.dropout(updated_feats)

            return self.classifier(updated_feats)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class PatchReader2ConvLayer(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout):
        super(PatchReader2ConvLayer, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_feats, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, int(hidden_dim / 2), bias=False)
        self.linear = nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4), bias=False)
        self.classifier = nn.Linear(int(hidden_dim / 4), out_feats, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(int(hidden_dim / 2), eps=1e-5)
        self.LinNorm = nn.InstanceNorm1d(int(hidden_dim / 4), eps=1e-05, momentum=0.1)
        self.EdgeNorm = EdgeWeightNorm(norm='both')

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, node_feats, edge_weights):
        # if edge_weights is not None:
        # edge_weights = self.EdgeNorm(g, edge_weights)
        updated_feats = self.conv1(g, node_feats, edge_weight=edge_weights)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.conv2(g, updated_feats, edge_weight=edge_weights)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormConv2(updated_feats)
        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            updated_feats = dgl.mean_nodes(g, 'updated_feats')
            updated_feats = self.linear(updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            updated_feats = self.LinNorm(updated_feats)
            updated_feats = self.dropout(updated_feats)
            return self.classifier(updated_feats)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class ExpandedPatchReader(nn.Module):
    def __init__(self, in_feats, expand, hidden_dim, out_feats, dropout):
        super(ExpandedPatchReader, self).__init__()

        ####  Layers  ####
        self.expand1 = nn.Linear(in_feats, expand, bias=False)
        self.expand2 = nn.Linear(expand, expand * 2, bias=False)
        self.conv1 = GraphConv(expand * 2, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, int(hidden_dim / 2), bias=False)
        self.linear = nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4), bias=False)
        self.classifier = nn.Linear(int(hidden_dim / 4), out_feats, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(int(hidden_dim / 2), eps=1e-5)
        self.LinNorm = nn.InstanceNorm1d(int(hidden_dim / 4), eps=1e-05, momentum=0.1)
        self.EdgeNorm = EdgeWeightNorm(norm='both')

        ####  Weights Initialization  ####
        torch.nn.init.xavier_uniform_(self.expand1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.expand1.weight, gain=1.0)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, node_feats, edge_weights):
        # if edge_weights is not None:
        # edge_weights = self.EdgeNorm(g, edge_weights)
        updated_feats = self.expand1(node_feats)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.LinNorm(updated_feats)
        updated_feats = self.dropout(updated_feats)
        updated_feats = self.expand2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.LinNorm(updated_feats)
        updated_feats = self.dropout(updated_feats)
        updated_feats = self.conv1(g, updated_feats, edge_weight=edge_weights)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.conv2(g, updated_feats, edge_weight=edge_weights)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormConv2(updated_feats)
        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            updated_feats = dgl.mean_nodes(g, 'updated_feats')
            updated_feats = self.linear(updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            updated_feats = self.LinNorm(updated_feats)
            updated_feats = self.dropout(updated_feats)
            return self.classifier(updated_feats)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class PatchEmbedder2ConvLayer(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout):
        super(PatchEmbedder2ConvLayer, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_feats, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, out_feats, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(out_feats, eps=1e-5)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, g, node_feats, edge_weight):
        updated_feats = self.conv1(g, node_feats, edge_weight=edge_weight)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.conv2(g, updated_feats, edge_weight=edge_weight)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormConv2(updated_feats)
        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            return dgl.mean_nodes(g, 'updated_feats')

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class PatchReaderComplex(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_feats, dropout):
        super(PatchReaderComplex, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_feats, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, int(hidden_dim / 2), bias=False)
        self.linear1 = nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4), bias=False)
        self.linear2 = nn.Linear(int(hidden_dim / 4), int(hidden_dim / 8), bias=False)
        self.linear3 = nn.Linear(int(hidden_dim / 8), int(hidden_dim / 16), bias=False)
        self.classifier = nn.Linear(int(hidden_dim / 16), out_feats, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(int(hidden_dim / 2), eps=1e-5)
        self.LinNorm1 = nn.InstanceNorm1d(int(hidden_dim / 4), eps=1e-05, momentum=0.1)
        self.LinNorm2 = nn.InstanceNorm1d(int(hidden_dim / 8), eps=1e-05, momentum=0.1)
        self.LinNorm3 = nn.InstanceNorm1d(int(hidden_dim / 16), eps=1e-05, momentum=0.1)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.linear2.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.linear3.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)
        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, node_feats, edge_weight):
        updated_feats = self.conv1(g, node_feats, edge_weight=edge_weight)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.conv2(g, updated_feats, edge_weight=edge_weight)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormConv2(updated_feats)
        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            updated_feats = dgl.mean_nodes(g, 'updated_feats')

            # First Linear Layer
            updated_feats = self.linear1(updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            updated_feats = self.LinNorm1(updated_feats)
            updated_feats = self.dropout(updated_feats)

            # Second Linear Layer
            updated_feats = self.linear2(updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            updated_feats = self.LinNorm2(updated_feats)
            updated_feats = self.dropout(updated_feats)

            # Third Linear Layer
            updated_feats = self.linear3(updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            updated_feats = self.LinNorm3(updated_feats)
            updated_feats = self.dropout(updated_feats)
            return self.classifier(updated_feats)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class GraphMeshReader1ConvLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(GraphMeshReader1ConvLayer, self).__init__()

        ####  Layers  ####
        self.conv = GraphConv(in_dim, hidden_dim, bias=False)
        self.linear = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.classifier = nn.Linear(int(hidden_dim / 2), out_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv = GraphNorm(hidden_dim, eps=1e-5)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, mesh_graph, features, edge_weights):
        updated_feats = self.conv(mesh_graph, features, edge_weight=edge_weights)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormConv(updated_feats)
        with mesh_graph.local_scope():
            mesh_graph.ndata['updated_feats'] = updated_feats
            # Calculate graph representation by average readout.
            updated_feats = dgl.mean_nodes(mesh_graph, 'updated_feats')
            updated_feats = self.linear(updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            updated_feats = self.dropout(updated_feats)
            return self.classifier(updated_feats)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class GraphMeshReader2ConvLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(GraphMeshReader2ConvLayer, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_dim, hidden_dim, weight=True, bias=True)
        self.conv2 = GraphConv(hidden_dim, int(hidden_dim / 2), weight=True, bias=True)
        self.linear = nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4), bias=True)
        self.classifier = nn.Linear(int(hidden_dim / 4), out_dim, bias=True)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(int(hidden_dim / 2), eps=1e-5)
        self.LinNorm = nn.InstanceNorm1d(1, eps=1e-05, momentum=0.1)
        self.EdgeNorm = EdgeWeightNorm(norm='both')

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, mesh_graph, features, edge_weights):
        # if edge_weights is not None:
        #   edge_weights = self.EdgeNorm(mesh_graph, edge_weights)
        updated_feats = self.conv1(mesh_graph, features, edge_weight=edge_weights)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.conv2(mesh_graph, updated_feats, edge_weight=edge_weights)
        updated_feats = self.LeakyReLU(updated_feats)
        updated_feats = self.GraphNormConv2(updated_feats)
        with mesh_graph.local_scope():
            mesh_graph.ndata['updated_feats'] = updated_feats
            # Calculate graph representation by average readout.
            updated_feats = dgl.mean_nodes(mesh_graph, 'updated_feats')
            updated_feats = self.linear(updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            updated_feats = self.LinNorm(updated_feats)
            updated_feats = self.dropout(updated_feats)
            return self.classifier(updated_feats)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class MeshNetwork(nn.Module):
    def __init__(self, patch_feat_dim, internal_hidden_dim, readout_dim, hidden_dim, out_feats, dropout, patch_batch=25, mesh_graph_edge_weights=True, patch_edge_weights=True):
        super(MeshNetwork, self).__init__()

        ####  Layers  ####
        self.patch_reader = PatchReader2ConvLayer(patch_feat_dim, hidden_dim=internal_hidden_dim, out_feats=readout_dim, dropout=dropout)
        # self.patch_reader = PatchReaderComplex(patch_feat_dim, hidden_dim=internal_hidden_dim, out_feats=readout_dim, dropout=dropout)
        # self.patch_reader = PatchEmbedder2ConvLayer(patch_feat_dim, hidden_dim=internal_hidden_dim, out_feats=readout_dim, dropout=dropout)
        self.mesh_reader = GraphMeshReader2ConvLayer(in_dim=readout_dim, hidden_dim=hidden_dim, out_dim=out_feats, dropout=dropout)
        # self.mesh_reader = GraphMeshReader1ConvLayer(in_dim=readout_dim, hidden_dim=hidden_dim, out_dim=out_feats, dropout=dropout)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormPatchReader = GraphNorm(readout_dim, eps=1e-5)

        ####  Variables  ####
        self.readout_dim = readout_dim
        self.patch_batch = patch_batch
        self.patch_edge_weights = patch_edge_weights
        self.mesh_graph_edge_weights = mesh_graph_edge_weights

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.readout_dim), device=device)
        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(patches, batch_size=self.patch_batch, drop_last=False)
        for patch in dataloader:
            if self.patch_edge_weights:
                readouts = torch.vstack((readouts, self.patch_reader(patch, patch.ndata["aggregated_feats"], patch.edata["weights"])))
            else:
                readouts = torch.vstack((readouts, self.patch_reader(patch, patch.ndata["aggregated_feats"], None)))
        readouts = self.LeakyReLU(readouts)
        readouts = self.GraphNormPatchReader(readouts)
        if self.mesh_graph_edge_weights:
            return self.mesh_reader(mesh_graph, readouts, mesh_graph.edata["weights"]), readouts
        else:
            return self.mesh_reader(mesh_graph, readouts, None), readouts

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
