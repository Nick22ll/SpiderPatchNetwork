import os

import dgl
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm, SortPooling

from Networks.MLP import MLP, GenericMLP
from Networks.SpiralReadout import SpiralReadout
from Networks.UniversalReadout import UniversalReadout


class PatchEmbedder2ConvLayer(nn.Module):
    def __init__(self, in_feats, hidden_dim, embedding_dim, dropout):
        super(PatchEmbedder2ConvLayer, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_feats, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, int(hidden_dim / 4), bias=False)
        self.embedder = nn.Linear(int(hidden_dim / 4) + hidden_dim + int(hidden_dim / 16) + int(hidden_dim / 4), embedding_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(int(hidden_dim / 4), eps=1e-5)
        self.LinNorm = nn.InstanceNorm1d(embedding_dim, eps=1e-05, momentum=0.1)

        #### Readout Layer ####
        self.readout1 = UniversalReadout(hidden_dim, hidden_dim * 2, int(hidden_dim / 4), dropout)
        self.readout2 = UniversalReadout(int(hidden_dim / 4), int(hidden_dim / 2), int(hidden_dim / 16), dropout)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.xavier_uniform_(self.embedder.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, node_feats, edge_weights):
        # TODO  Valutare inserimento di un altro readout sul grafo di base
        # with g.local_scope():
        #     g.ndata['updated_feats'] = node_feats
        #     readout = self.readout1(g, 'updated_feats')
        #     readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats")))
        #
        updated_feats = self.conv1(g, node_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = self.readout1(g, 'updated_feats')
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats")))

        updated_feats = self.conv2(g, updated_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, self.readout2(g, 'updated_feats')))
            updated_feats = self.embedder(torch.hstack((readout, dgl.mean_nodes(g, "updated_feats"))))
            updated_feats = self.LinNorm(updated_feats)
            return self.LeakyReLU(updated_feats)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class PatchEmbedder2ConvLayerAR(nn.Module):
    def __init__(self, in_feats, hidden_dim, embedding_dim, dropout):
        super(PatchEmbedder2ConvLayerAR, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_feats, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, int(hidden_dim / 4), bias=False)
        self.embedder = nn.Linear(in_feats + hidden_dim + int(hidden_dim / 4), embedding_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(int(hidden_dim / 4), eps=1e-5)
        self.LinNorm = nn.InstanceNorm1d(embedding_dim, eps=1e-05, momentum=0.1)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.xavier_uniform_(self.embedder.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, node_feats, edge_weights):
        with g.local_scope():
            g.ndata['updated_feats'] = node_feats
            readout = dgl.mean_nodes(g, "updated_feats")

        updated_feats = self.conv1(g, node_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats")))

        updated_feats = self.conv2(g, updated_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            updated_feats = self.embedder(torch.hstack((readout, dgl.mean_nodes(g, "updated_feats"))))
            updated_feats = self.LinNorm(updated_feats)
            return self.LeakyReLU(updated_feats)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class GMReader2ConvUniversalReadout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(GMReader2ConvUniversalReadout, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_dim, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, bias=False)
        # self.classifier = GenericMLP(int(hidden_dim / 4) * 2, hidden_dim, out_dim, dropout)
        self.classifier = nn.Linear(int(hidden_dim / 4) * 2, out_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(hidden_dim, eps=1e-5)

        #### Readout Layer ####
        self.readout1 = UniversalReadout(hidden_dim, hidden_dim * 2, int(hidden_dim / 4), dropout)
        self.readout2 = UniversalReadout(hidden_dim, hidden_dim * 2, int(hidden_dim / 4), dropout)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

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

        return self.classifier(readouts)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class GMReader2ConvAverageReadout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(GMReader2ConvAverageReadout, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_dim, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, bias=False)
        # self.classifier = GenericMLP(2 * hidden_dim, hidden_dim, out_dim, dropout)
        self.classifier = nn.Linear(hidden_dim * 2, out_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(hidden_dim, eps=1e-5)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

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

        return self.classifier(readouts)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class GMEmbedder2ConvAverageReadout(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super(GMEmbedder2ConvAverageReadout, self).__init__()

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


class GMEmbedder2ConvUniversalReadout(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super(GMEmbedder2ConvUniversalReadout, self).__init__()

        self.embed_dim = int(hidden_dim / 4) * 2

        ####  Layers  ####
        self.conv1 = GraphConv(in_dim, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(hidden_dim, eps=1e-5)

        #### Readout Layer ####
        self.readout1 = UniversalReadout(hidden_dim, hidden_dim * 2, int(hidden_dim / 4), dropout)
        self.readout2 = UniversalReadout(hidden_dim, hidden_dim * 2, int(hidden_dim / 4), dropout)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

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


class GMReader2ConvSortPoolReadout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, pool_dim, dropout):
        super(GMReader2ConvSortPoolReadout, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_dim, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, int(hidden_dim / 4), bias=False)
        # self.classifier = GenericMLP(pool_dim * int(hidden_dim / 4) + (pool_dim * hidden_dim), pool_dim * int(hidden_dim / 4), out_dim, dropout)
        self.classifier = nn.Linear(pool_dim * int(hidden_dim / 4) + (pool_dim * hidden_dim), out_dim, bias=False)

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
        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, mesh_graph, features, edge_weights):
        updated_feats = self.conv1(mesh_graph, features, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        readouts = self.readout(mesh_graph, updated_feats)

        updated_feats = self.conv2(mesh_graph, updated_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        readouts = torch.hstack((readouts, self.readout(mesh_graph, updated_feats)))

        return self.classifier(readouts)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class MeshNetworkARAR(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_feats, dropout, mesh_graph_edge_weights=True):
        super(MeshNetworkARAR, self).__init__()

        ####  Variables  ####
        self.name = "CONVMeshNetworkARAR"
        self.mesh_graph_edge_weights = mesh_graph_edge_weights

        ####  Layers  ####
        self.mesh_reader = GMReader2ConvAverageReadout(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_feats, dropout=dropout)

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, patches[0].ndata["aggregated_feats"].shape[1]), device=device)
        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(patches, batch_size=10, drop_last=False)
        for spider_patch in dataloader:
            readouts = torch.vstack((readouts, dgl.mean_nodes(spider_patch, "aggregated_feats")))

        if self.mesh_graph_edge_weights:
            return self.mesh_reader(mesh_graph, readouts, mesh_graph.edata["weights"])
        else:
            return self.mesh_reader(mesh_graph, readouts, None)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class MeshNetworkPEARAR(nn.Module):
    def __init__(self, in_dim, readout_dim, hidden_dim, out_feats, dropout, mesh_graph_edge_weights=True):
        super(MeshNetworkPEARAR, self).__init__()

        ####  Variables  ####
        self.name = "CONVMeshNetworkPEARAR"
        self.readout_dim = readout_dim
        self.mesh_graph_edge_weights = mesh_graph_edge_weights

        ####  Layers  ####
        self.patch_embedder = PatchEmbedder2ConvLayerAR(in_dim, in_dim * 4, readout_dim, dropout)
        self.mesh_reader = GMReader2ConvAverageReadout(in_dim=readout_dim, hidden_dim=hidden_dim, out_dim=out_feats, dropout=dropout)

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.readout_dim), device=device)
        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(np.random.permutation(patches), batch_size=10, drop_last=False)
        for spider_patch in dataloader:
            readouts = torch.vstack((readouts, self.patch_embedder(spider_patch, spider_patch.ndata["aggregated_feats"], spider_patch.edata["weights"])))

        if self.mesh_graph_edge_weights:
            return self.mesh_reader(mesh_graph, readouts, mesh_graph.edata["weights"])
        else:
            return self.mesh_reader(mesh_graph, readouts, None)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class MeshNetworkPEARUR(nn.Module):
    def __init__(self, in_dim, readout_dim, hidden_dim, out_feats, dropout, mesh_graph_edge_weights=True):
        super(MeshNetworkPEARUR, self).__init__()

        ####  Variables  ####
        self.name = "CONVMeshNetworkPEARUR"
        self.readout_dim = readout_dim
        self.mesh_graph_edge_weights = mesh_graph_edge_weights

        ####  Layers  ####
        self.patch_embedder = PatchEmbedder2ConvLayer(in_dim, in_dim * 4, readout_dim, dropout)
        self.mesh_reader = GMReader2ConvUniversalReadout(in_dim=readout_dim, hidden_dim=hidden_dim, out_dim=out_feats, dropout=dropout)

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.readout_dim), device=device)
        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(patches, batch_size=10, drop_last=False)
        for spider_patch in dataloader:
            readouts = torch.vstack((readouts, self.patch_embedder(spider_patch, spider_patch.ndata["aggregated_feats"], spider_patch.edata["weights"])))

        if self.mesh_graph_edge_weights:
            return self.mesh_reader(mesh_graph, readouts, mesh_graph.edata["weights"])
        else:
            return self.mesh_reader(mesh_graph, readouts, None)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class MeshNetworkPEUR(nn.Module):
    def __init__(self, in_dim, readout_dim, hidden_dim, out_feats, dropout, mesh_graph_edge_weights=True):
        super(MeshNetworkPEUR, self).__init__()

        ####  Variables  ####
        self.name = "CONVMeshNetworkPEUR"
        self.readout_dim = readout_dim
        self.mesh_graph_edge_weights = mesh_graph_edge_weights

        ####  Layers  ####
        self.patch_embedder = PatchEmbedder2ConvLayer(in_dim, in_dim * 16, readout_dim, dropout)
        self.mesh_reader = GMReader2ConvUniversalReadout(in_dim=readout_dim, hidden_dim=hidden_dim, out_dim=out_feats, dropout=dropout)

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.readout_dim), device=device)
        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(np.random.permutation(patches), batch_size=25, drop_last=False)
        for spider_patch in dataloader:
            readouts = torch.vstack((readouts, self.patch_embedder(spider_patch, spider_patch.ndata["aggregated_feats"], spider_patch.edata["weights"])))

        if self.mesh_graph_edge_weights:
            return self.mesh_reader(mesh_graph, readouts, mesh_graph.edata["weights"])
        else:
            return self.mesh_reader(mesh_graph, readouts, None)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class MeshNetworkURUR(nn.Module):
    def __init__(self, in_dim, readout_dim, hidden_dim, out_feats, dropout, mesh_graph_edge_weights=True):
        super(MeshNetworkURUR, self).__init__()

        ####  Variables  ####
        self.name = "CONVMeshNetworkURUR"
        self.readout_dim = readout_dim
        self.mesh_graph_edge_weights = mesh_graph_edge_weights

        ####  Layers  ####
        self.readout = UniversalReadout(in_dim, in_dim, readout_dim, dropout=0.05)
        self.mesh_reader = GMReader2ConvUniversalReadout(in_dim=readout_dim, hidden_dim=hidden_dim, out_dim=out_feats, dropout=dropout)

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.readout_dim), device=device)
        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(np.random.permutation(patches), batch_size=25, drop_last=False)
        for spider_patch in dataloader:
            readouts = torch.vstack((readouts, self.readout(spider_patch, "aggregated_feats")))

        if self.mesh_graph_edge_weights:
            return self.mesh_reader(mesh_graph, readouts, mesh_graph.edata["weights"])
        else:
            return self.mesh_reader(mesh_graph, readouts, None)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class MeshNetworkSRUR(nn.Module):
    def __init__(self, readout_dim, hidden_dim, out_feats, dropout, mesh_graph_edge_weights=True):
        super(MeshNetworkSRUR, self).__init__()

        ####  Variables  ####
        self.name = "CONVMeshNetworkSRUR"
        self.readout_dim = readout_dim
        self.mesh_graph_edge_weights = mesh_graph_edge_weights

        ####  Layers  ####

        self.readout = SpiralReadout(readout_dim)
        # self.readout_conv = nn.Conv1d(1, 1, 5, stride=5, padding=0, dilation=0, groups=1, bias=True, padding_mode='zeros')
        self.mesh_reader = GMReader2ConvUniversalReadout(in_dim=self.readout_dim, hidden_dim=hidden_dim, out_dim=out_feats, dropout=dropout)

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.readout_dim), device=device)
        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(patches, batch_size=int(mesh_graph.num_nodes() / 2), drop_last=False)
        for spider_patch in dataloader:
            readouts = torch.vstack((readouts, self.readout(spider_patch, "aggregated_feats")))

        # readouts = readouts.reshape((50, 1, 625))

        # readouts = self.readout_conv(readouts)

        if self.mesh_graph_edge_weights:
            return self.mesh_reader(mesh_graph, readouts, mesh_graph.edata["weights"])
        else:
            return self.mesh_reader(mesh_graph, readouts, None)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class MeshNetworkSRAR(nn.Module):
    def __init__(self, readout_dim, hidden_dim, out_feats, dropout, mesh_graph_edge_weights=True):
        super(MeshNetworkSRAR, self).__init__()

        ####  Variables  ####
        self.name = "CONVMeshNetworkSRAR"
        self.readout_dim = readout_dim
        self.mesh_graph_edge_weights = mesh_graph_edge_weights

        ####  Layers  ####
        self.readout = SpiralReadout(readout_dim)
        self.mesh_reader = GMReader2ConvAverageReadout(in_dim=self.readout_dim, hidden_dim=hidden_dim, out_dim=out_feats, dropout=dropout)

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.readout_dim), device=device)
        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(np.random.permutation(patches), batch_size=int(mesh_graph.num_nodes() / 2), drop_last=False)
        for spider_patch in dataloader:
            readouts = torch.vstack((readouts, self.readout(spider_patch, "aggregated_feats")))
        if self.mesh_graph_edge_weights:
            return self.mesh_reader(mesh_graph, readouts, mesh_graph.edata["weights"])
        else:
            return self.mesh_reader(mesh_graph, readouts, None)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class MeshNetworkSRPR(nn.Module):
    def __init__(self, readout_dim, hidden_dim, out_feats, dropout, mesh_graph_edge_weights=True):
        super(MeshNetworkSRPR, self).__init__()

        ####  Variables  ####
        self.name = "CONVMeshNetworkSRPR"
        self.readout_dim = readout_dim
        self.mesh_graph_edge_weights = mesh_graph_edge_weights

        ####  Layers  ####
        self.readout = SpiralReadout(readout_dim)
        self.mesh_reader = GMReader2ConvSortPoolReadout(in_dim=self.readout_dim, hidden_dim=hidden_dim, out_dim=out_feats, dropout=dropout, pool_dim=10)

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.readout_dim), device=device)
        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(patches, batch_size=int(mesh_graph.num_nodes() / 2), drop_last=False)
        for spider_patch in dataloader:
            readouts = torch.vstack((readouts, self.readout(spider_patch, "aggregated_feats")))
        if self.mesh_graph_edge_weights:
            return self.mesh_reader(mesh_graph, readouts, mesh_graph.edata["weights"])
        else:
            return self.mesh_reader(mesh_graph, readouts, None)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class AverageMeshNetworkPEARAR(nn.Module):
    def __init__(self, in_dim, readout_dim, hidden_dim, block_dim, out_feats, dropout, mesh_graph_edge_weights=True):
        super(AverageMeshNetworkPEARAR, self).__init__()

        ####  Variables  ####
        self.name = "AverageCONVMeshNetworkPEARAR"
        self.readout_dim = readout_dim
        self.block_readout_dim = hidden_dim * 2
        self.mesh_graph_edge_weights = mesh_graph_edge_weights

        ####  Layers  ####
        self.patch_embedder = PatchEmbedder2ConvLayerAR(in_dim, in_dim * 4, readout_dim, dropout)
        self.mesh_reader = GMEmbedder2ConvAverageReadout(in_dim=readout_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = nn.Linear(in_features=self.block_readout_dim * block_dim, out_features=out_feats, bias=False)

    def forward(self, mesh_graphs, patches_list, device):
        block_readout = torch.empty(0, device=device)
        mesh_graph_dataloader = GraphDataLoader(mesh_graphs, batch_size=1, drop_last=False)
        for sampler, mesh_graph in enumerate(mesh_graph_dataloader):
            readouts = torch.empty((0, self.readout_dim), device=device)
            # noinspection PyTypeChecker
            dataloader = GraphDataLoader(np.random.permutation(patches_list[sampler]), batch_size=10, drop_last=False)
            for spider_patch in dataloader:
                readouts = torch.vstack((readouts, self.patch_embedder(spider_patch, spider_patch.ndata["aggregated_feats"], spider_patch.edata["weights"])))

            if self.mesh_graph_edge_weights:
                block_readout = torch.hstack((block_readout, self.mesh_reader(mesh_graph, readouts, mesh_graph.edata["weights"])))
            else:
                block_readout = torch.hstack((block_readout, self.mesh_reader(mesh_graph, readouts, None)))
        return self.classifier(block_readout)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class AverageMeshNetworkPEARUR(nn.Module):
    def __init__(self, in_dim, readout_dim, hidden_dim, block_dim, out_feats, dropout, mesh_graph_edge_weights=True):
        super(AverageMeshNetworkPEARUR, self).__init__()

        ####  Variables  ####
        self.name = "AverageCONVMeshNetworkPEARUR"
        self.readout_dim = readout_dim
        self.mesh_graph_edge_weights = mesh_graph_edge_weights

        ####  Layers  ####
        self.patch_embedder = PatchEmbedder2ConvLayerAR(in_dim, in_dim * 2, readout_dim, dropout)
        self.mesh_reader = GMEmbedder2ConvUniversalReadout(in_dim=readout_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = nn.Linear(in_features=self.mesh_reader.embed_dim * block_dim, out_features=out_feats, bias=False)

    def forward(self, mesh_graphs, patches_list, device):
        block_readout = torch.empty(0, device=device)
        mesh_graph_dataloader = GraphDataLoader(mesh_graphs, batch_size=1, drop_last=False)
        for sampler, mesh_graph in enumerate(mesh_graph_dataloader):
            readouts = torch.empty((0, self.readout_dim), device=device)
            # noinspection PyTypeChecker
            dataloader = GraphDataLoader(np.random.permutation(patches_list[sampler]), batch_size=10, drop_last=False)
            for spider_patch in dataloader:
                readouts = torch.vstack((readouts, self.patch_embedder(spider_patch, spider_patch.ndata["aggregated_feats"], spider_patch.edata["weights"])))

            if self.mesh_graph_edge_weights:
                block_readout = torch.hstack((block_readout, self.mesh_reader(mesh_graph, readouts, mesh_graph.edata["weights"])))
            else:
                block_readout = torch.hstack((block_readout, self.mesh_reader(mesh_graph, readouts, None)))
        return self.classifier(block_readout)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
