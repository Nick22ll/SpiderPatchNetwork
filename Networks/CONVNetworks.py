import os

import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader

from Networks.CONVMeshGraphModules import GMEmbedder, GMEmbedder2ConvUR, JumpResGMEmbedder, JumpGMEmbedder
from Networks.CONVSpiderPatchModules import SPEmbedder3Conv, WEIGHTSPEmbedder3Conv, READOUTWEIGHTSPEmbedder3Conv, SPEmbedder3ConvUNIVERSAL, JumpResSPEmbedder, WEIGHTSPEmbedderAR
from Networks.SpiralReadout import SpiralReadout
from Networks.UniversalReadout import UniversalReadout


class UNIVERSALCONVMeshNetwork(nn.Module):
    def __init__(self, in_dim, node_numbers, out_feats, dropout):
        super(UNIVERSALCONVMeshNetwork, self).__init__()

        ####  Variables  ####
        self.name = self.__class__.__name__
        self.rng = np.random.default_rng(22)

        ####  Layers  ####
        self.patch_embedder = WEIGHTSPEmbedderAR(in_channels=in_dim, layers_num=3, dropout=dropout, nodes_number=node_numbers)
        self.mesh_embedder = JumpGMEmbedder(in_channels=self.patch_embedder.embed_dim, layers_num=3, dropout=dropout)
        self.classifier = nn.Linear(in_features=self.mesh_embedder.embed_dim, out_features=out_feats, bias=False)

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.patch_embedder.embed_dim), device=device)
        random_sequence = self.rng.permutation(len(patches))

        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(patches[random_sequence], batch_size=25, drop_last=False)
        for spider_patch in dataloader:
            readouts = torch.vstack((readouts, self.patch_embedder(spider_patch, spider_patch.ndata["aggregated_feats"])))

        with mesh_graph.local_scope():
            dgl.reorder_graph(mesh_graph, node_permute_algo='custom', permute_config={'nodes_perm': random_sequence})
            mesh_embedding = self.mesh_embedder(mesh_graph, readouts)

            return self.classifier(mesh_embedding)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class READOUTWEIGHTCONVMeshNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_feats, dropout, nodes_number, edges_number):
        super(READOUTWEIGHTCONVMeshNetwork, self).__init__()

        ####  Variables  ####
        self.name = self.__class__.__name__
        self.rng = np.random.default_rng(22)

        ####  Layers  ####
        self.patch_embedder = READOUTWEIGHTSPEmbedder3Conv(in_feats=in_dim, hidden_dim=in_dim * 4, dropout=dropout, nodes_number=nodes_number, edges_number=edges_number)
        self.mesh_embedder = GMEmbedder(in_dim=self.patch_embedder.embed_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = nn.Linear(in_features=self.mesh_embedder.embed_dim, out_features=out_feats, bias=False)

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.patch_embedder.embed_dim), device=device)
        random_sequence = self.rng.permutation(len(patches))

        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(patches[random_sequence], batch_size=10, drop_last=False)
        for spider_patch in dataloader:
            readouts = torch.vstack((readouts, self.patch_embedder(spider_patch, spider_patch.ndata["aggregated_feats"])))

        with mesh_graph.local_scope():
            dgl.reorder_graph(mesh_graph, node_permute_algo='custom', permute_config={'nodes_perm': random_sequence})
            mesh_embedding = self.mesh_embedder(mesh_graph, readouts, None)

            return self.classifier(mesh_embedding)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class WEIGHTCONVMeshNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_feats, dropout, nodes_number, edges_number):
        super(WEIGHTCONVMeshNetwork, self).__init__()

        ####  Variables  ####
        self.name = self.__class__.__name__
        self.rng = np.random.default_rng(22)

        ####  Layers  ####
        self.patch_embedder = WEIGHTSPEmbedder3Conv(in_feats=in_dim, hidden_dim=in_dim * 4, dropout=dropout, nodes_number=nodes_number, edges_number=edges_number)
        self.mesh_embedder = GMEmbedder(in_dim=self.patch_embedder.embed_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = nn.Linear(in_features=self.mesh_embedder.embed_dim, out_features=out_feats, bias=False)

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.patch_embedder.embed_dim), device=device)
        random_sequence = self.rng.permutation(len(patches))

        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(patches[random_sequence], batch_size=10, drop_last=False)
        for spider_patch in dataloader:
            readouts = torch.vstack((readouts, self.patch_embedder(spider_patch, spider_patch.ndata["aggregated_feats"], spider_patch.edata["weights"])))

        with mesh_graph.local_scope():
            dgl.reorder_graph(mesh_graph, node_permute_algo='custom', permute_config={'nodes_perm': random_sequence})
            mesh_embedding = self.mesh_embedder(mesh_graph, readouts, None)

            return self.classifier(mesh_embedding)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class CONVMeshNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_feats, dropout, mesh_graph_edge_weights=True):
        super(CONVMeshNetwork, self).__init__()

        ####  Variables  ####
        self.name = self.__class__.__name__
        self.mesh_graph_edge_weights = mesh_graph_edge_weights
        self.rng = np.random.default_rng(22)

        ####  Layers  ####
        self.patch_embedder = SPEmbedder3Conv(in_feats=in_dim, hidden_dim=in_dim * 4, dropout=dropout)
        self.mesh_embedder = GMEmbedder(in_dim=self.patch_embedder.embed_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = nn.Linear(in_features=self.mesh_embedder.embed_dim, out_features=out_feats, bias=False)

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.patch_embedder.embed_dim), device=device)
        random_sequence = self.rng.permutation(len(patches))

        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(patches[random_sequence], batch_size=10, drop_last=False)
        for spider_patch in dataloader:
            readouts = torch.vstack((readouts, self.patch_embedder(spider_patch, spider_patch.ndata["aggregated_feats"], spider_patch.edata["weights"])))

        with mesh_graph.local_scope():
            dgl.reorder_graph(mesh_graph, node_permute_algo='custom', permute_config={'nodes_perm': random_sequence})
            if self.mesh_graph_edge_weights:
                mesh_embedding = self.mesh_embedder(mesh_graph, readouts, mesh_graph.edata["weights"])
            else:
                mesh_embedding = self.mesh_embedder(mesh_graph, readouts, None)

            return self.classifier(mesh_embedding)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class MeshNetworkARAR(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_feats, dropout, mesh_graph_edge_weights=True):
        super(MeshNetworkARAR, self).__init__()

        ####  Variables  ####
        self.name = self.__class__.__name__
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
        self.name = self.__class__.__name__
        self.readout_dim = readout_dim
        self.mesh_graph_edge_weights = mesh_graph_edge_weights

        ####  Layers  ####
        self.patch_embedder = PatchEmbedder2ConvLayerAR(in_dim, in_dim * 4, dropout)
        self.mesh_reader = GMReader2ConvAverageReadout(in_dim=self.patch_embedder.embed_dim, hidden_dim=hidden_dim, out_dim=out_feats, dropout=dropout)

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
        self.name = self.__class__.__name__
        self.readout_dim = readout_dim
        self.mesh_graph_edge_weights = mesh_graph_edge_weights

        ####  Layers  ####
        self.patch_embedder = PatchEmbedder2ConvLayer(in_dim, in_dim * 4, dropout)
        self.mesh_embedder = GMEmbedder2ConvUR(in_dim=self.patch_embedder.embed_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = nn.Linear(self.mesh_embedder.embed_dim, out_feats, bias=False)

        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.readout_dim), device=device)
        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(patches, batch_size=10, drop_last=False)
        for spider_patch in dataloader:
            readouts = torch.vstack((readouts, self.patch_embedder(spider_patch, spider_patch.ndata["aggregated_feats"], spider_patch.edata["weights"])))

        if self.mesh_graph_edge_weights:
            mesh_embedding = self.mesh_embedder(mesh_graph, readouts, mesh_graph.edata["weights"])
        else:
            mesh_embedding = self.mesh_embedder(mesh_graph, readouts, None)

        return self.classifier(mesh_embedding)

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
        self.name = self.__class__.__name__
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
        self.name = self.__class__.__name__
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
        self.name = self.__class__.__name__
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
        self.name = self.__class__.__name__
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
        self.name = self.__class__.__name__
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
