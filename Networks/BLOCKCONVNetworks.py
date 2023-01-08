import os

import numpy as np
import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader

from Networks.CONVMeshGraphModules import GMEmbedder
from Networks.CONVSpiderPatchModules import SPEmbedder2Conv


class BlockMeshNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, block_dim, out_feats, dropout, block_reduction="concat", mesh_graph_edge_weights=True):
        super(BlockMeshNetwork, self).__init__()

        ####  Variables  ####
        self.name = "BlockCONVMeshNetwork"
        self.mesh_graph_edge_weights = mesh_graph_edge_weights
        self.block_reduction = block_reduction

        ####  Layers  ####
        self.patch_embedder = SPEmbedder2Conv(in_feats=in_dim, hidden_dim=in_dim * 4, dropout=dropout)
        self.mesh_reader = GMEmbedder(in_dim=self.patch_embedder.embed_dim, hidden_dim=hidden_dim, dropout=dropout)

        if self.block_reduction == "concat":
            self.classifier = nn.Linear(in_features=self.mesh_reader.embed_dim * block_dim, out_features=out_feats, bias=False)
        elif self.block_reduction == "average":
            self.classifier = nn.Linear(in_features=self.mesh_reader.embed_dim, out_features=out_feats, bias=False)
        else:
            raise ()

    def forward(self, mesh_graphs, patches_list, device):
        block_readout = torch.empty((0, self.mesh_reader.embed_dim), device=device)
        mesh_graph_dataloader = GraphDataLoader(mesh_graphs, batch_size=1, drop_last=False)
        for sampler, mesh_graph in enumerate(mesh_graph_dataloader):
            readouts = torch.empty((0, self.patch_embedder.embed_dim), device=device)
            # noinspection PyTypeChecker
            dataloader = GraphDataLoader(np.random.permutation(patches_list[sampler]), batch_size=10, drop_last=False)
            for spider_patch in dataloader:
                readouts = torch.vstack((readouts, self.patch_embedder(spider_patch, spider_patch.ndata["aggregated_feats"], spider_patch.edata["weights"])))
            if self.mesh_graph_edge_weights:
                block_readout = torch.vstack((block_readout, self.mesh_reader(mesh_graph, readouts, mesh_graph.edata["weights"])))
            else:
                block_readout = torch.vstack((block_readout, self.mesh_reader(mesh_graph, readouts, None)))

        if self.block_reduction == "concat":
            block_readout = block_readout[1:]
            block_readout = block_readout.flatten()
        elif self.block_reduction == "average":
            block_readout = torch.mean(block_readout, dim=0)
        else:
            raise ()
        return self.classifier(block_readout)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class BlockMeshNetworkPEARAR(nn.Module):
    def __init__(self, in_dim, hidden_dim, block_dim, out_feats, dropout, block_reduction="concat", mesh_graph_edge_weights=True):
        super(BlockMeshNetworkPEARAR, self).__init__()

        ####  Variables  ####
        self.name = "BlockMeshNetworkPEARAR"
        self.mesh_graph_edge_weights = mesh_graph_edge_weights
        self.block_reduction = block_reduction

        ####  Layers  ####
        self.patch_embedder = PatchEmbedder2ConvLayerAR(in_dim, 32, dropout)
        self.mesh_reader = GMEmbedder2ConvAverageReadout(in_dim=self.patch_embedder.embed_dim, hidden_dim=hidden_dim, dropout=dropout)

        if self.block_reduction == "concat":
            self.classifier = nn.Linear(in_features=self.mesh_reader.embed_dim * block_dim, out_features=out_feats, bias=False)
        elif self.block_reduction == "average":
            self.classifier = nn.Linear(in_features=self.mesh_reader.embed_dim, out_features=out_feats, bias=False)
        else:
            raise ()

    def forward(self, mesh_graphs, patches_list, device):
        block_readout = torch.empty((0, self.mesh_reader.embed_dim), device=device)
        mesh_graph_dataloader = GraphDataLoader(mesh_graphs, batch_size=1, drop_last=False)
        for sampler, mesh_graph in enumerate(mesh_graph_dataloader):
            readouts = torch.empty((0, self.patch_embedder.embed_dim), device=device)
            # noinspection PyTypeChecker
            dataloader = GraphDataLoader(np.random.permutation(patches_list[sampler]), batch_size=10, drop_last=False)
            for spider_patch in dataloader:
                readouts = torch.vstack((readouts, self.patch_embedder(spider_patch, spider_patch.ndata["aggregated_feats"], spider_patch.edata["weights"])))

            if self.mesh_graph_edge_weights:
                block_readout = torch.vstack((block_readout, self.mesh_reader(mesh_graph, readouts, mesh_graph.edata["weights"])))
            else:
                block_readout = torch.vstack((block_readout, self.mesh_reader(mesh_graph, readouts, None)))

        if self.block_reduction == "concat":
            block_readout = block_readout[1:]
            block_readout = block_readout.flatten()
        elif self.block_reduction == "average":
            block_readout = torch.mean(block_readout, dim=0)
        else:
            raise ()
        return self.classifier(block_readout)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class BlockMeshNetworkPEARUR(nn.Module):
    def __init__(self, in_dim, block_dim, out_feats, dropout, block_reduction="concat", mesh_graph_edge_weights=True):
        super(BlockMeshNetworkPEARUR, self).__init__()

        ####  Variables  ####
        self.name = "BlockMeshNetworkPEARUR"
        self.mesh_graph_edge_weights = mesh_graph_edge_weights
        self.block_reduction = block_reduction

        ####  Layers  ####
        self.patch_embedder = PatchEmbedder3ConvLayerAR(in_dim, in_dim * 2, dropout)
        self.mesh_reader = GMEmbedder(in_dim=self.patch_embedder.embed_dim, hidden_dim=self.patch_embedder.embed_dim * 4, dropout=dropout)

        if self.block_reduction == "concat":
            self.classifier = nn.Linear(in_features=self.mesh_reader.embed_dim * block_dim, out_features=out_feats, bias=False)
        elif self.block_reduction == "average":
            self.classifier = nn.Linear(in_features=self.mesh_reader.embed_dim, out_features=out_feats, bias=False)
        else:
            raise ()

        self.activation = nn.LeakyReLU()

    def forward(self, mesh_graphs, patches_list, device):
        block_readout = torch.empty((0, self.mesh_reader.embed_dim), device=device)
        mesh_graph_dataloader = GraphDataLoader(mesh_graphs, batch_size=1, drop_last=False)
        for sampler, mesh_graph in enumerate(mesh_graph_dataloader):
            readouts = torch.empty((0, self.patch_embedder.embed_dim), device=device)
            # noinspection PyTypeChecker
            dataloader = GraphDataLoader(np.random.permutation(patches_list[sampler]), batch_size=10, drop_last=False)
            for spider_patch in dataloader:
                readouts = torch.vstack((readouts, self.patch_embedder(spider_patch, spider_patch.ndata["aggregated_feats"], spider_patch.edata["weights"])))
            if self.mesh_graph_edge_weights:
                block_readout = torch.vstack((block_readout, self.mesh_reader(mesh_graph, readouts, mesh_graph.edata["weights"])))
            else:
                block_readout = torch.vstack((block_readout, self.mesh_reader(mesh_graph, readouts, None)))

        if self.block_reduction == "concat":
            block_readout = block_readout[1:]
            block_readout = block_readout.flatten()
        elif self.block_reduction == "average":
            block_readout = torch.mean(block_readout, dim=0)
        else:
            raise ()
        return self.classifier(self.activation(block_readout))

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
