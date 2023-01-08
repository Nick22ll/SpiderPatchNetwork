import os
from online_triplet_loss.losses import *
import dgl
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GATConv, SortPooling

from Networks.GATMeshGraphModules import GATJumpMGEmbedderAR
from Networks.GATSpiderPatchModules import GATJumpSPEmbedderUR, GATJumpSPEmbedderAR
from Networks.MLP import GenericMLP
from Networks.SpiralReadout import SpiralReadout
from Networks.UniversalReadout import UniversalReadout
from SpiderPatch.SpiderPatch import SP_matrix_distanceV1


class GATCriterion(nn.Module):
    def __init__(self, alpha=1, beta=1, gamma=1, use_SP_triplet=False):
        super(GATCriterion, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.use_SP_triplet = use_SP_triplet
        if use_SP_triplet:
            self.SP_triplet = nn.TripletMarginLoss()

    def forward(self, pred_class, labels, MG_embedding, SP_embeddings, positive_SP_emb, negative_SP_emb):
        # normalize input embeddings
        MG_embedding = torch.nn.functional.normalize(MG_embedding)

        if self.use_SP_triplet:
            # normalize input embeddings
            SP_embeddings = torch.nn.functional.normalize(SP_embeddings)
            positive_SP_emb = torch.nn.functional.normalize(positive_SP_emb)
            negative_SP_emb = torch.nn.functional.normalize(negative_SP_emb)
            SP_loss = self.SP_triplet(SP_embeddings, positive_SP_emb, negative_SP_emb)
            return self.cross_entropy(pred_class, labels) + batch_hard_triplet_loss(labels, MG_embedding, margin=self.MG_margin) + SP_loss
        else:
            return (self.alpha * self.cross_entropy(pred_class, labels)) + (self.beta * batch_hard_triplet_loss(labels, MG_embedding, margin=1))


class SimplestURNetwork(nn.Module):
    def __init__(self, feat_in_channels, weights_in_channels, out_feats, dropout):
        super(SimplestURNetwork, self).__init__()

        ####  Variables  ####
        self.name = self.__class__.__name__
        self.rng = np.random.default_rng(22)

        ####  Layers  ####
        self.patch_embedder = GATJumpSPEmbedderUR(feat_in_channels=feat_in_channels, weights_in_channels=weights_in_channels, layers_num=3, residual=True, dropout=dropout)

        self.classifier = nn.Linear(in_features=self.patch_embedder.embed_dim, out_features=out_feats, bias=False)

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.patch_embedder.embed_dim), device=device)
        random_sequence = self.rng.permutation(len(patches))

        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(patches[random_sequence], batch_size=25, drop_last=False)
        for spider_patch in dataloader:
            readouts = torch.vstack((readouts, self.patch_embedder(spider_patch, spider_patch.ndata["aggregated_feats"])))

        reordered_graph = dgl.reorder_graph(mesh_graph, node_permute_algo='custom', permute_config={'nodes_perm': random_sequence})
        reordered_graph.ndata["readouts"] = readouts
        return self.classifier(dgl.mean_nodes(reordered_graph, "readouts"))

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class SimplestNetwork(nn.Module):
    def __init__(self, feat_in_channels, out_feats, dropout, use_node_weights=False, weights_in_channels=0):
        super(SimplestNetwork, self).__init__()

        ####  Variables  ####
        self.name = self.__class__.__name__
        self.rng = np.random.default_rng(22)

        ####  Layers  ####
        self.patch_embedder = GATJumpSPEmbedderAR(feat_in_channels=feat_in_channels, weights_in_channels=weights_in_channels, layers_num=3, residual=True, dropout=dropout, exp_heads=True, use_node_weights=use_node_weights)
        self.classifier = nn.Linear(in_features=self.patch_embedder.embed_dim, out_features=out_feats, bias=False)

    def forward(self, mesh_graph, patches, device):
        readouts = torch.empty((0, self.patch_embedder.embed_dim), device=device)
        random_sequence = self.rng.permutation(len(patches))

        # noinspection PyTypeChecker
        dataloader = GraphDataLoader(patches[random_sequence], batch_size=25, drop_last=False)
        for spider_patch in dataloader:
            readouts = torch.vstack((readouts, self.patch_embedder(spider_patch, spider_patch.ndata["aggregated_feats"])))

        reordered_graph = dgl.reorder_graph(mesh_graph, node_permute_algo='custom', permute_config={'nodes_perm': random_sequence})
        reordered_graph.ndata["readouts"] = readouts
        return self.classifier(dgl.mean_nodes(reordered_graph, "readouts"))

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class GATJumpARNetwork(nn.Module):
    def __init__(self, feat_in_channels, weights_in_channels, out_feats, dropout, use_node_weights=True):
        super(GATJumpARNetwork, self).__init__()

        ####  Variables  ####
        self.name = self.__class__.__name__
        self.rng = np.random.default_rng(22)

        ####  Layers  ####
        self.patch_embedder = GATJumpSPEmbedderAR(feat_in_channels=feat_in_channels, weights_in_channels=weights_in_channels, layers_num=3, residual=True, dropout=dropout, exp_heads=False, use_node_weights=use_node_weights)
        self.mesh_embedder = GATJumpMGEmbedderAR(feat_in_channels=self.patch_embedder.embed_dim, layers_num=3, residual=True, dropout=dropout, exp_heads=False)
        self.classifier = nn.Linear(in_features=self.mesh_embedder.embed_dim, out_features=out_feats, bias=False)

    def forward(self, mesh_graph, patches, device):
        if self.training:
            BATCH_SIZE = 50 if len(patches) >= 50 else len(patches)
            SP_embeddings = torch.empty((len(patches), self.patch_embedder.embed_dim), device=device)
            random_sequence = self.rng.permutation(len(patches))
            positive_embeddings = torch.empty((len(patches), self.patch_embedder.embed_dim), dtype=torch.long, device=device)
            negative_embeddings = torch.empty((len(patches), self.patch_embedder.embed_dim), dtype=torch.long, device=device)

            # noinspection PyTypeChecker

            dataloader = GraphDataLoader(patches[random_sequence], batch_size=BATCH_SIZE, drop_last=False)
            for idx, SP_batch in enumerate(dataloader):
                batch_distance_matrix = SP_matrix_distanceV1(SP_batch, "aggregated_feats")
                _, sorted_indices = torch.sort(batch_distance_matrix, dim=1)
                negative_SP_idx = sorted_indices[:, -5:]
                positive_SP_idx = sorted_indices[:, 1:6]  # select from 1 to 6 to avoid to keep the same SP
                batch_embeddings = self.patch_embedder(SP_batch, SP_batch.ndata["aggregated_feats"])
                _, hardest_negatives_idx = torch.min(torch.pairwise_distance(batch_embeddings[:, None, ], batch_embeddings[negative_SP_idx]), dim=1)
                _, hardest_positives_idx = torch.max(torch.pairwise_distance(batch_embeddings[:, None, ], batch_embeddings[positive_SP_idx]), dim=1)
                positive_embeddings[(idx * BATCH_SIZE):((idx * BATCH_SIZE) + SP_batch.batch_size)] = torch.clone(batch_embeddings[positive_SP_idx][torch.arange(batch_embeddings.size(0)), hardest_positives_idx])
                negative_embeddings[(idx * BATCH_SIZE):((idx * BATCH_SIZE) + SP_batch.batch_size)] = torch.clone(batch_embeddings[negative_SP_idx][torch.arange(batch_embeddings.size(0)), hardest_negatives_idx])
                SP_embeddings[(idx * BATCH_SIZE):((idx * BATCH_SIZE) + SP_batch.batch_size)] = torch.clone(batch_embeddings)

            with mesh_graph.local_scope():
                mesh_graph.__class__ = dgl.DGLGraph
                dgl.reorder_graph(mesh_graph, node_permute_algo='custom', permute_config={'nodes_perm': random_sequence})
                MG_embedding = self.mesh_embedder(mesh_graph, SP_embeddings)

            return self.classifier(MG_embedding), MG_embedding, SP_embeddings, positive_embeddings, negative_embeddings

        else:
            SP_embeddings = torch.empty((0, self.patch_embedder.embed_dim), device=device)
            # noinspection PyTypeChecker
            BATCH_SIZE = 50
            dataloader = GraphDataLoader(patches, batch_size=BATCH_SIZE, drop_last=False)
            for SP_batch in dataloader:
                SP_embeddings = torch.vstack((SP_embeddings, self.patch_embedder(SP_batch, SP_batch.ndata["aggregated_feats"])))

            MG_embedding = self.mesh_embedder(mesh_graph, SP_embeddings)
            return self.classifier(MG_embedding)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


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


class GATJumpARNetworkBatch(nn.Module):
    def __init__(self, feat_in_channels, weights_in_channels, out_feats, dropout, use_node_weights=True, use_SP_triplet=False):
        super(GATJumpARNetworkBatch, self).__init__()

        ####  Variables  ####
        self.name = self.__class__.__name__
        self.rng = np.random.default_rng(22)
        self.SP_triplet = use_SP_triplet

        ####  Layers  ####
        self.patch_embedder = GATJumpSPEmbedderAR(feat_in_channels=feat_in_channels, weights_in_channels=weights_in_channels, layers_num=3, residual=True, dropout=dropout, exp_heads=False, use_node_weights=use_node_weights)
        self.mesh_embedder = GATJumpMGEmbedderAR(feat_in_channels=self.patch_embedder.embed_dim, layers_num=2, residual=True, dropout=dropout, exp_heads=False)
        self.classifier = nn.Linear(in_features=self.mesh_embedder.embed_dim, out_features=out_feats, bias=False)

    def forward(self, mesh_graph, batched_patches, device):
        BATCH_SIZE = 128 if len(batched_patches) >= 128 else len(batched_patches)
        SP_embeddings = torch.empty((len(batched_patches), self.patch_embedder.embed_dim), device=device)
        dataloader = GraphDataLoader(batched_patches, batch_size=BATCH_SIZE, drop_last=False)  # patches[random_sequence]
        if self.SP_triplet:
            # random_sequence = self.rng.permutation(len(patches))
            positive_embeddings = torch.empty((len(batched_patches), self.patch_embedder.embed_dim), dtype=torch.long, device=device)
            negative_embeddings = torch.empty((len(batched_patches), self.patch_embedder.embed_dim), dtype=torch.long, device=device)

            for idx, SP_batch in enumerate(dataloader):
                batch_distance_matrix = SP_matrix_distanceV1(SP_batch, "aggregated_feats")
                _, sorted_indices = torch.sort(batch_distance_matrix, dim=1)
                negative_SP_idx = sorted_indices[:, -5:]
                positive_SP_idx = sorted_indices[:, 1:6]  # select from 1 to 6 to avoid to keep the same SP
                batch_embeddings = self.patch_embedder(SP_batch, SP_batch.ndata["aggregated_feats"])
                _, hardest_negatives_idx = torch.min(torch.pairwise_distance(batch_embeddings[:, None, ], batch_embeddings[negative_SP_idx]), dim=1)
                _, hardest_positives_idx = torch.max(torch.pairwise_distance(batch_embeddings[:, None, ], batch_embeddings[positive_SP_idx]), dim=1)
                positive_embeddings[(idx * BATCH_SIZE):((idx * BATCH_SIZE) + SP_batch.batch_size)] = torch.clone(batch_embeddings[positive_SP_idx][torch.arange(batch_embeddings.size(0)), hardest_positives_idx])
                negative_embeddings[(idx * BATCH_SIZE):((idx * BATCH_SIZE) + SP_batch.batch_size)] = torch.clone(batch_embeddings[negative_SP_idx][torch.arange(batch_embeddings.size(0)), hardest_negatives_idx])
                SP_embeddings[(idx * BATCH_SIZE):((idx * BATCH_SIZE) + SP_batch.batch_size)] = torch.clone(batch_embeddings)
        else:
            # noinspection PyTypeChecker
            for idx, SP_batch in enumerate(dataloader):
                batch_embeddings = self.patch_embedder(SP_batch, SP_batch.ndata["aggregated_feats"])
                SP_embeddings[(idx * BATCH_SIZE):((idx * BATCH_SIZE) + SP_batch.batch_size)] = torch.clone(batch_embeddings)
            positive_embeddings, negative_embeddings = None, None

        with mesh_graph.local_scope():
            # mesh_graph.__class__ = dgl.DGLGraph
            # dgl.reorder_graph(mesh_graph, node_permute_algo='custom', permute_config={'nodes_perm': random_sequence})
            MG_embeddings = self.mesh_embedder(mesh_graph, SP_embeddings)

        return self.classifier(MG_embeddings), MG_embeddings, SP_embeddings, positive_embeddings, negative_embeddings

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
