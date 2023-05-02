import os

import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader

from Networks.CONVSpiderPatchModules import CONVSPEmbedder
from Networks.GATMeshGraphModules import GATMGEmbedder
from Networks.CONVMeshGraphModules import CONVMGEmbedder
from Networks.GATSpiderPatchModules import GATSPEmbedder

from Networks.MLP import MLP
from SpiderPatch.SpiderPatch import SPMatrixDistanceV1


class SimplestNetwork(nn.Module):
    def __init__(self, feat_in_channels, out_feats, readout_function="AR", use_node_weights=False, weights_in_channels=0):
        super(SimplestNetwork, self).__init__()

        ####  Variables  ####
        self.name = self.__class__.__name__
        self.rng = np.random.default_rng(17)

        ####  Layers  ####
        self.patch_embedder = GATSPEmbedder(feat_in_dim=feat_in_channels, weights_in_dim=weights_in_channels, readout_function=readout_function, layers_num=3, residual_attn=True)

        self.classifier = nn.Linear(in_features=self.patch_embedder.embed_dim, out_features=out_feats, bias=False)

    def forward(self, mesh_graph, batched_patches, device):
        BATCH_SIZE = 128 if len(batched_patches) >= 128 else len(batched_patches)
        SP_embeddings = torch.empty((len(batched_patches), self.patch_embedder.embed_dim), device=device)
        dataloader = GraphDataLoader(batched_patches, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)  # patches[random_sequence]
        # noinspection PyTypeChecker
        for idx, SP_batch in enumerate(dataloader):
            SP_embeddings[(idx * BATCH_SIZE):((idx * BATCH_SIZE) + SP_batch.batch_size)] = self.patch_embedder(SP_batch, SP_batch.ndata["aggregated_feats"])

        with mesh_graph.local_scope():
            # mesh_graph.__class__ = dgl.DGLGraph
            # dgl.reorder_graph(mesh_graph, node_permute_algo='custom', permute_config={'nodes_perm': random_sequence})
            mesh_graph.ndata["readouts"] = SP_embeddings
            MG_embeddings = dgl.mean_nodes(mesh_graph, "readouts")

        return self.classifier(MG_embeddings), MG_embeddings

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class TestNetwork(nn.Module):
    def __init__(self, feat_in_channels, weights_in_channels, out_feats, network_parameters, use_SP_triplet=False):
        super(TestNetwork, self).__init__()

        ####  Variables  ####
        self.name = self.__class__.__name__
        self.rng = np.random.default_rng(17)
        self.SP_triplet = use_SP_triplet

        self.BATCH_SIZE = network_parameters["MG"]["SP_batch_size"]

        ####  Layers  ####
        self.patch_embedder = network_parameters["SP"]["module"](feat_in_dim=feat_in_channels,
                                                                 readout_function=network_parameters["SP"]["readout_function"],
                                                                 jumping_mode=network_parameters["SP"]["jumping_mode"],
                                                                 layers_num=network_parameters["SP"]["layers_num"],
                                                                 weigher_mode=network_parameters["SP"]["weigher_mode"],
                                                                 weights_in_dim=weights_in_channels,
                                                                 dropout=network_parameters["SP"]["dropout"],
                                                                 residual_attn=network_parameters["SP"]["residual"],
                                                                 exp_heads=network_parameters["SP"]["exp_heads"]
                                                                 )

        self.mesh_embedder = network_parameters["MG"]["module"](feat_in_channels=self.patch_embedder.embed_dim,
                                                                readout_function=network_parameters["MG"]["readout_function"],
                                                                jumping_mode=network_parameters["MG"]["jumping_mode"],
                                                                layers_num=network_parameters["MG"]["layers_num"],
                                                                residual_attn=network_parameters["MG"]["residual"],
                                                                exp_heads=network_parameters["MG"]["exp_heads"],
                                                                dropout=network_parameters["MG"]["dropout"])

        self.classifier = nn.Linear(in_features=self.mesh_embedder.embed_dim, out_features=out_feats, bias=False)
        # self.classifier = MLP(in_dim=self.mesh_embedder.embed_dim, hidden_dim=int(self.mesh_embedder.embed_dim / 2), out_dim=out_feats)

    def forward(self, mesh_graph, batched_patches, device):
        BATCH_SIZE = self.BATCH_SIZE if len(batched_patches) >= self.BATCH_SIZE else len(batched_patches)
        SP_embeddings = torch.empty((len(batched_patches), self.patch_embedder.embed_dim), device=device)
        dataloader = GraphDataLoader(batched_patches, batch_size=BATCH_SIZE, drop_last=False)  # patches[random_sequence]
        if self.SP_triplet:
            # random_sequence = self.rng.permutation(len(patches))
            positive_embeddings = torch.empty((len(batched_patches), self.patch_embedder.embed_dim), dtype=torch.long, device=device)
            negative_embeddings = torch.empty((len(batched_patches), self.patch_embedder.embed_dim), dtype=torch.long, device=device)

            for idx, SP_batch in enumerate(dataloader):
                batch_distance_matrix = SPMatrixDistanceV1(SP_batch, "aggregated_feats")
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
                MG_embeddings = self.mesh_embedder(mesh_graph, SP_embeddings)

            return self.classifier(MG_embeddings), MG_embeddings, SP_embeddings, positive_embeddings, negative_embeddings

        else:

            for idx, SP_batch in enumerate(dataloader):
                SP_embeddings[(idx * BATCH_SIZE):((idx * BATCH_SIZE) + SP_batch.batch_size)] = self.patch_embedder(SP_batch, SP_batch.ndata["aggregated_feats"])

            with mesh_graph.local_scope():
                MG_embeddings = self.mesh_embedder(mesh_graph, SP_embeddings)

            return self.classifier(MG_embeddings), MG_embeddings

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

# class GATJumpNetwork(nn.Module):
#     def __init__(self, feat_in_channels, weights_in_channels, out_feats, readout_function="AR", use_node_weights=True, use_SP_triplet=False):
#         super(GATJumpNetwork, self).__init__()
#
#
#         ####  Variables  ####
#         self.name = self.__class__.__name__
#         self.rng = np.random.default_rng(17)
#         self.SP_triplet = use_SP_triplet
#
#         ####  Layers  ####
#         self.patch_embedder = GATJumpSPEmbedder(feat_in_channels=feat_in_channels, weights_in_channels=weights_in_channels, readout_function=readout_function, layers_num=3, residual=True, exp_heads=False, use_node_weights=use_node_weights)
#         self.mesh_embedder = GATJumpMGEmbedder(feat_in_channels=self.patch_embedder.embed_dim, readout_function=readout_function, layers_num=2, residual=True, exp_heads=False)
#         self.classifier = nn.Linear(in_features=self.mesh_embedder.embed_dim, out_features=out_feats, bias=False)
#
#     def forward(self, mesh_graph, batched_patches, device):
#         BATCH_SIZE = 128 if len(batched_patches) >= 128 else len(batched_patches)
#         SP_embeddings = torch.empty((len(batched_patches), self.patch_embedder.embed_dim), device=device)
#         dataloader = GraphDataLoader(batched_patches, batch_size=BATCH_SIZE, drop_last=False)  # patches[random_sequence]
#         if self.SP_triplet:
#             # random_sequence = self.rng.permutation(len(patches))
#             positive_embeddings = torch.empty((len(batched_patches), self.patch_embedder.embed_dim), dtype=torch.long, device=device)
#             negative_embeddings = torch.empty((len(batched_patches), self.patch_embedder.embed_dim), dtype=torch.long, device=device)
#
#             for idx, SP_batch in enumerate(dataloader):
#                 batch_distance_matrix = SPMatrixDistanceV1(SP_batch, "aggregated_feats")
#                 _, sorted_indices = torch.sort(batch_distance_matrix, dim=1)
#                 negative_SP_idx = sorted_indices[:, -5:]
#                 positive_SP_idx = sorted_indices[:, 1:6]  # select from 1 to 6 to avoid to keep the same SP
#                 batch_embeddings = self.patch_embedder(SP_batch, SP_batch.ndata["aggregated_feats"])
#                 _, hardest_negatives_idx = torch.min(torch.pairwise_distance(batch_embeddings[:, None, ], batch_embeddings[negative_SP_idx]), dim=1)
#                 _, hardest_positives_idx = torch.max(torch.pairwise_distance(batch_embeddings[:, None, ], batch_embeddings[positive_SP_idx]), dim=1)
#                 positive_embeddings[(idx * BATCH_SIZE):((idx * BATCH_SIZE) + SP_batch.batch_size)] = torch.clone(batch_embeddings[positive_SP_idx][torch.arange(batch_embeddings.size(0)), hardest_positives_idx])
#                 negative_embeddings[(idx * BATCH_SIZE):((idx * BATCH_SIZE) + SP_batch.batch_size)] = torch.clone(batch_embeddings[negative_SP_idx][torch.arange(batch_embeddings.size(0)), hardest_negatives_idx])
#                 SP_embeddings[(idx * BATCH_SIZE):((idx * BATCH_SIZE) + SP_batch.batch_size)] = torch.clone(batch_embeddings)
#
#             with mesh_graph.local_scope():
#                 MG_embeddings = self.mesh_embedder(mesh_graph, SP_embeddings)
#
#             return self.classifier(MG_embeddings), MG_embeddings, SP_embeddings, positive_embeddings, negative_embeddings
#
#         else:
#             for idx, SP_batch in enumerate(dataloader):
#                 SP_embeddings[(idx * BATCH_SIZE):((idx * BATCH_SIZE) + SP_batch.batch_size)] = self.patch_embedder(SP_batch, SP_batch.ndata["aggregated_feats"])
#
#             with mesh_graph.local_scope():
#                 MG_embeddings = self.mesh_embedder(mesh_graph, SP_embeddings)
#
#             return self.classifier(MG_embeddings), MG_embeddings
#
#     def save(self, path):
#         os.makedirs(path, exist_ok=True)
#         torch.save(self.state_dict(), path + "/network.pt")
#
#     def load(self, path):
#         self.load_state_dict(torch.load(path))
#         self.eval()
#
#
#
#
#
