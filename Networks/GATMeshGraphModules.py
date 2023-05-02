import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch.utils import JumpingKnowledge
from tqdm import tqdm

from Networks.NormalizationModules import UnitedNormCommon
from Networks.UniversalReadout import UniversalReadout


class GATMGEmbedder(nn.Module):
    def __init__(self, feat_in_channels, readout_function="AR", jumping_mode=None, layers_num=3, residual_attn=True, exp_heads=False, dropout=0):
        super(GATMGEmbedder, self).__init__()
        exp_heads = 1 if exp_heads else 0
        feat_dimensions_multipliers = [(2 ** (i * exp_heads + 1)) for i in range(layers_num)]

        self.JUMPING_MODE = jumping_mode

        ####  Layers  ####
        self.GAT_layers = nn.ModuleList()
        self.GAT_layers.append(GATConv(in_feats=feat_in_channels, out_feats=feat_in_channels, num_heads=feat_dimensions_multipliers[0], feat_drop=dropout, attn_drop=dropout, residual=residual_attn, bias=False))
        for i in range(1, layers_num):
            self.GAT_layers.append(GATConv(in_feats=feat_in_channels * feat_dimensions_multipliers[i - 1], out_feats=feat_in_channels, num_heads=feat_dimensions_multipliers[i], feat_drop=dropout, attn_drop=dropout, residual=residual_attn, bias=False))

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)

        ####  Normalization Layers  ####
        self.normalizations = nn.ModuleList()
        for i in range(0, layers_num):
            self.normalizations.append(UnitedNormCommon(feat_in_channels * feat_dimensions_multipliers[i]))

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
        updated_feats = node_feats
        embeddings = []
        for idx, gat_layer in enumerate(self.GAT_layers):
            updated_feats = gat_layer(mesh_graph, updated_feats).flatten(1)
            updated_feats = self.normalizations[idx](mesh_graph, updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            updated_feats = self.dropout(updated_feats)
            mesh_graph.ndata['updated_feats'] = torch.unflatten(updated_feats, -1, (gat_layer._num_heads, -1)).mean(1)
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
