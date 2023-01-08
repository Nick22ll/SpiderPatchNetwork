import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv
from torch_geometric.nn import GraphNorm


class GATJumpMGEmbedderAR(nn.Module):
    def __init__(self, feat_in_channels, layers_num, residual, dropout, exp_heads=False):
        super(GATJumpMGEmbedderAR, self).__init__()
        exp_heads = 1 if exp_heads else 0
        feat_dimensions_multipliers = [(2 ** (i * exp_heads + 1)) for i in range(layers_num)]

        ####  Layers  ####
        self.GAT_layers = nn.ModuleList()
        self.GAT_layers.append(GATConv(in_feats=feat_in_channels, out_feats=feat_in_channels, num_heads=feat_dimensions_multipliers[0], feat_drop=dropout, attn_drop=dropout, residual=residual, bias=False))
        for i in range(1, layers_num):
            self.GAT_layers.append(
                GATConv(in_feats=feat_in_channels * np.prod(feat_dimensions_multipliers[:i]), out_feats=feat_in_channels * np.prod(feat_dimensions_multipliers[:i]), num_heads=feat_dimensions_multipliers[i], feat_drop=dropout, attn_drop=dropout, residual=residual, bias=False))

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.normalizations = nn.ModuleList()
        self.normalizations.append(GraphNorm(feat_in_channels, eps=1e-5))
        for i in range(1, layers_num):
            self.normalizations.append(GraphNorm(feat_in_channels * np.prod(feat_dimensions_multipliers[:i]), eps=1e-5))

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = feat_in_channels + sum(feat_in_channels * np.prod(feat_dimensions_multipliers[:i]) for i in range(1, layers_num))

    def forward(self, spider_patch, node_feats):
        updated_feats = node_feats
        readout = torch.empty(0, device=node_feats.device)
        for idx, gat_layer in enumerate(self.GAT_layers):
            updated_feats = gat_layer(spider_patch, updated_feats.flatten(1))
            updated_feats = self.normalizations[idx](updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            updated_feats = self.dropout(updated_feats)
            spider_patch.ndata['updated_feats'] = updated_feats.mean(1)
            readout = torch.hstack((readout, dgl.mean_nodes(spider_patch, 'updated_feats')))
        return self.LeakyReLU(readout)
