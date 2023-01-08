import os

import numpy as np
import torch.nn.functional as F
import dgl
import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GATConv, SortPooling

from Networks.UniversalReadout import UniversalReadout


class GATNodeWeigher(nn.Module):
    def __init__(self, feat_in_channels, weights_in_channels):
        super(GATNodeWeigher, self).__init__()

        ####  Readouts Utils  ####
        self.readout_fc = nn.Linear(weights_in_channels, weights_in_channels * 2, bias=False)
        self.readout_attn_fc = nn.Linear((weights_in_channels * 2) + feat_in_channels, 1, bias=False)

        #### Activations  ####
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, spider_patch, weights_feats, feats_feats):
        out = self.readout_fc(spider_patch.ndata[weights_feats])
        out = torch.hstack((out, spider_patch.ndata[feats_feats]))
        spider_patch.ndata["tmp"] = self.activation(self.readout_attn_fc(out))
        return dgl.softmax_nodes(spider_patch, "tmp") * torch.mean(spider_patch.batch_num_nodes().float())


class GATJumpSPEmbedderUR(nn.Module):
    def __init__(self, feat_in_channels, weights_in_channels, layers_num, residual, dropout, exp_heads=False, use_node_weights=False):
        super(GATJumpSPEmbedderUR, self).__init__()
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

        ####  Readouts Utils  ####
        if use_node_weights:
            self.weighers = nn.ModuleList()
            self.weighers.append(GATNodeWeigher(feat_in_channels, weights_in_channels))
            for i in range(1, layers_num):
                self.weighers.append(GATNodeWeigher(feat_in_channels * np.prod(feat_dimensions_multipliers[:i]), weights_in_channels))

        #### Readout Layers  ####
        self.readouts = nn.ModuleList()
        self.readouts.append(UniversalReadout(feat_in_channels, 0.05))
        for i in range(1, layers_num):
            self.readouts.append(UniversalReadout(feat_in_channels * np.prod(feat_dimensions_multipliers[:i]), 0.05))

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = sum(r.readout_dim for r in self.readouts)
        self.use_node_weights = use_node_weights
        self.weights_in_channels = weights_in_channels

    def forward(self, spider_patch, node_feats):
        # node_feats = self.chn_fixer(node_feats)
        updated_feats = node_feats
        readout = torch.empty(0, device=node_feats.device)
        for idx, gat_layer in enumerate(self.GAT_layers):
            updated_feats = gat_layer(spider_patch, updated_feats.flatten(1))
            updated_feats = self.normalizations[idx](updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            updated_feats = self.dropout(updated_feats)
            spider_patch.ndata['updated_feats'] = updated_feats.mean(1)
            if self.use_node_weights:
                if self.weights_in_channels > 0:
                    spider_patch.ndata['node_weights'] = self.weighers[idx](spider_patch, "aggregated_weights", "updated_feats")
                else:
                    spider_patch.ndata['node_weights'] = spider_patch["aggregated_weights"]
            else:
                readout = torch.hstack((readout, dgl.mean_nodes(spider_patch, 'updated_feats', None)))
        return self.LeakyReLU(readout)


class GATJumpSPEmbedderAR(nn.Module):
    def __init__(self, feat_in_channels, weights_in_channels, layers_num, residual, dropout, exp_heads=False, use_node_weights=False):
        super(GATJumpSPEmbedderAR, self).__init__()
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

        ####  Readouts Utils  ####
        if use_node_weights and weights_in_channels > 0:
            self.weighers = nn.ModuleList()
            self.weighers.append(GATNodeWeigher(feat_in_channels, weights_in_channels))
            for i in range(1, layers_num):
                self.weighers.append(GATNodeWeigher(feat_in_channels * np.prod(feat_dimensions_multipliers[:i]), weights_in_channels))

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = feat_in_channels + sum(feat_in_channels * np.prod(feat_dimensions_multipliers[:i]) for i in range(1, layers_num))
        self.use_node_weights = use_node_weights
        self.weights_in_channels = weights_in_channels

    def forward(self, spider_patch, node_feats):
        # node_feats = self.chn_fixer(node_feats)
        updated_feats = node_feats
        readout = torch.empty(0, device=node_feats.device)
        for idx, gat_layer in enumerate(self.GAT_layers):
            updated_feats = gat_layer(spider_patch, updated_feats.flatten(1))
            updated_feats = self.normalizations[idx](updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            updated_feats = self.dropout(updated_feats)
            spider_patch.ndata['updated_feats'] = updated_feats.mean(1)
            if self.use_node_weights:
                if self.weights_in_channels > 0:
                    spider_patch.ndata['node_weights'] = self.weighers[idx](spider_patch, "aggregated_weights", "updated_feats")
                else:
                    spider_patch.ndata['node_weights'] = spider_patch["aggregated_weights"]
                readout = torch.hstack((readout, dgl.mean_nodes(spider_patch, 'updated_feats', "node_weights")))
            else:
                readout = torch.hstack((readout, dgl.mean_nodes(spider_patch, 'updated_feats', None)))
        return self.LeakyReLU(readout)
