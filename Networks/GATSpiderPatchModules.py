import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv, JumpingKnowledge
from tqdm import tqdm

from Networks.NormalizationModules import UnitedNormCommon
from Networks.UniversalReadout import UniversalReadout


class GATNodeWeigher(nn.Module):
    def __init__(self, feat_in_channels, weights_dim, use_additional_feats=False):
        super(GATNodeWeigher, self).__init__()
        self.use_additional_feats = use_additional_feats

        ####  Readouts Utils  ####
        self.readout_fc = nn.Linear(weights_dim, weights_dim * 2, bias=False)
        if self.use_additional_feats:
            self.readout_attn_fc = nn.Linear((weights_dim * 2) + feat_in_channels, 1, bias=False)
        else:
            self.readout_attn_fc = nn.Linear((weights_dim * 2), 1, bias=False)

        #### Activations  ####
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, spider_patch, weights_feats, feats_feats):
        out = self.readout_fc(spider_patch.ndata[weights_feats])
        if self.use_additional_feats:
            out = torch.hstack((out, spider_patch.ndata[feats_feats]))
        spider_patch.ndata["tmp"] = self.activation(self.readout_attn_fc(out))
        return dgl.softmax_nodes(spider_patch, "tmp") * torch.mean(spider_patch.batch_num_nodes().float())


class GATWeightedSP(nn.Module):
    def __init__(self, feat_in_dim, readout_function="AR", weigher_mode=None, weights_in_dim=0, dropout=0, *args):
        super(GATWeightedSP, self).__init__()

        ####  Readouts Utils  ####
        self.weigher_mode = weigher_mode

        if self.weigher_mode is not None:
            if self.weigher_mode == "sp_weights":
                self.weigher = nn.Identity()
            elif self.weigher_mode == "attn_weights":
                self.weigher = GATNodeWeigher(feat_in_dim, weights_in_dim, False)
            elif self.weigher_mode == "attn_weights+feats":
                self.weigher = GATNodeWeigher(feat_in_dim, weights_in_dim, True)
            else:
                raise "Wrong Weigher Mode"

        ####  Readout  ####
        if readout_function == "UR":
            self.readout = UniversalReadout(feat_in_dim)
            self.embed_dim = self.readout.readout_dim
        elif readout_function == "AR":
            self.readouts = dgl.mean_nodes
            self.embed_dim = feat_in_dim
        else:
            raise "Unknown Readout Function"

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1)

        #### DROpout Layer  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, batched_SP, node_feats_name="aggregated_feats", weights_feats_name="aggregated_weights"):
        if self.weigher_mode is not None:
            batched_SP.ndata['node_weights'] = self.weigher(batched_SP, weights_feats_name, node_feats_name)
            return self.dropout(self.LeakyReLU(self.readout(batched_SP, node_feats_name, "node_weights")))
        else:
            return self.dropout(self.LeakyReLU(self.readout(batched_SP, node_feats_name, None)))


class GATSPEmbedder(nn.Module):
    def __init__(self, feat_in_dim, readout_function="AR", jumping_mode="lstm", layers_num=3, residual_attn=True, exp_heads=False, weigher_mode=None, weights_in_dim=0, dropout=0):
        super(GATSPEmbedder, self).__init__()
        exp_heads = 1 if exp_heads else 0
        feat_dimensions_multipliers = [(2 ** (i * exp_heads + 1)) for i in range(layers_num)]

        self.JUMPING_MODE = jumping_mode

        ####  Layers  ####
        self.GAT_layers = nn.ModuleList()
        self.GAT_layers.append(GATConv(in_feats=feat_in_dim, out_feats=feat_in_dim, num_heads=feat_dimensions_multipliers[0], feat_drop=dropout, attn_drop=dropout, residual=residual_attn, bias=False))
        for i in range(1, layers_num):
            self.GAT_layers.append(GATConv(in_feats=feat_in_dim * feat_dimensions_multipliers[i - 1], out_feats=feat_in_dim, num_heads=feat_dimensions_multipliers[i], feat_drop=dropout, attn_drop=dropout, residual=residual_attn, bias=False))

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1)

        ####  Normalization Layers  ####
        self.normalizations = nn.ModuleList()
        for i in range(layers_num):
            self.normalizations.append(UnitedNormCommon(feat_in_dim * feat_dimensions_multipliers[i]))

        ####  Readouts Utils  ####
        self.weigher_mode = weigher_mode

        if self.weigher_mode is not None:
            self.weighers = nn.ModuleList()
            for i in range(layers_num):
                if self.weigher_mode == "sp_weights":
                    self.weighers.append(nn.Identity())
                elif self.weigher_mode == "attn_weights":
                    self.weighers.append(GATNodeWeigher(feat_in_dim, weights_in_dim, False))
                elif self.weigher_mode == "attn_weights+feats":
                    self.weighers.append(GATNodeWeigher(feat_in_dim, weights_in_dim, True))
                else:
                    raise "Wrong Weigher Mode"

        #### Readout Layers  ####
        if readout_function == "UR":
            self.readouts = nn.ModuleList()
            self.readouts.append(UniversalReadout(feat_in_dim))
            for i in range(1, layers_num):
                self.readouts.append(UniversalReadout(feat_in_dim))
        elif readout_function == "AR":
            self.readouts = []
            for i in range(layers_num):
                self.readouts.append(dgl.mean_nodes)
        else:
            raise "Unknown Readout Function"

        ####   JumpKnowledge Modules  ####
        if self.JUMPING_MODE == "lstm" or self.JUMPING_MODE == "max":
            if readout_function == "UR":
                self.embed_dim = self.readouts[0].readout_dim
            else:
                self.embed_dim = feat_in_dim
            self.jumping = JumpingKnowledge(mode=self.JUMPING_MODE, in_feats=self.embed_dim, num_layers=layers_num)
        elif self.JUMPING_MODE == "cat":
            self.jumping = JumpingKnowledge(mode=self.JUMPING_MODE)
            if readout_function == "UR":
                self.embed_dim = sum(r.readout_dim for r in self.readouts)
            else:
                self.embed_dim = feat_in_dim + sum([feat_in_dim for i in range(1, layers_num)])
        elif self.JUMPING_MODE is None:
            if readout_function == "UR":
                self.embed_dim = self.readout_list[0].readout_dim
            else:
                self.embed_dim = feat_in_dim

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

    def forward(self, spider_patch, node_feats):
        updated_feats = node_feats
        embeddings = []
        for idx, gat_layer in enumerate(self.GAT_layers):
            updated_feats = gat_layer(spider_patch, updated_feats).flatten(1)
            updated_feats = self.normalizations[idx](spider_patch, updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            updated_feats = self.dropout(updated_feats)
            spider_patch.ndata['updated_feats'] = torch.unflatten(updated_feats, -1, (gat_layer._num_heads, -1)).mean(1)
            if self.weigher_mode is not None:
                spider_patch.ndata['node_weights'] = self.weighers[idx](spider_patch, "aggregated_weights", "updated_feats")
                weights_name = 'node_weights'
            else:
                weights_name = None
            embeddings.append(self.readouts[idx](spider_patch, 'updated_feats', weights_name))
        return self.LeakyReLU(self.jumping(embeddings))

    def extractBestNormLayers(self):
        bests = []
        for idx, norm_layer in enumerate(self.normalizations):
            norm_powers = []
            for lambda_type in norm_layer.lambdas:
                norm_powers.append(torch.norm(lambda_type).detach().cpu().numpy())
            bests.append(norm_layer.norm_names[np.argmax(norm_powers)])
            tqdm.write(f"Best Norm for PATCH embedder normalizer {idx} is {bests[-1]}")
        return bests
