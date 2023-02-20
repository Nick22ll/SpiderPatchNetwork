import dgl
import numpy as np

import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, JumpingKnowledge
from tqdm import tqdm

from Networks.GATSpiderPatchModules import GATNodeWeigher
from Networks.NormalizationModules import UnitedNormCommon
from Networks.UniversalReadout import UniversalReadout


class CONVSPEmbedder(nn.Module):
    def __init__(self, feat_in_dim, readout_function="AR", jumping_mode="lstm", layers_num=3, weigher_mode=None, weights_in_dim=0, dropout=0, *args, **kwargs):
        super(CONVSPEmbedder, self).__init__()

        self.JUMPING_MODE = jumping_mode

        ####  Layers  ####
        self.convolutions = nn.ModuleList()
        for _ in range(layers_num):
            self.convolutions.append(GraphConv(feat_in_dim, feat_in_dim, bias=False))

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1)

        ####  Normalization Layers  ####
        self.normalizations = nn.ModuleList()
        for i in range(layers_num):
            self.normalizations.append(UnitedNormCommon(feat_in_dim))

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
                self.embed_dim = sum(r.readout_dim for r in self.readout_list)
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
        for idx, conv_layer in enumerate(self.convolutions):
            updated_feats = conv_layer(spider_patch, updated_feats)
            updated_feats = self.normalizations[idx](spider_patch, updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)
            updated_feats = self.dropout(updated_feats)
            spider_patch.ndata['updated_feats'] = updated_feats

            if self.weigher_mode is not None:
                spider_patch.ndata['node_weights'] = self.weighers[idx](spider_patch, "aggregated_weights", "updated_feats")
                weights_name = 'node_weights'
            else:
                weights_name = None

            embeddings.append(self.readouts[idx](spider_patch, 'updated_feats', weights_name))

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
            tqdm.write(f"Best Norm for PATCH embedder normalizer {idx} is {bests[-1]}")
        return bests


class JumpResSPEmbedder(nn.Module):
    def __init__(self, in_channels, layers_num, dropout):
        super(JumpResSPEmbedder, self).__init__()

        ####  Layers  ####
        # out_channels = in_channels * 4
        # self.chn_fixer = nn.Linear(in_channels, out_channels, bias=False)
        out_channels = in_channels
        self.conLayers = nn.ModuleList()
        for _ in range(layers_num):
            self.conLayers.append(GraphConv(out_channels, out_channels, bias=False))

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        #### Readout Layer ####
        self.readouts = nn.ModuleList()
        for _ in range(layers_num):
            self.readouts.append(UniversalReadout(out_channels, dropout))

        ####  Normalization Layers  ####
        self.normalizations = nn.ModuleList()
        for _ in range(layers_num):
            self.normalizations.append(GraphNorm(out_channels, eps=1e-5))

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = self.readouts[0].readout_dim * layers_num

    def forward(self, g, node_feats):
        # node_feats = self.chn_fixer(node_feats)
        updated_feats = self.conLayers[0](g, node_feats)
        updated_feats = self.normalizations[0](updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = self.readouts[0](g, 'updated_feats')

        prev0 = torch.clone(updated_feats)
        updated_feats = updated_feats + node_feats
        for idx, conv_layer in enumerate(self.conLayers[1:], start=1):
            updated_feats = conv_layer(g, updated_feats)
            updated_feats = self.normalizations[idx](updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)

            prev1 = torch.clone(updated_feats)
            updated_feats = updated_feats + prev0
            prev0 = torch.clone(prev1)

            with g.local_scope():
                g.ndata['updated_feats'] = updated_feats
                readout = torch.hstack((readout, self.readouts[idx](g, 'updated_feats')))

        return self.LeakyReLU(readout)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class SPEmbedder3ConvUNIVERSAL(nn.Module):
    def __init__(self, in_feats, hidden_dim, dropout):
        super(SPEmbedder3ConvUNIVERSAL, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_feats, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, bias=False)
        self.conv3 = GraphConv(hidden_dim, hidden_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        #### Readout Layer ####
        self.readout1 = UniversalReadout(hidden_dim, dropout)
        self.readout2 = UniversalReadout(hidden_dim, dropout)
        self.readout3 = UniversalReadout(hidden_dim, dropout)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv3 = GraphNorm(hidden_dim, eps=1e-5)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv3.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = self.readout1.readout_dim + self.readout2.readout_dim + self.readout3.readout_dim

    def forward(self, g, node_feats):
        updated_feats = self.conv1(g, node_feats)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = self.readout1(g, 'updated_feats')

        updated_feats = self.conv2(g, updated_feats)
        updated_feats = self.GraphNormConv2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, self.readout2(g, 'updated_feats')))

        updated_feats = self.conv3(g, updated_feats)
        updated_feats = self.GraphNormConv3(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, self.readout3(g, 'updated_feats')))

        return self.LeakyReLU(readout)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class WEIGHTSPEmbedderAR(nn.Module):
    def __init__(self, in_channels, layers_num, dropout, nodes_number):
        super(WEIGHTSPEmbedderAR, self).__init__()

        self.AR_weights = nn.Parameter(torch.ones((layers_num, nodes_number), requires_grad=True))

        ####  Layers  ####
        # out_channels = in_channels * 4
        # self.chn_fixer = nn.Linear(in_channels, out_channels, bias=False)
        out_channels = in_channels
        self.conLayers = nn.ModuleList()
        for _ in range(layers_num):
            self.conLayers.append(GraphConv(out_channels, out_channels, bias=False))

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.normalizations = nn.ModuleList()
        for _ in range(layers_num):
            self.normalizations.append(GraphNorm(out_channels, eps=1e-5))

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = out_channels * (layers_num + 1)

        self.embed_dim = out_channels * layers_num
        self.nodes_number = nodes_number

    def forward(self, g, node_feats):
        # node_feats = self.chn_fixer(node_feats)
        updated_feats = self.conLayers[0](g, node_feats)
        updated_feats = self.normalizations[0](updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            g.ndata["node_weights"] = (g.ndata["weights"].view(-1, self.nodes_number) * self.AR_weights[0][None, :]).view((-1, 1))
            readout = dgl.mean_nodes(g, "updated_feats", "node_weights")

        prev0 = torch.clone(updated_feats)
        updated_feats = updated_feats + node_feats
        for idx, conv_layer in enumerate(self.conLayers[1:], start=1):
            updated_feats = conv_layer(g, updated_feats)
            updated_feats = self.normalizations[idx](updated_feats)
            updated_feats = self.LeakyReLU(updated_feats)

            prev1 = torch.clone(updated_feats)
            updated_feats = updated_feats + prev0
            prev0 = torch.clone(prev1)

            with g.local_scope():
                g.ndata['updated_feats'] = updated_feats
                g.ndata["node_weights"] = (g.ndata["weights"].view(-1, self.nodes_number) * self.AR_weights[idx][None, :]).view((-1, 1))
                readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats", "node_weights")))

        return self.LeakyReLU(readout)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class WEIGHTSPEmbedder3Conv(nn.Module):
    def __init__(self, in_feats, hidden_dim, dropout, nodes_number, edges_number):
        super(WEIGHTSPEmbedder3Conv, self).__init__()

        ####  Weights Parameters  ####
        self.suppl_weights_conv1 = nn.Parameter(torch.ones(edges_number, requires_grad=True))
        self.suppl_weights_conv2 = nn.Parameter(torch.ones(edges_number, requires_grad=True))
        self.suppl_weights_conv3 = nn.Parameter(torch.ones(edges_number, requires_grad=True))

        self.AR1_weights = nn.Parameter(torch.ones(nodes_number, requires_grad=True))
        self.AR2_weights = nn.Parameter(torch.ones(nodes_number, requires_grad=True))
        self.AR3_weights = nn.Parameter(torch.ones(nodes_number, requires_grad=True))
        self.AR4_weights = nn.Parameter(torch.ones(nodes_number, requires_grad=True))

        ####  Layers  ####
        self.conv1 = GraphConv(in_feats, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, bias=False)
        self.conv3 = GraphConv(hidden_dim, hidden_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        #### Readout Layer ####
        self.readout1 = UniversalReadout(hidden_dim, dropout)
        self.readout2 = UniversalReadout(hidden_dim, dropout)
        self.readout3 = UniversalReadout(hidden_dim, dropout)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv3 = GraphNorm(hidden_dim, eps=1e-5)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv3.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = in_feats + (hidden_dim * 3) + self.readout1.readout_dim + self.readout2.readout_dim + self.readout3.readout_dim
        self.nodes_number = nodes_number
        self.edges_number = edges_number

    def forward(self, g, node_feats, edge_weights):
        with g.local_scope():
            g.ndata['updated_feats'] = node_feats
            g.ndata["node_weights"] = (g.ndata["weights"].view(-1, self.nodes_number) * self.AR1_weights[:, None]).view((-1, 1))
            readout = dgl.mean_nodes(g, "updated_feats", "node_weights")

        ew = edge_weights * torch.tile(self.suppl_weights_conv1, (g.batch_size,))

        updated_feats = self.conv1(g, node_feats, edge_weight=ew)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, self.readout1(g, 'updated_feats')))
            g.ndata["node_weights"] = (g.ndata["weights"] * torch.tile(self.AR2_weights, (g.batch_size,))).view((-1, 1))
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats", "node_weights")))

        ew = edge_weights * torch.tile(self.suppl_weights_conv2, (g.batch_size,))

        updated_feats = self.conv2(g, updated_feats, edge_weight=ew)
        updated_feats = self.GraphNormConv2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, self.readout2(g, 'updated_feats')))
            g.ndata["node_weights"] = (g.ndata["weights"] * torch.tile(self.AR3_weights, (g.batch_size,))).view((-1, 1))
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats", "node_weights")))

        ew = edge_weights * torch.tile(self.suppl_weights_conv3, (g.batch_size,))

        updated_feats = self.conv3(g, updated_feats, edge_weight=ew)
        updated_feats = self.GraphNormConv3(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, self.readout3(g, 'updated_feats')))
            g.ndata["node_weights"] = (g.ndata["weights"] * torch.tile(self.AR4_weights, (g.batch_size,))).view((-1, 1))
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats", "node_weights")))

        return self.LeakyReLU(readout)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class READOUTWEIGHTSPEmbedder3Conv(nn.Module):
    def __init__(self, in_feats, hidden_dim, dropout, nodes_number, edges_number):
        super(READOUTWEIGHTSPEmbedder3Conv, self).__init__()

        ####  Weights Parameters  ####

        self.AR1_weights = nn.Parameter(torch.ones((1, nodes_number), requires_grad=True))
        self.AR2_weights = nn.Parameter(torch.ones((1, nodes_number), requires_grad=True))
        self.AR3_weights = nn.Parameter(torch.ones((1, nodes_number), requires_grad=True))
        self.AR4_weights = nn.Parameter(torch.ones((1, nodes_number), requires_grad=True))

        ####  Layers  ####
        self.conv1 = GraphConv(in_feats, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, bias=False)
        self.conv3 = GraphConv(hidden_dim, hidden_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        #### Readout Layer ####
        self.readout1 = UniversalReadout(hidden_dim, dropout)
        self.readout2 = UniversalReadout(hidden_dim, dropout)
        self.readout3 = UniversalReadout(hidden_dim, dropout)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv3 = GraphNorm(hidden_dim, eps=1e-5)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv3.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = in_feats + (hidden_dim * 3) + self.readout1.readout_dim + self.readout2.readout_dim + self.readout3.readout_dim
        self.nodes_number = nodes_number
        self.edges_number = edges_number

    def forward(self, g, node_feats):
        with g.local_scope():
            g.ndata['updated_feats'] = node_feats
            g.ndata["node_weights"] = (g.ndata["weights"].view(-1, self.nodes_number) * self.AR1_weights).view((-1, 1))
            readout = dgl.mean_nodes(g, "updated_feats", "node_weights")

        updated_feats = self.conv1(g, node_feats)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, self.readout1(g, 'updated_feats')))
            g.ndata["node_weights"] = (g.ndata["weights"].view(-1, self.nodes_number) * self.AR2_weights).view((-1, 1))
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats", "node_weights")))

        updated_feats = self.conv2(g, updated_feats)
        updated_feats = self.GraphNormConv2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, self.readout2(g, 'updated_feats')))
            g.ndata["node_weights"] = (g.ndata["weights"].view(-1, self.nodes_number) * self.AR3_weights).view((-1, 1))
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats", "node_weights")))

        updated_feats = self.conv3(g, updated_feats)
        updated_feats = self.GraphNormConv3(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, self.readout3(g, 'updated_feats')))
            g.ndata["node_weights"] = (g.ndata["weights"].view(-1, self.nodes_number) * self.AR4_weights).view((-1, 1))
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats", "node_weights")))

        return self.LeakyReLU(readout)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class SPEmbedder2Conv(nn.Module):
    def __init__(self, in_feats, hidden_dim, dropout):
        super(SPEmbedder2Conv, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_feats, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(hidden_dim, eps=1e-5)

        #### Readout Layer ####
        self.readout1 = UniversalReadout(hidden_dim, dropout)
        self.readout2 = UniversalReadout(hidden_dim, dropout)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = (hidden_dim * 2) + self.readout1.readout_dim + self.readout2.readout_dim

    def forward(self, g, node_feats, edge_weights):
        updated_feats = self.conv1(g, node_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        updated_feats = self.dropout(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = self.readout1(g, 'updated_feats')
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats")))

        updated_feats = self.conv2(g, updated_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        updated_feats = self.dropout(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, self.readout2(g, 'updated_feats')))
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats")))

        return self.LeakyReLU(readout)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class SPEmbedder3Conv(nn.Module):
    def __init__(self, in_feats, hidden_dim, dropout):
        super(SPEmbedder3Conv, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_feats, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, bias=False)
        self.conv3 = GraphConv(hidden_dim, hidden_dim, bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        #### Readout Layer ####
        self.readout1 = UniversalReadout(hidden_dim, dropout)
        self.readout2 = UniversalReadout(hidden_dim, dropout)
        self.readout3 = UniversalReadout(hidden_dim, dropout)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv3 = GraphNorm(hidden_dim, eps=1e-5)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv3.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = in_feats + (hidden_dim * 3) + self.readout1.readout_dim + self.readout2.readout_dim + self.readout3.readout_dim

    def forward(self, g, node_feats, edge_weights):
        with g.local_scope():
            g.ndata['updated_feats'] = node_feats
            readout = dgl.mean_nodes(g, "updated_feats")

        updated_feats = self.conv1(g, node_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, self.readout1(g, 'updated_feats')))
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats")))

        updated_feats = self.conv2(g, updated_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, self.readout2(g, 'updated_feats')))
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats")))

        updated_feats = self.conv3(g, updated_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv3(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, self.readout3(g, 'updated_feats')))
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats")))

        return self.LeakyReLU(readout)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class SPEmbedder3ConvAR(nn.Module):
    def __init__(self, in_feats, hidden_dim, dropout):
        super(SPEmbedder3ConvAR, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_feats, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, bias=False)
        self.conv3 = GraphConv(hidden_dim, int(hidden_dim / 4), bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv3 = GraphNorm(int(hidden_dim / 4), eps=1e-5)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv3.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = in_feats + int(hidden_dim / 4) + (hidden_dim * 2)

    def forward(self, g, node_feats, edge_weights):
        with g.local_scope():
            g.ndata['updated_feats'] = node_feats
            readout = dgl.mean_nodes(g, "updated_feats", "weight")

        updated_feats = self.conv1(g, node_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats", "weight")))

        updated_feats = self.conv2(g, updated_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats", "weight")))

        updated_feats = self.conv3(g, updated_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv3(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats", "weight")))

        return self.LeakyReLU(readout)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class SPEmbedder2ConvAR(nn.Module):
    def __init__(self, in_feats, hidden_dim, dropout):
        super(SPEmbedder2ConvAR, self).__init__()

        ####  Layers  ####
        self.conv1 = GraphConv(in_feats, hidden_dim, bias=False)
        self.conv2 = GraphConv(hidden_dim, int(hidden_dim / 4), bias=False)

        ####  Activation Functions  ####
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01)

        ####  Normalization Layers  ####
        self.GraphNormConv1 = GraphNorm(hidden_dim, eps=1e-5)
        self.GraphNormConv2 = GraphNorm(int(hidden_dim / 4), eps=1e-5)

        ####  Weights Initialization  ####
        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=self.LeakyReLU.negative_slope, mode='fan_in', nonlinearity='leaky_relu')

        ####  Dropout  ####
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = in_feats + int(hidden_dim / 4) + hidden_dim

    def forward(self, g, node_feats, edge_weights):
        with g.local_scope():
            g.ndata['updated_feats'] = node_feats
            readout = dgl.mean_nodes(g, "updated_feats", "weight")

        updated_feats = self.conv1(g, node_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv1(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats", "weight")))

        updated_feats = self.conv2(g, updated_feats, edge_weight=edge_weights)
        updated_feats = self.GraphNormConv2(updated_feats)
        updated_feats = self.LeakyReLU(updated_feats)

        with g.local_scope():
            g.ndata['updated_feats'] = updated_feats
            readout = torch.hstack((readout, dgl.mean_nodes(g, "updated_feats", "weight")))

        return self.LeakyReLU(readout)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
