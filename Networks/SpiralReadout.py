import os

import dgl
import torch
import torch.nn as nn


class SpiralReadout(nn.Module):
    def __init__(self, readout_dim):
        super(SpiralReadout, self).__init__()
        self.readout_dim = readout_dim

    def forward(self, batched_graph, features_name, weights_names):
        # Select the node features of seed_point as updated feats
        spiral_readout = torch.zeros((batched_graph.batch_size, self.readout_dim), device=batched_graph.ndata[features_name][0].device)
        if weights_names:
            for i, graph in enumerate(dgl.unbatch(batched_graph)):
                spiral_readout[i] = graph.ndata[features_name].reshape((1, -1)) * graph.ndata[weights_names]
        else:
            for i, graph in enumerate(dgl.unbatch(batched_graph)):
                spiral_readout[i] = graph.ndata[features_name].reshape((1, -1))
        return spiral_readout
