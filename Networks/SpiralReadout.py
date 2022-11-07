import os

import torch
import torch.nn as nn


class SpiralReadout(nn.Module):
    def __init__(self, readout_dim):
        super(SpiralReadout, self).__init__()
        self.readout_dim = readout_dim

    def forward(self, g, features_name):
        # Select the node features of seed_point as updated feats
        num_nodes = g.batch_num_nodes()
        spiral_readout = torch.empty((0, self.readout_dim), device=g.ndata[features_name][0].device)
        last_idx = 0
        for i in range(g.batch_size):
            unbatched_feats = g.ndata[features_name][last_idx: last_idx + num_nodes[i]].reshape((1, -1))
            spiral_readout = torch.vstack((spiral_readout, unbatched_feats))
            last_idx += num_nodes[i]
        return spiral_readout
