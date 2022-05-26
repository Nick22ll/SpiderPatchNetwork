import warnings
from copy import deepcopy

import dgl
import torch

from scipy.spatial import KDTree
import dgl.data
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class ReadoutMeshGraph(dgl.DGLGraph):
    def __init__(self, patches, feats_names="all"):

        if feats_names == "all":
            feats_names = patches[0].getNodeFeatsNames()

        start_nodes = np.empty(0, dtype=int)
        end_nodes = np.empty(0, dtype=int)
        seed_points = np.empty((0, 3))
        ndata_shape = 0
        for name in feats_names:
            ndata_shape += patches[0].ndata[name].shape[1]
        ndata = np.empty((0, ndata_shape))
        for i in range(len(patches)):
            patch = patches[i]
            # Aggrego le features dei nodi di ogni patch per poter fare il readout (lo faccio in local_scope coosì non rischio di modificare la patch)
            with patch.local_scope():
                patch.ndata["aggregated_feats"] = patch.ndata[feats_names[0]]
                for name in feats_names[1:]:
                    if patch.node_attr_schemes()[name].shape == ():
                        patch.ndata["aggregated_feats"] = torch.hstack((patch.ndata["aggregated_feats"], torch.reshape(patches[i].ndata[name], (-1, 1))))
                    else:
                        patch.ndata["aggregated_feats"] = torch.hstack((patch.ndata["aggregated_feats"], patches[i].ndata[name]))
                ndata = np.vstack((ndata, dgl.mean_nodes(patch, "aggregated_feats")))

            seed_points = np.vstack((seed_points, patch.seed_point))

        # Calculate the MeshGraphs edges (5 NearestNeighbours)
        neighbour_points = 6  # metto 6 perchè considero anche la distanza tra lo stesso punto quindi è ovvio che mi venga ritornato il punto stesso
        kdtree = KDTree(seed_points)
        _, points_idx = kdtree.query(seed_points, neighbour_points)
        start_nodes = np.hstack((start_nodes, [val for val in range(points_idx.shape[0]) for _ in range(neighbour_points - 1)]))
        end_nodes = np.hstack((end_nodes, points_idx[:, 1:].flatten()))

        # Makes the ReadoutMeshGraph Bidirectional
        tmp = np.array(start_nodes)
        start_nodes = np.hstack((start_nodes, end_nodes))
        end_nodes = np.hstack((end_nodes, tmp))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().__init__((start_nodes, end_nodes))

        self.ndata["patch_readout"] = torch.tensor(ndata, dtype=torch.float32)

    def getNodeFeatsNames(self):
        return list(self.ndata.keys())

    def getEdgeFeatsNames(self):
        return list(self.edata.keys())

    def draw(self):
        nx_G = self.to_networkx().to_undirected()
        # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
        pos = nx.kamada_kawai_layout(nx_G)
        nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
        plt.show()

