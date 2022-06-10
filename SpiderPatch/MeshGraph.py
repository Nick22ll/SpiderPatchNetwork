import warnings
from copy import deepcopy
import open3d as o3d
import torch

from scipy.spatial import KDTree
import dgl.data
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class MeshGraph(dgl.DGLGraph):
    def __init__(self, patches, feats_names="all", neighbours_number=0):
        """

        @param patches: a list of Patch representing nodes of the MeshGraph
        @param feats_names: list of names of features to extract from patches
        @param neighbours_number: number of Patches to consider neighbours (to generate the MeshGraph edges), leave "0" to make a fully connected graph
        """
        if feats_names == "all":
            feats_names = patches[0].getNodeFeatsNames()

        self.patches = []

        start_nodes = np.empty(0, dtype=int)
        end_nodes = np.empty(0, dtype=int)
        seed_points = np.empty((0, 3))

        for i in range(len(patches)):
            patch = patches[i]
            # Aggrego le features dei nodi di ogni patch per poter fare il readout (lo faccio in local_scope coosì non rischio di modificare la patch)
            with patch.local_scope():
                patch.ndata["aggregated_feats"] = patch.ndata[feats_names[1]]
                for name in feats_names[2:]:
                    if patch.node_attr_schemes()[name].shape == ():
                        patch.ndata["aggregated_feats"] = torch.hstack((patch.ndata["aggregated_feats"], torch.reshape(patch.ndata[name], (-1, 1))))
                    else:
                        patch.ndata["aggregated_feats"] = torch.hstack((patch.ndata["aggregated_feats"], patch.ndata[name]))

                edge_feats_names = patches[0].getEdgeFeatsNames()
                patch.edata["weights"] = patch.edata[edge_feats_names[0]]
                for name in edge_feats_names[1:]:
                    if patch.edge_attr_schemes()[name].shape == ():
                        patch.edata["weights"] = torch.hstack((patch.edata["weights"], torch.reshape(patch.edata[name], (-1, 1))))
                    else:
                        patch.edata["weights"] = torch.hstack((patch.edata["weights"], patch.edata[name]))

                self.patches.append(deepcopy(patch))
            seed_points = np.vstack((seed_points, patch.seed_point))

        # Calculate the MeshGraphs edges
        if neighbours_number > 0:
            neighbour_points = neighbours_number + 1
            kdtree = KDTree(seed_points)
            _, points_idx = kdtree.query(seed_points, neighbour_points)
            start_nodes = np.hstack((start_nodes, [val for val in range(points_idx.shape[0]) for _ in range(neighbour_points - 1)]))
            end_nodes = np.hstack((end_nodes, points_idx[:, 1:].flatten()))
        else:
            for i in range(len(patches)):
                start_nodes = np.hstack((start_nodes, np.tile(i, len(patches) - 1)))
                tmp_edges = list(range(len(patches)))
                tmp_edges.remove(i)
                end_nodes = np.hstack((end_nodes, tmp_edges))

        # Makes the ReadoutMeshGraph Bidirectional
        tmp = np.array(start_nodes)
        start_nodes = np.hstack((start_nodes, end_nodes))
        end_nodes = np.hstack((end_nodes, tmp))

        # Remove self-loops

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().__init__((start_nodes, end_nodes))

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

    def draw_on_mesh(self, mesh):
        # points = np.empty((0,3))
        # lines = np.empty((0,2), dtype=int)

        # for patch in self.patches:
        #     points = np.vstack((points, patch.ndata["vertices"]))
        #     edges = patch.edges()
        #     lines = np.vstack((lines, [[edges[0][i], edges[1][i]] for i in range(len(edges[0]))]))

        # points = o3d.utility.Vector3dVector(points)
        # lines = o3d.utility.Vector2iVector(lines)
        points = np.empty((0, 3))
        for patch in self.patches:
            points = np.vstack((points, patch.ndata["vertices"][0].detach().cpu().numpy()))

        points = o3d.utility.Vector3dVector(points)
        edges = self.edges()
        lines = o3d.utility.Vector2iVector([[edges[0][i], edges[1][i]] for i in range(len(edges[0]))])

        line_set = o3d.geometry.LineSet()
        line_set.points = points
        line_set.lines = lines

        line_set = o3d.geometry.LineSet()
        line_set.points = points
        line_set.lines = lines

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = points
        o3d.visualization.draw_geometries([mesh.mesh, point_cloud, lines], mesh_show_back_face=True)

    def to(self, device, **kwargs):

        self.patches = [self.patches[i].to(device) for i in range(len(self.patches))]
        return super().to(device, **kwargs)
