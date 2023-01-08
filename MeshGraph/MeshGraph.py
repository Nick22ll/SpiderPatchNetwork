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
    def __init__(self, patches, neighbours_number=0, sample_id=None, mesh_id=None, resolution_level=None, keep_feats_names=None):
        """

        @param patches: a list of Patch representing nodes of the MeshGraph
        @param keep_feats_names: list of names of features to extract from patches
        @param neighbours_number: number of SpiderPatches to consider neighbours (to generate the MeshGraph edges), leave "0" to make a fully connected graph
        """

        if keep_feats_names is None:
            keep_feats_names = patches[0].getNodeFeatsNames()

        self.patches = []
        self.sample_id = sample_id
        self.mesh_id = mesh_id
        self.resolution_level = resolution_level
        start_nodes = np.empty(0, dtype=int)
        end_nodes = np.empty(0, dtype=int)
        seed_points = np.empty((0, 3))

        feats_names_toremove = patches[0].getNodeFeatsNames()
        for name in keep_feats_names:
            feats_names_toremove.remove(name)

        for patch in patches:
            self.patches.append(deepcopy(patch))
            for name in feats_names_toremove:
                self.patches[-1].ndata.pop(name)
            seed_points = np.vstack((seed_points, patch.seed_point))

        self.patches = np.array(self.patches)

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

        # Makes the MeshGraph Bidirectional
        tmp = np.array(start_nodes)
        start_nodes = np.hstack((start_nodes, end_nodes))
        end_nodes = np.hstack((end_nodes, tmp))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().__init__((start_nodes, end_nodes))

        #  Calculates Edge Weights
        weights = []
        for edge_id in range(int(len(self.edges()[0]) / 2)):
            start = self.patches[self.edges()[0][edge_id]].seed_point
            end = self.patches[self.edges()[1][edge_id]].seed_point
            distance = np.linalg.norm(end - start)
            weights.append(1 - (1 / distance))
        weights = np.concatenate((weights, weights))
        self.edata["weights"] = torch.tensor(weights, dtype=torch.float32)

    def getNodeFeatsNames(self):
        return np.array(self.ndata.keys(), dtype=object)

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

        self.patches = np.array([self.patches[i].to(device) for i in range(len(self.patches))])
        return super().to(device, **kwargs)
