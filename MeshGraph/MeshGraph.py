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
        @param neighbours_number: number of Patches to consider neighbours (to generate the MeshGraph edges), leave "0" to make a fully connected graph
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


class SpiralMeshGraph(dgl.DGLGraph):
    def __init__(self, mesh, concRings, neighbours_number=0, sample_id=None, mesh_id=None, resolution_level=None):
        """

        @param concRings: a list of ConcRings representing nodes of the SpiralMeshGraph
        @param keep_feats_names: list of names of features to extract from patches
        @param neighbours_number: number of Patches to consider neighbours (to generate the MeshGraph edges), leave "0" to make a fully connected graph
        """

        self.sample_id = sample_id
        self.mesh_id = mesh_id
        self.resolution_level = resolution_level
        start_nodes = np.empty(0, dtype=int)
        end_nodes = np.empty(0, dtype=int)
        seed_points = np.empty((0, 3))

        for ring in concRings:
            seed_points = np.vstack((seed_points, ring.seed_point))

        # Calculate the MeshGraphs edges
        if neighbours_number > 0:
            neighbour_points = neighbours_number + 1
            kdtree = KDTree(seed_points)
            _, points_idx = kdtree.query(seed_points, neighbour_points)
            start_nodes = np.hstack((start_nodes, [val for val in range(points_idx.shape[0]) for _ in range(neighbour_points - 1)]))
            end_nodes = np.hstack((end_nodes, points_idx[:, 1:].flatten()))
        else:
            for i in range(len(concRings)):
                start_nodes = np.hstack((start_nodes, np.tile(i, len(concRings) - 1)))
                tmp_edges = list(range(len(concRings)))
                tmp_edges.remove(i)
                end_nodes = np.hstack((end_nodes, tmp_edges))

        # Makes the MeshGraph Bidirectional
        tmp = np.array(start_nodes)
        start_nodes = np.hstack((start_nodes, end_nodes))
        end_nodes = np.hstack((end_nodes, tmp))

        # Calculate NODE features
        node_features = {}

        node_features["vertices"] = concRings[0].getNonNaNPoints().flatten()
        mesh_face_idx = concRings[0].getNonNaNFacesIdx()

        # node_features["f_normal"] = mesh.face_normals[mesh_face_idx]

        if not mesh.has_curvatures():
            raise RuntimeError("Mesh  doesn't have computed curvatures!")
        for curvature in ["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"]:
            node_features[curvature] = mesh.face_curvatures[0][curvature][mesh_face_idx].reshape((-1, 1))
            for radius_level in range(1, 5):
                tmp_curvature = mesh.face_curvatures[radius_level][curvature][mesh_face_idx].reshape((-1, 1))
                node_features[curvature] = np.hstack((node_features[curvature], tmp_curvature))
            node_features[curvature] = node_features[curvature].flatten()

        for concRing in concRings[1:]:
            node_features["vertices"] = np.vstack((node_features["vertices"], concRing.getNonNaNPoints().flatten()))
            mesh_face_idx = concRing.getNonNaNFacesIdx()

            # node_features["f_normal"] = mesh.face_normals[mesh_face_idx]

            if not mesh.has_curvatures():
                raise RuntimeError("Mesh  doesn't have computed curvatures!")
            for curvature in ["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"]:
                tmp_curvature = mesh.face_curvatures[0][curvature][mesh_face_idx].reshape((-1, 1))
                for radius_level in range(1, 5):
                    tmp_curvature = np.hstack((tmp_curvature, mesh.face_curvatures[radius_level][curvature][mesh_face_idx].reshape((-1, 1))))
                node_features[curvature] = np.vstack((node_features[curvature], tmp_curvature.flatten()))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().__init__((start_nodes, end_nodes))

        for key, value in node_features.items():
            self.ndata[key] = torch.tensor(value, dtype=torch.float32)

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
        return super().to(device, **kwargs)
