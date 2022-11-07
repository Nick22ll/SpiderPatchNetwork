import warnings
import open3d as o3d
import torch

from scipy.spatial import KDTree
import dgl.data
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import pairwise_distances
from CSIRS.CSIRS import CSIRSv2Spiral, SpiralCSIRSArbitrary


class SpiralMeshCSI(dgl.DGLGraph):
    def __init__(self, mesh, seed_point, radius, rings, points, node_radius, node_rings, node_points, neighbours_number=0, sample_id=None, mesh_id=None, resolution_level=None):

        principalConcentricRings = CSIRSv2Spiral(mesh, seed_point, radius, rings, points)
        if not principalConcentricRings.firstValidRings(rings):
            raise RuntimeError("The SpiralMesh is NOT completed!")
        if np.any(pairwise_distances(principalConcentricRings[-1][0].reshape((1, -1)), principalConcentricRings[-1][int(points / 2)].reshape((1, -1))) < radius * 1.60):
            raise RuntimeError("The SpiralMesh is NOT good!")
        concRings = []
        face_indices = principalConcentricRings.getNonNaNFacesIdx()
        for idx, node_seed_point in enumerate(principalConcentricRings.getNonNaNPoints()):
            concRings.append(SpiralCSIRSArbitrary(mesh, node_seed_point, node_radius, node_rings, node_points, inFace=face_indices[idx]))
            if not concRings[-1].firstValidRings(node_rings):
                raise RuntimeError("A concentric ring is NOT completed!")
        mesh.drawWithConcRings(principalConcentricRings)
        self.sample_id = sample_id
        self.mesh_id = mesh_id
        self.resolution_level = resolution_level
        self.principalConcentricRings = principalConcentricRings
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
            node_features[curvature] = mesh.face_curvatures[0][curvature][mesh_face_idx].flatten()
            for radius_level in range(1, 5):
                tmp_curvature = mesh.face_curvatures[radius_level][curvature][mesh_face_idx].flatten()
                node_features[curvature] = np.hstack((node_features[curvature], tmp_curvature))
            node_features[curvature] = node_features[curvature].flatten()

        for concRing in concRings[1:]:
            node_features["vertices"] = np.vstack((node_features["vertices"], concRing.getNonNaNPoints().flatten()))
            mesh_face_idx = concRing.getNonNaNFacesIdx()

            # node_features["f_normal"] = mesh.face_normals[mesh_face_idx]

            if not mesh.has_curvatures():
                raise RuntimeError("Mesh  doesn't have computed curvatures!")
            for curvature in ["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"]:
                tmp_curvature = mesh.face_curvatures[0][curvature][mesh_face_idx].flatten()
                for radius_level in range(1, 5):
                    tmp_curvature = np.hstack((tmp_curvature, mesh.face_curvatures[radius_level][curvature][mesh_face_idx].flatten()))
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

    def to(self, device, **kwargs):
        return super().to(device, **kwargs)
