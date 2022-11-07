import warnings

import torch
from sklearn.decomposition import PCA

from CSIRS.CSIRS import CSIRSv2, CSIRSv2Arbitrary
from GeometricUtils import faceArea, intersectLineDiscreteSurface, LRF
import open3d as o3d
import dgl.data
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from Mesh import Mesh


class SuperPatch(dgl.DGLGraph):
    def __init__(self, sample_id, sample_mesh, seed_point, radius, n_rings, n_points):
        self.sample_id = sample_id
        self.concentricRings = []

        mesh_id = sample_mesh["level_0"][0]
        level_0 = Mesh()
        level_0.load(f"U:\AssegnoDiRicerca\MeshDataset\SHREC17\PatternDB\{mesh_id}.off")

        # Calculate ConcentricRing for resolution 0
        concentricRings = CSIRSv2Arbitrary(level_0, seed_point, radius, n_rings, n_points)
        if not concentricRings.firstValidRings(n_rings):
            raise RuntimeError("A ConcentricRings is not complete!")
        self.concentricRings.append(concentricRings)
        self.seed_point = concentricRings.seed_point

        # Calculate the LRF for the SuperPatch and the change basis matrix
        self.LRF = LRF(level_0, self.seed_point, radius)
        old_basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        new_basis = np.array([self.LRF[0], self.LRF[1], self.LRF[2]])
        mat_transform = np.linalg.solve(new_basis, old_basis)

        self.rings = len(concentricRings)
        self.points_per_ring = len(concentricRings[0])

        # Create the base graph structure (i.e the SpiderPatch structure)
        start_nodes = np.empty(0, dtype=int)
        end_nodes = np.empty(0, dtype=int)
        first_ring_nodes_id = np.empty(0, dtype=int)
        current_node = -1
        node_distances = np.empty(0)

        for ring in range(concentricRings.getRingsNumber()):
            for elem in range(len(concentricRings[ring])):
                # Generate the graph node (it is represented by edges)
                if any(~np.isnan(concentricRings[ring, elem])):
                    current_node += 1
                    next_elem_same_ring = concentricRings[ring][(elem + 1) % len(concentricRings[ring].points)]
                    if any(~np.isnan(next_elem_same_ring)) and ring == 0:  # if the next element in the ring is not NaN AND first ring
                        adjacent_node_id = (current_node + 1) % concentricRings[ring].getElementsNumber()
                        start_nodes = np.hstack((start_nodes, [current_node], [adjacent_node_id]))
                        end_nodes = np.hstack((end_nodes, [adjacent_node_id], [current_node]))
                        node_distances = np.hstack((node_distances, np.tile(0.75, 2)))  # np.hstack((node_distances, np.tile(np.linalg.norm(next_elem_same_ring - concentricRings[ring][elem]), 2)))
                        first_ring_nodes_id = np.hstack((first_ring_nodes_id, current_node))
                    else:
                        # Check if the next element in the ring is not NaN
                        if any(~np.isnan(next_elem_same_ring)):
                            if elem != len(concentricRings[ring]) - 1:  # mi serve per collegare l'ultimo nodo dell'anello al primo
                                adjacent_node_id = (current_node + 1)
                            else:
                                adjacent_node_id = (current_node + 1) - concentricRings[ring].getElementsNumber()
                            start_nodes = np.hstack((start_nodes, [current_node], [adjacent_node_id]))
                            end_nodes = np.hstack((end_nodes, [adjacent_node_id], [current_node]))
                            # do all'edge peso pari a 0.75 essendo un edge di collegamento tra due nodi sullo stesso anello
                            node_distances = np.hstack((node_distances, np.tile(0.75, 2)))  # np.hstack((node_distances, np.tile(np.linalg.norm(next_elem_same_ring - concentricRings[ring][elem]), 2)))

                    if ring < concentricRings.getRingsNumber() - 1:  # If not last ring
                        if any(~np.isnan(concentricRings[(ring + 1)][elem])):  # Controllo che il nodo nella stessa posizione del nodo corrente ma nell'anello successivo sia NON nan
                            outer_node_id = current_node + len(concentricRings[ring].getNonNan()[concentricRings[ring].getNonNan().index(elem):]) + len(concentricRings[ring + 1].getNonNan()[:concentricRings[ring + 1].getNonNan().index(
                                elem)])  # Sommo al nodo corrente gli elementi non nan rimanenti nel ring e gli elementi non nan che precedono il nodo nella stessa posizione del nodo corrente ma nell'anello successivo
                            start_nodes = np.hstack((start_nodes, [current_node], [outer_node_id]))
                            end_nodes = np.hstack((end_nodes, [outer_node_id], [current_node]))
                            # do all'edge peso pari a 1 essendo un edge di collegamento tra due nodi su anelli diversi
                            node_distances = np.hstack((node_distances, np.tile(1, 2)))  # np.hstack((node_distances, np.tile(np.linalg.norm(concentricRings[(ring + 1)][elem] - concentricRings[ring][elem]), 2)))

        # Add the center of the patch
        start_nodes = start_nodes + 1
        end_nodes = end_nodes + 1
        node_distances = np.hstack((node_distances, np.tile(1, len(first_ring_nodes_id) * 2)))  # np.hstack((node_distances, np.tile(np.linalg.norm(mesh.vertices[seed_point] - concentricRings[0][first_ring_nodes_id], axis=1), 2)))
        first_ring_nodes_id = np.array(first_ring_nodes_id) + 1
        start_nodes = np.hstack((start_nodes, [0] * len(first_ring_nodes_id), first_ring_nodes_id))
        end_nodes = np.hstack((end_nodes, first_ring_nodes_id, [0] * len(first_ring_nodes_id)))

        # Calculate NODE features for level_0 (i.e. initialize the structures)
        node_features = {}

        node_features["vertices"] = concentricRings.getNonNaNPoints()

        mesh_face_idx = concentricRings.getNonNaNFacesIdx()
        node_features["f_normal"] = level_0.face_normals[mesh_face_idx].dot(mat_transform.T)

        if not level_0.has_curvatures():
            level_0.computeCurvatures(radius)
        for curvature in ["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"]:
            node_features[curvature] = level_0.face_curvatures[2][curvature][mesh_face_idx].reshape((-1, 1))

        # Calculate EDGE features
        edge_features = {}
        edge_features["node_distance"] = node_distances

        # Calculate NODE features for other resolution levels
        del level_0

        for resolution_level in ["level_1", "level_2", "level_3"]:
            print(f"Calculating resolution {resolution_level} of SuperPatch...")
            mesh_id = sample_mesh[resolution_level][0]
            mesh = Mesh()
            mesh.load(f"U:\AssegnoDiRicerca\MeshDataset\SHREC17\PatternDB\{mesh_id}.off")

            # Calculate the concentric ring to extract the features
            concentricRings = CSIRSv2Arbitrary(mesh, seed_point, radius, n_rings, n_points)
            if not concentricRings.firstValidRings(n_rings):
                raise RuntimeError("A ConcentricRings is not complete!")
            self.concentricRings.append(concentricRings)
            mesh_face_idx = concentricRings.getNonNaNFacesIdx()
            f_normals = mesh.face_normals[mesh_face_idx].dot(mat_transform.T)
            node_features["f_normal"] = np.hstack((node_features["f_normal"], f_normals))

            if not mesh.has_curvatures():
                mesh.computeCurvatures(radius)
            for curvature in ["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"]:
                tmp_curvature = mesh.face_curvatures[2][curvature][mesh_face_idx].reshape((-1, 1))
                node_features[curvature] = np.hstack((node_features[curvature], tmp_curvature))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().__init__((start_nodes, end_nodes))

        for key, value in node_features.items():
            self.ndata[key] = torch.tensor(value, dtype=torch.float32)

        for key, value in edge_features.items():
            self.edata[key] = torch.tensor(value, dtype=torch.float32)

    def draw(self):
        o3d.visualization.draw_geometries(self.to_draw())

    def to_draw(self):
        points = o3d.utility.Vector3dVector(self.ndata["vertices"])
        edges = self.edges()
        lines = o3d.utility.Vector2iVector([[edges[0][i], edges[1][i]] for i in range(len(edges[0]))])

        line_set = o3d.geometry.LineSet()
        line_set.points = points
        line_set.lines = lines

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = points

        return [line_set, point_cloud]

    def getNodeFeatsNames(self):
        return list(self.ndata.keys())

    def getEdgeFeatsNames(self):
        return list(self.edata.keys())


class SuperPatch_preloaded(dgl.DGLGraph):
    def __init__(self, sample_id, sample_meshes, seed_point, radius, n_rings, n_points):
        self.sample_id = sample_id
        self.concentricRings = []
        level_0 = sample_meshes[0]
        radius = np.max(level_0.oriented_bounding_box["extent"]) * radius / 100
        # Calculate ConcentricRing for resolution 0
        concentricRings = CSIRSv2Arbitrary(level_0, seed_point, radius, n_rings, n_points)
        if concentricRings is None or not concentricRings.firstValidRings(n_rings):
            raise RuntimeError("The ConcentricRings at resolution 0 is not complete!")
        self.concentricRings.append(concentricRings)
        self.seed_point = concentricRings.seed_point

        # Calculate the LRF for the SuperPatch and the change basis matrix
        # self.LRF = LRF(level_0, self.seed_point, radius)
        # old_basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # new_basis = np.array([self.LRF[0], self.LRF[1], self.LRF[2]])
        # mat_transform = np.linalg.solve(new_basis, old_basis)

        self.rings = len(concentricRings)
        self.points_per_ring = len(concentricRings[0])

        # Create the base graph structure (i.e the SpiderPatch structure)
        start_nodes = np.empty(0, dtype=int)
        end_nodes = np.empty(0, dtype=int)
        first_ring_nodes_id = np.empty(0, dtype=int)
        current_node = -1
        node_distances = np.empty(0)

        for ring in range(concentricRings.getRingsNumber()):
            for elem in range(len(concentricRings[ring])):
                # Generate the graph node (it is represented by edges)
                if any(~np.isnan(concentricRings[ring, elem])):
                    current_node += 1
                    next_elem_same_ring = concentricRings[ring][(elem + 1) % len(concentricRings[ring].points)]
                    if any(~np.isnan(next_elem_same_ring)) and ring == 0:  # if the next element in the ring is not NaN AND first ring
                        adjacent_node_id = (current_node + 1) % concentricRings[ring].getElementsNumber()
                        start_nodes = np.hstack((start_nodes, [current_node], [adjacent_node_id]))
                        end_nodes = np.hstack((end_nodes, [adjacent_node_id], [current_node]))
                        node_distances = np.hstack((node_distances, np.tile(0.75, 2)))  # np.hstack((node_distances, np.tile(np.linalg.norm(next_elem_same_ring - concentricRings[ring][elem]), 2)))
                        first_ring_nodes_id = np.hstack((first_ring_nodes_id, current_node))
                    else:
                        # Check if the next element in the ring is not NaN
                        if any(~np.isnan(next_elem_same_ring)):
                            if elem != len(concentricRings[ring]) - 1:  # mi serve per collegare l'ultimo nodo dell'anello al primo
                                adjacent_node_id = (current_node + 1)
                            else:
                                adjacent_node_id = (current_node + 1) - concentricRings[ring].getElementsNumber()
                            start_nodes = np.hstack((start_nodes, [current_node], [adjacent_node_id]))
                            end_nodes = np.hstack((end_nodes, [adjacent_node_id], [current_node]))
                            # do all'edge peso pari a 0.75 essendo un edge di collegamento tra due nodi sullo stesso anello
                            node_distances = np.hstack((node_distances, np.tile(0.75, 2)))  # np.hstack((node_distances, np.tile(np.linalg.norm(next_elem_same_ring - concentricRings[ring][elem]), 2)))

                    if ring < concentricRings.getRingsNumber() - 1:  # If not last ring
                        if any(~np.isnan(concentricRings[(ring + 1)][elem])):  # Controllo che il nodo nella stessa posizione del nodo corrente ma nell'anello successivo sia NON nan
                            outer_node_id = current_node + len(concentricRings[ring].getNonNan()[concentricRings[ring].getNonNan().index(elem):]) + len(concentricRings[ring + 1].getNonNan()[:concentricRings[ring + 1].getNonNan().index(
                                elem)])  # Sommo al nodo corrente gli elementi non nan rimanenti nel ring e gli elementi non nan che precedono il nodo nella stessa posizione del nodo corrente ma nell'anello successivo
                            start_nodes = np.hstack((start_nodes, [current_node], [outer_node_id]))
                            end_nodes = np.hstack((end_nodes, [outer_node_id], [current_node]))
                            # do all'edge peso pari a 1 essendo un edge di collegamento tra due nodi su anelli diversi
                            node_distances = np.hstack((node_distances, np.tile(1, 2)))  # np.hstack((node_distances, np.tile(np.linalg.norm(concentricRings[(ring + 1)][elem] - concentricRings[ring][elem]), 2)))

        # Add the center of the patch
        start_nodes = start_nodes + 1
        end_nodes = end_nodes + 1
        node_distances = np.hstack((node_distances, np.tile(1, len(first_ring_nodes_id) * 2)))  # np.hstack((node_distances, np.tile(np.linalg.norm(mesh.vertices[seed_point] - concentricRings[0][first_ring_nodes_id], axis=1), 2)))
        first_ring_nodes_id = np.array(first_ring_nodes_id) + 1
        start_nodes = np.hstack((start_nodes, [0] * len(first_ring_nodes_id), first_ring_nodes_id))
        end_nodes = np.hstack((end_nodes, first_ring_nodes_id, [0] * len(first_ring_nodes_id)))

        # Calculate NODE features for level_0 (i.e. initialize the structures)
        node_features = {}

        node_features["vertices"] = concentricRings.getNonNaNPoints()

        mesh_face_idx = concentricRings.getNonNaNFacesIdx()
        # node_features["f_normal"] = level_0.face_normals[mesh_face_idx].dot(mat_transform.T)

        if not level_0.has_curvatures():
            raise RuntimeError("Mesh  doesn't have computed curvatures!")
        for curvature in ["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"]:
            node_features[curvature] = level_0.face_curvatures[0][curvature][mesh_face_idx].reshape((-1, 1))
            for radius_level in range(1, 5):
                node_features[curvature] = np.hstack((node_features[curvature], level_0.face_curvatures[radius_level][curvature][mesh_face_idx].reshape((-1, 1))))

        # Calculate EDGE features
        edge_features = {}
        edge_features["node_distance"] = node_distances

        # Calculate NODE features for other resolution levels
        del level_0

        for id, mesh in enumerate(sample_meshes[1:]):
            # Calculate the concentric ring to extract the features
            concentricRings = CSIRSv2Arbitrary(mesh, seed_point, radius, n_rings, n_points)
            if concentricRings is None or not concentricRings.firstValidRings(n_rings):
                raise RuntimeError(f"The ConcentricRings at resolution {id + 1} is not complete!")
            self.concentricRings.append(concentricRings)
            mesh_face_idx = concentricRings.getNonNaNFacesIdx()

            # f_normals = mesh.face_normals[mesh_face_idx].dot(mat_transform.T)
            # node_features["f_normal"] = np.hstack((node_features["f_normal"], f_normals))

            if not mesh.has_curvatures():
                raise RuntimeError("Mesh  doesn't have computed curvatures!")
            for curvature in ["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"]:
                for radius_level in range(5):
                    tmp_curvature = mesh.face_curvatures[radius_level][curvature][mesh_face_idx].reshape((-1, 1))
                    node_features[curvature] = np.hstack((node_features[curvature], tmp_curvature))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            super().__init__((start_nodes, end_nodes))

        for key, value in node_features.items():
            self.ndata[key] = torch.tensor(value, dtype=torch.float32)

        for key, value in edge_features.items():
            self.edata[key] = torch.tensor(value, dtype=torch.float32)

    def draw(self):
        o3d.visualization.draw_geometries(self.to_draw())

    def to_draw(self):
        points = o3d.utility.Vector3dVector(self.ndata["vertices"])
        edges = self.edges()
        lines = o3d.utility.Vector2iVector([[edges[0][i], edges[1][i]] for i in range(len(edges[0]))])

        line_set = o3d.geometry.LineSet()
        line_set.points = points
        line_set.lines = lines

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = points

        return [line_set, point_cloud]

    def getNodeFeatsNames(self):
        return list(self.ndata.keys())

    def getEdgeFeatsNames(self):
        return list(self.edata.keys())
