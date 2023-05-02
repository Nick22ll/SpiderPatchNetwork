import gc
import os

import dgl
import numpy as np
import scipy.io

from CSIRS.CSIRS import CSIRSv2Spiral
from Mesh.Mesh import Mesh
from SpiderDatasets.SpiderPatchDataset import SpiderPatchDataset

from SpiderPatch.SpiderPatch import SpiderPatch
import warnings
import pickle as pkl

from tqdm import tqdm


class RetrievalDataset(SpiderPatchDataset):
    def __init__(self, dataset_name="", spiral_spider_patches=None, labels=None, mesh_id=None):
        self.mesh_id = mesh_id
        super().__init__(dataset_name=dataset_name, graphs=spiral_spider_patches, labels=labels)

    def generate(self, mesh_path, labels_path, radius, rings, points, relative_radius, CSIRS_type=CSIRSv2Spiral):
        warnings.filterwarnings("ignore")
        with open(mesh_path, "rb") as mesh_file:
            mesh = pkl.load(mesh_file)

        with open(labels_path, "rb") as labels_file:
            vertex_labels = pkl.load(labels_file)

        if relative_radius:
            radius = radius * mesh.edge_length
        boundary_vertices = mesh.getBoundaryVertices(neighbors_level=int(np.ceil(radius / mesh.edge_length)))
        rng = np.random.default_rng(17)
        seed_point_sequence = [i for i in range(len(mesh.vertices))]
        rng.shuffle(seed_point_sequence)
        self.graphs = np.empty(0)
        self.seed_point_indices = np.empty(0)
        self.labels = np.empty(0)
        for seed_point in tqdm(seed_point_sequence):
            if seed_point in boundary_vertices:
                continue
            concentric_ring = CSIRS_type(mesh, seed_point, radius, rings, points)
            if not concentric_ring.firstValidRings(2):
                continue
            try:
                self.graphs = np.append(self.graphs, SpiderPatch(concentric_ring, mesh, seed_point))
            except dgl.DGLError:
                continue
            self.labels = np.append(self.labels, vertex_labels[seed_point])
            self.seed_point_indices = np.append(self.seed_point_indices, seed_point)

        self.save_to(f"../Retrieval/Datasets")

    def save_to(self, save_path=None):
        os.makedirs(save_path, exist_ok=True)
        if save_path is not None:
            path = f"{save_path}/{self.name}.pkl"
        else:
            path = f"{self.save_path}/{self.name}.pkl"
        with open(path, "wb") as dataset_file:
            pkl.dump(self, dataset_file, protocol=-1)

    def load_from(self, load_path=None, dataset_name=None):
        """
        Load a RetrievalDataset instance from a pkl file.
        @param load_path: (string) the path to the folder containing the dataset structure folder e.g. "Datasets/MeshGraphs/SHREC17_R10_R16_P6_Normalized".
        @param dataset_name: (string) the name of the dataset file e.g. "SHREC17_R10_R16_P6_level_0_Normalized_test"
        @return:
        """
        if load_path is not None:
            path = f"{load_path}/{dataset_name}.pkl"
        else:
            path = f"{self.save_path}/{dataset_name}.pkl"

        gc.disable()
        with open(path, "rb") as dataset_file:
            loaded_dataset = pkl.load(dataset_file)
        gc.enable()
        self.graphs = loaded_dataset.graphs
        self.labels = loaded_dataset.labels
        self.seed_point_indices = loaded_dataset.seed_point_indices
        self.mesh_id = loaded_dataset.mesh_id
        self._name = dataset_name


def generateMesh(path, name):
    mesh = Mesh()
    mesh.loadFromMeshFile(path)
    for i in tqdm(range(5)):
        mesh.computeCurvatures(i)
    os.makedirs(f"../Retrieval/Meshes", exist_ok=True)
    mesh.save(f"../Retrieval/Meshes/{name}.pkl")


def generateLabels(mesh_name, labels_path):
    mat = scipy.io.loadmat(labels_path)
    face_labels = mat["label"].flatten()
    with open(f"../Retrieval/Meshes/{mesh_name}.pkl", "rb") as mesh_file:
        mesh = pkl.load(mesh_file)
    vertex_labels = []
    for face_list in mesh.vertex_faces:
        vertex_labels.append(np.argmax(np.bincount(face_labels[face_list])))
    for id, elem in enumerate(np.unique(vertex_labels)):
        vertex_labels = [id if x == elem else x for x in vertex_labels]
    os.makedirs(f"../Retrieval/Labels", exist_ok=True)
    with open(f"../Retrieval/Labels/{mesh_name}.pkl", "wb") as label_file:
        pkl.dump(vertex_labels, label_file)
