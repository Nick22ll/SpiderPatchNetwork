import gc
import random

import scipy.io
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

from CSIRS.CSIRS import CSIRSv2Spiral
from Mesh.Mesh import Mesh
from SpiderDatasets.SpiderPatchDataset import SpiderPatchDataset

from SpiderPatch.SpiderPatch import SpiderPatch
import warnings
import pickle as pkl
import numpy as np

from Networks.CONVNetworks import *
from tqdm import tqdm


class RetrievalDataset(SpiderPatchDataset):
    def __init__(self, dataset_name="", spiral_spider_patches=None, labels=None, mesh_id=None):
        self.mesh_id = mesh_id
        super().__init__(dataset_name=dataset_name, graphs=spiral_spider_patches, labels=labels)

    def generate(self, mesh_path, labels_path, radius, rings, points, relative_radius):
        warnings.filterwarnings("ignore")
        with open(mesh_path, "rb") as mesh_file:
            mesh = pkl.load(mesh_file)

        with open(labels_path, "rb") as labels_file:
            vertex_labels = pkl.load(labels_file)

        if relative_radius:
            radius = radius * mesh.edge_length
        boundary_vertices = mesh.getBoundaryVertices(neighbors_level=int(np.ceil(1.5 * radius)))
        random.seed(222)
        seed_point_sequence = [i for i in range(len(mesh.vertices))]
        random.shuffle(seed_point_sequence)
        self.graphs = np.empty(0)
        self.seed_point_indices = np.empty(0)
        self.labels = np.empty(0)
        for seed_point in tqdm(seed_point_sequence):
            if seed_point in boundary_vertices:
                continue
            concentric_ring = CSIRSv2Spiral(mesh, seed_point, radius, rings, points)
            if not concentric_ring.firstValidRings(1):
                continue
            try:
                self.graphs = np.append(self.graphs, SpiderPatch(concentric_ring, mesh, seed_point))
            except dgl.DGLError:
                continue
            self.labels = np.append(self.labels, vertex_labels[seed_point])
            self.seed_point_indices = np.append(self.seed_point_indices, seed_point)

        self.save_to(f"../Retrieval/Datasets")

    def getTrainTestMask(self, train_samples=200, percentage=False):
        random.seed(22)
        train_indices = []
        test_indices = []
        for label in np.unique(self.labels):
            label_indices = np.where(self.labels == label)[0]
            if percentage:
                train_label_indices = np.random.choice(label_indices, int((train_samples / 100) * len(label_indices)), replace=False)
            else:
                train_label_indices = np.random.choice(label_indices, train_samples, replace=False)
            test_indices.extend(list(np.delete(label_indices, np.argwhere(np.isin(label_indices, train_label_indices)))))
            train_indices.extend(list(train_label_indices))
        return train_indices, test_indices

    def normalize(self, fit_indices):
        feats_names = self.getNodeFeatsName()
        if "vertices" in feats_names:
            feats_names.remove("vertices")
        if "weight" in feats_names:
            feats_names.remove("weight")

        node_normalizers = {}
        for feature in feats_names:
            node_normalizers[feature] = MinMaxScaler((0, 1))

        for spider_patch in tqdm(self.graphs[fit_indices], position=0, leave=True, desc=f"Normalizer Fitting: ", colour="white", ncols=80):
            for feature in feats_names:
                if spider_patch.node_attr_schemes()[feature].shape == ():
                    node_normalizers[feature].partial_fit(spider_patch.ndata[feature].reshape((-1, 1)))
                else:
                    node_normalizers[feature].partial_fit(spider_patch.ndata[feature])

        for spider_patch in tqdm(self.graphs, position=0, leave=True, desc=f"Normalizing: ", colour="white", ncols=80):
            for feature in feats_names:
                if spider_patch.node_attr_schemes()[feature].shape == ():
                    spider_patch.ndata[feature] = torch.tensor(node_normalizers[feature].transform(spider_patch.ndata[feature].reshape((-1, 1))), dtype=torch.float32)
                else:
                    spider_patch.ndata[feature] = torch.tensor(node_normalizers[feature].transform(spider_patch.ndata[feature]), dtype=torch.float32)

        return node_normalizers

    def normalize_edge(self, fit_indices):

        edge_normalizers = {}
        feats_names = self.getEdgeFeatsName()
        for feature in feats_names:
            edge_normalizers[feature] = MinMaxScaler((0, 1))

        for spider_patch in tqdm(self.graphs[fit_indices], position=0, leave=True, desc=f"Normalizer Fitting: ", colour="white", ncols=80):
            for feature in feats_names:
                if spider_patch.edge_attr_schemes()[feature].shape == ():
                    edge_normalizers[feature].partial_fit(spider_patch.edata[feature].reshape((-1, 1)))
                else:
                    edge_normalizers[feature].partial_fit(spider_patch.edata[feature])

        for spider_patch in tqdm(self.graphs, position=0, leave=True, desc=f"Normalizing: ", colour="white", ncols=80):
            for feature in edge_normalizers.keys():
                if spider_patch.edge_attr_schemes()[feature].shape == ():
                    spider_patch.edata[feature] = torch.tensor(edge_normalizers[feature].transform(spider_patch.edata[feature].reshape((-1, 1))), dtype=torch.float32)
                else:
                    spider_patch.edata[feature] = torch.tensor(edge_normalizers[feature].transform(spider_patch.edata[feature]), dtype=torch.float32)

        return edge_normalizers

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


def generate_mesh(path, name, curvature):
    mesh = Mesh()
    mesh.loadFromMeshFile(path)
    mesh.computeCurvaturesTemp(curvature)
    mesh.save(f"../Retrieval/Mesh/{name}.pkl")


def generate_labels(mesh_name, labels_path):
    mat = scipy.io.loadmat(labels_path)
    face_labels = mat["label"].flatten()
    with open(f"../Retrieval/Mesh/{mesh_name}.pkl", "rb") as mesh_file:
        mesh = pkl.load(mesh_file)
    vertex_labels = []
    for face_list in mesh.vertex_faces:
        vertex_labels.append(np.argmax(np.bincount(face_labels[face_list])))
    for id, elem in enumerate(np.unique(vertex_labels)):
        vertex_labels = [id if x == elem else x for x in vertex_labels]
    with open(f"../Retrieval/Labels/{mesh_name.replace('BC', '').replace('LC', '')}.pkl", "wb") as label_file:
        pkl.dump(vertex_labels, label_file)
