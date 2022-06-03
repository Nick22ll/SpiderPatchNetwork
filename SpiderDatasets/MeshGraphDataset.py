import warnings
import random

from dgl.data import DGLDataset
import os
import pickle as pkl
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from SpiderPatch.MeshGraph import MeshGraph


class MeshGraphDataset(DGLDataset):
    def __init__(self, dataset_name="", graphs=None, labels=None, mesh_id=None):
        self.mesh_id = mesh_id
        self.graphs = graphs
        self.labels = labels
        super().__init__(name=dataset_name)

    def process(self):
        pass

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def getNodeFeatsName(self):
        return list(self.graphs[0].ndata.keys())

    def getEdgeFeatsName(self):
        return list(self.graphs[0].edata.keys())

    def numClasses(self):
        return len(torch.unique(self.labels))

    def to(self, device):
        self.graphs = [self.graphs[idx].to(device) for idx in range(len(self.graphs))]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.labels = torch.tensor(self.labels, device=device, dtype=torch.int64)

    def normalize(self, node_normalizers=None):
        """
        Normalizes node data of the graphs. The normalization process is applied per feature
        :param node_normalizers: (dict) { "feat_name" : sklearn.preprocessing.Scaler}, a dict that map a feature to his normalizer. If None normalizers are calculated at runtime.
        :return: node_normalizers: (dict) { "feat_name" : sklearn.preprocessing.Scaler}, a dict that map a feature to his normalizer used in the normalization process.
        """

        if node_normalizers is None:
            node_normalizers = {}
            for feature in self.graphs[0].patches[0].getNodeFeatsNames():
                node_normalizers[feature] = MinMaxScaler((0, 1))

            for mesh_graph in tqdm(self.graphs, position=0, leave=True, desc=f"Normalizer Fitting: ", colour="white", ncols=80):
                for patch in mesh_graph.patches:
                    for feature in patch.getNodeFeatsNames():
                        if patch.node_attr_schemes()[feature].shape == ():
                            node_normalizers[feature].partial_fit(patch.ndata[feature].reshape((-1, 1)))
                        else:
                            node_normalizers[feature].partial_fit(patch.ndata[feature])

        for mesh_graph in tqdm(self.graphs, position=0, leave=True, desc=f"Normalizing: ", colour="white", ncols=80):
            for patch in mesh_graph.patches:
                for feature in patch.getNodeFeatsNames():
                    if patch.node_attr_schemes()[feature].shape == ():
                        patch.ndata[feature] = torch.tensor(node_normalizers[feature].transform(patch.ndata[feature].reshape((-1, 1))), dtype=torch.float32)
                    else:
                        patch.ndata[feature] = torch.tensor(node_normalizers[feature].transform(patch.ndata[feature]), dtype=torch.float32)

        return node_normalizers

    def save_to(self, save_path=None):
        os.makedirs(save_path, exist_ok=True)
        if save_path is not None:
            path = f"{save_path}/{self.name}.pkl"
        else:
            path = f"{self.save_path}/{self.name}.pkl"
        with open(path, "wb") as dataset_file:
            pkl.dump(self, dataset_file)

    def load_from(self, load_path=None, dataset_name=None):
        """
        Load a MeshDataset instance from a pkl file.
        :param load_path: (string) the path to the folder containing the dataset structure folder e.g. "Datasets/MeshGraphs/SHREC17_R10_R16_P6_Normalized".
        :param dataset_name: (string) the name of the dataset file e.g. "SHREC17_R10_R16_P6_level_0_Normalized_test.pkl"
        :return:
        """
        if load_path is not None:
            path = f"{load_path}/{dataset_name}.pkl"
        else:
            path = f"{self.save_path}/{dataset_name}.pkl"

        with open(path, "rb") as dataset_file:
            loaded_dataset = pkl.load(dataset_file)
        self.graphs = loaded_dataset.graphs
        self.labels = loaded_dataset.labels
        self.mesh_id = loaded_dataset.mesh_id
        self._name = dataset_name

    def fromRawPatches(self, load_path, resolution_level="all", graph_for_mesh=10, patch_for_graph=10):
        """
        Generate a MeshGraphDataset from a PatchesDataset.
        :param load_path: (string) the path to the Patches dataset folder structure.
        :param resolution_level: (string) e.g. all, level_0, level_1, ecc... The resolution level of the mesh to use.
        :param graph_for_mesh: (int) number of MeshGraph to generate per Mesh.
        :param patch_for_graph: (int) number of Patch to use as nodes per MeshGraph
        :return: self
        """

        print(f"Loading MeshGraph Dataset from: {load_path}")
        import re
        random.seed(22)
        self.graphs = []
        self.labels = []
        self.mesh_id = []
        if resolution_level != "all":
            for label in tqdm(os.listdir(f"{load_path}/{resolution_level}"), position=0, desc=f"Mesh Class Loading: ", colour="white", ncols=80):
                for patches_filename in os.listdir(f"{load_path}/{resolution_level}/{label}"):
                    with open(f"{load_path}/{resolution_level}/{label}/{patches_filename}", "rb") as pkl_file:
                        patches = pkl.load(pkl_file)
                        for _ in range(graph_for_mesh):
                            self.graphs.append(MeshGraph(random.sample(patches, patch_for_graph)))
                            self.labels.append(int(label))
                            self.mesh_id.extend([int(s) for s in re.findall(r'\d+', patches_filename)])
        else:
            for resolution_level in tqdm(os.listdir(f"{load_path}"), position=0, desc=f"Resolution Level Loading: ", colour="green", ncols=100):
                for label in tqdm(os.listdir(f"{load_path}/{resolution_level}"), position=0, desc=f"Mesh Class Loading: ", colour="white", ncols=80):
                    for patches_filename in os.listdir(f"{load_path}/{resolution_level}/{label}"):
                        with open(f"{load_path}/{resolution_level}/{label}/{patches_filename}", "rb") as pkl_file:
                            patches = pkl.load(pkl_file)
                            for _ in range(graph_for_mesh):
                                self.graphs.append(MeshGraph(random.sample(patches, patch_for_graph)))
                                self.labels.append(int(label))
                                self.mesh_id.extend([int(s) for s in re.findall(r'\d+', patches_filename)])

        super().__init__(name=self.name)
        return self


