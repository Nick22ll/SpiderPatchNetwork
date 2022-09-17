import warnings
import random
import gc
from dgl.data import DGLDataset
import os
import pickle as pkl

import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from MeshGraph.MeshGraph import SpiralMeshGraph


class SpiralMeshGraphDataset(DGLDataset):
    def __init__(self, dataset_name="", graphs=None, labels=None, sample_id=None, mesh_id=None):
        self.mesh_id = mesh_id
        self.sample_id = sample_id
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
        @param node_normalizers: (dict) { "feat_name" : sklearn.preprocessing.Scaler}, a dict that map a feature to his normalizer. If None normalizers are calculated at runtime.
        @return: node_normalizers: (dict) { "feat_name" : sklearn.preprocessing.Scaler}, a dict that map a feature to his normalizer used in the normalization process.
        """

        if node_normalizers is None:
            node_normalizers = {}
            feats_names = self.graphs[0].getNodeFeatsNames()
            if "vertices " in feats_names:
                feats_names.remove("vertices")
            for feature in feats_names:
                node_normalizers[feature] = MinMaxScaler((0, 1))

            for mesh_graph in tqdm(self.graphs, position=0, leave=True, desc=f"Normalizer Fitting: ", colour="white", ncols=80):
                for feature in feats_names:
                    if mesh_graph.node_attr_schemes()[feature].shape == ():
                        node_normalizers[feature].partial_fit(mesh_graph.ndata[feature].reshape((-1, 1)))
                    else:
                        node_normalizers[feature].partial_fit(mesh_graph.ndata[feature])

        for mesh_graph in tqdm(self.graphs, position=0, leave=True, desc=f"Normalizing: ", colour="white", ncols=80):
            for feature in node_normalizers.keys():
                if mesh_graph.node_attr_schemes()[feature].shape == ():
                    mesh_graph.ndata[feature] = torch.tensor(node_normalizers[feature].transform(mesh_graph.ndata[feature].reshape((-1, 1))), dtype=torch.float32)
                else:
                    mesh_graph.ndata[feature] = torch.tensor(node_normalizers[feature].transform(mesh_graph.ndata[feature]), dtype=torch.float32)

        return node_normalizers

    def normalize_edge(self, edge_normalizers=None):
        """
        Normalizes node data of the graphs. The normalization process is applied per feature
        @param edge_normalizers: (dict) { "feat_name" : sklearn.preprocessing.Scaler}, a dict that map a feature to his normalizer. If None normalizers are calculated at runtime.
        @return: edge_normalizers: (dict) { "feat_name" : sklearn.preprocessing.Scaler}, a dict that map a feature to his normalizer used in the normalization process.
        """

        if edge_normalizers is None:
            edge_normalizers = {}
            feats_names = self.graphs[0].getEdgeFeatsNames()
            for feature in feats_names:
                edge_normalizers[feature] = MinMaxScaler((0, 1))

            for mesh_graph in tqdm(self.graphs, position=0, leave=True, desc=f"Normalizer Fitting: ", colour="white", ncols=80):
                for feature in feats_names:
                    if mesh_graph.edge_attr_schemes()[feature].shape == ():
                        edge_normalizers[feature].partial_fit(mesh_graph.edata[feature].reshape((-1, 1)))
                    else:
                        edge_normalizers[feature].partial_fit(mesh_graph.edata[feature])

        for mesh_graph in tqdm(self.graphs, position=0, leave=True, desc=f"Normalizing: ", colour="white", ncols=80):
            for feature in edge_normalizers.keys():
                if mesh_graph.edge_attr_schemes()[feature].shape == ():
                    mesh_graph.edata[feature] = torch.tensor(edge_normalizers[feature].transform(mesh_graph.edata[feature].reshape((-1, 1))), dtype=torch.float32)
                else:
                    mesh_graph.edata[feature] = torch.tensor(edge_normalizers[feature].transform(mesh_graph.edata[feature]), dtype=torch.float32)

        return edge_normalizers

    def aggregateNodeFeatures(self, feat_names=None):
        """
        @param feat_names: list of features key to aggregate in a single feature (called aggregate_feature)
        @return: the aggregate feature shape along axis 1 ( if shape 30x4 --> returns 4 )
        """
        if feat_names is None:
            feat_names = self.getNodeFeatsName()
            feat_names.remove("vertices")

        for mesh_graph in tqdm(self.graphs):
            mesh_graph.ndata["aggregated_feats"] = mesh_graph.ndata[feat_names[0]]
            mesh_graph.ndata.pop(feat_names[0])
            for name in feat_names[1:]:
                if mesh_graph.node_attr_schemes()[name].shape == ():
                    mesh_graph.ndata["aggregated_feats"] = torch.hstack((mesh_graph.ndata["aggregated_feats"], torch.reshape(mesh_graph.ndata[name], (-1, 1))))
                    mesh_graph.ndata.pop(name)
                else:
                    mesh_graph.ndata["aggregated_feats"] = torch.hstack((mesh_graph.ndata["aggregated_feats"], mesh_graph.ndata[name]))
                    mesh_graph.ndata.pop(name)

        return self.graphs[0].ndata["aggregated_feats"].shape

    def aggregateEdgeFeatures(self, feat_names="all"):
        """
        @param feat_names: list of features key to aggregate in a single feature (called aggregate_feature)
        @return: the aggregate feature shape along axis 1 ( if shape 30x4 --> returns 4 )
        """
        if feat_names == "all":
            feat_names = self.getEdgeFeatsName()

        for mesh_graph in self.graphs:
            mesh_graph.edata["weights"] = mesh_graph.edata[feat_names[0]]
            for name in feat_names[1:]:
                if mesh_graph.edge_attr_schemes()[name].shape == ():
                    mesh_graph.edata["weights"] = torch.hstack((mesh_graph.edata["weights"], torch.reshape(mesh_graph.edata[name], (-1, 1))))
                    mesh_graph.edata.pop(name)
                else:
                    mesh_graph.edata["weights"] = torch.hstack((mesh_graph.edata["weights"], mesh_graph.edata[name]))
                    mesh_graph.edata.pop(name)

        return self.graphs[0].edata["weights"].shape

    def removeNonAggregatedFeatures(self):
        feats_name = self.getNodeFeatsName()
        feats_name.remove("aggregated_feats")
        for mesh_graph in self.graphs:
            for name in feats_name:
                mesh_graph.ndata.pop(name)

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
        Load a MeshDataset instance from a pkl file.
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
        self.sample_id = loaded_dataset.sample_id
        self.mesh_id = loaded_dataset.mesh_id
        self._name = dataset_name

    def fromRawConcRings(self, configuration, resolution_level="all", graph_for_mesh=10, conc_for_graph=10, connection_number=0, feature_to_keep=None):
        """
        Generate a MeshGraphDataset from a ConcRingDataset.

        @param configuration: (string) the path to the Patches dataset folder structure.
        @param resolution_level: (string) e.g. all, level_0, level_1, ecc... The resolution level of the mesh to use.
        @param graph_for_mesh: (int) number of MeshGraph to generate per Mesh.
        @param conc_for_graph: (int) number of Patch to use as nodes per MeshGraph
        @param connection_number: (int) number of neighbours per patch, leave zero for a fully connected graph
        @param feature_to_keep: (list) A list of node features keys to keep, if None keep all features
        @return: self
        """

        print(f"Loading MeshGraph Dataset from: {configuration}")
        import re
        random.seed(22)
        self.graphs = []
        self.labels = []
        self.sample_id = []
        self.mesh_id = []
        if resolution_level != "all":
            for label in tqdm(os.listdir(f"Datasets/ConcentricRings/SHREC17_{configuration}"), position=0, desc=f"Mesh Class Loading: ", colour="white", ncols=80):
                for mesh_sample_id in os.listdir(f"Datasets/ConcentricRings/SHREC17_{configuration}/{label}"):
                    concRings_filename = os.listdir(f"Datasets/ConcentricRings/SHREC17_{configuration}/{label}/{mesh_sample_id}/resolution_{resolution_level}")[0]
                    gc.disable()
                    with open(f"Datasets/ConcentricRings/SHREC17_{configuration}/{label}/{mesh_sample_id}/resolution_{resolution_level}/{concRings_filename}", "rb") as pkl_file:
                        concentricRings = pkl.load(pkl_file)
                    with open(f"Datasets/Meshes/SHREC17/{label}/{mesh_sample_id}/resolution_{resolution_level}/{concRings_filename}", "rb") as pkl_file:
                        mesh = pkl.load(pkl_file)
                        gc.enable()
                        for _ in range(graph_for_mesh):
                            mesh_id = [int(s) for s in re.findall(r'\d+', concRings_filename)]
                            samples_id = list(range(len(concentricRings)))
                            random.shuffle(samples_id)
                            samples_number = 0
                            samples = []
                            for id in samples_id:
                                if samples_number >= conc_for_graph:
                                    break
                                RI_pos = configuration.find("RI")
                                if concentricRings[id].first_valid_rings(int(configuration[RI_pos + 2:configuration.find("_", RI_pos + 2)])):
                                    samples.append(concentricRings[id])
                                    samples_number += 1
                            self.graphs.append(
                                SpiralMeshGraph(mesh, samples, sample_id=int(mesh_sample_id.removeprefix("id_")), mesh_id=int(mesh_id[0]), resolution_level=resolution_level.removeprefix("resolution_"), neighbours_number=connection_number, keep_feats_names=feature_to_keep))
                            self.labels.append(int(label.removeprefix("class_")))
                            self.sample_id.append(int(mesh_sample_id.removeprefix("id_")))
                            self.mesh_id.extend(mesh_id)
        else:
            for label in tqdm(os.listdir(f"Datasets/ConcentricRings/SHREC17_{configuration}"), position=0, desc=f"Mesh Class Loading: ", colour="white", ncols=80):
                for mesh_sample_id in tqdm(os.listdir(f"Datasets/ConcentricRings/SHREC17_{configuration}/{label}"), position=1, desc=f"Sample loading: ", colour="white", ncols=80, leave=False):
                    for resolution_level in os.listdir(f"Datasets/ConcentricRings/SHREC17_{configuration}/{label}/{mesh_sample_id}"):
                        concRings_filename = os.listdir(f"Datasets/ConcentricRings/SHREC17_{configuration}/{label}/{mesh_sample_id}/{resolution_level}")[0]
                        gc.disable()
                        with open(f"Datasets/ConcentricRings/SHREC17_{configuration}/{label}/{mesh_sample_id}/{resolution_level}/{concRings_filename}", "rb") as pkl_file:
                            concentricRings = pkl.load(pkl_file)
                        with open(f"Datasets/Meshes/SHREC17/{label}/{mesh_sample_id}/{resolution_level}/{concRings_filename.replace('concRing', 'mesh')}", "rb") as pkl_file:
                            mesh = pkl.load(pkl_file)
                            gc.enable()
                            for _ in range(graph_for_mesh):
                                mesh_id = [int(s) for s in re.findall(r'\d+', concRings_filename)]
                                samples_id = list(range(len(concentricRings)))
                                random.shuffle(samples_id)
                                samples_number = 0
                                samples = []
                                for id in samples_id:
                                    if samples_number >= conc_for_graph:
                                        break
                                    RI_pos = configuration.find("RI")
                                    rings = int(configuration[RI_pos + 2:configuration.find("_", RI_pos + 2)])
                                    if concentricRings[id].first_valid_rings(rings):
                                        samples.append(concentricRings[id])
                                        samples_number += 1
                                self.graphs.append(
                                    SpiralMeshGraph(mesh, samples, sample_id=int(mesh_sample_id.removeprefix("id_")), mesh_id=int(mesh_id[0]), resolution_level=resolution_level.removeprefix("resolution_"), neighbours_number=connection_number))
                                self.labels.append(int(label.removeprefix("class_")))
                                self.sample_id.append(int(mesh_sample_id.removeprefix("id_")))
                                self.mesh_id.extend(mesh_id)

        super().__init__(name=self.name)
        return self
