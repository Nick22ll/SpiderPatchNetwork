import warnings
import gc

import numpy as np
from dgl.data import DGLDataset
import os
import pickle as pkl
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class SpiderPatchDataset(DGLDataset):
    def __init__(self, dataset_name="", graphs=None, labels=None):
        self.graphs = graphs
        self.labels = labels
        self.seed_point_indices = None
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
            feats_names = self.getNodeFeatsName()
            if "vertices" in feats_names:
                feats_names.remove("vertices")
            if "weight" in feats_names:
                feats_names.remove("weight")
            for feature in feats_names:
                node_normalizers[feature] = MinMaxScaler((0, 1))

            for spider_patch in tqdm(self.graphs, position=0, leave=True, desc=f"Normalizer Fitting: ", colour="white", ncols=80):
                for feature in feats_names:
                    if spider_patch.node_attr_schemes()[feature].shape == ():
                        node_normalizers[feature].partial_fit(spider_patch.ndata[feature].reshape((-1, 1)))
                    else:
                        node_normalizers[feature].partial_fit(spider_patch.ndata[feature])

        for spider_patch in tqdm(self.graphs, position=0, leave=True, desc=f"Normalizing: ", colour="white", ncols=80):
            for feature in node_normalizers.keys():
                if spider_patch.node_attr_schemes()[feature].shape == ():
                    spider_patch.ndata[feature] = torch.tensor(node_normalizers[feature].transform(spider_patch.ndata[feature].reshape((-1, 1))), dtype=torch.float32)
                else:
                    spider_patch.ndata[feature] = torch.tensor(node_normalizers[feature].transform(spider_patch.ndata[feature]), dtype=torch.float32)

        return node_normalizers

    def normalize_edge(self, edge_normalizers=None):
        """
        Normalizes edge data of the graphs. The normalization process is applied per feature
        @param edge_normalizers: (dict) { "feat_name" : sklearn.preprocessing.Scaler}, a dict that map a feature to his normalizer. If None normalizers are calculated at runtime.
        @return: edge_normalizers: (dict) { "feat_name" : sklearn.preprocessing.Scaler}, a dict that map a feature to his normalizer used in the normalization process.
        """

        if edge_normalizers is None:
            edge_normalizers = {}
            feats_names = self.getEdgeFeatsName()
            for feature in feats_names:
                edge_normalizers[feature] = MinMaxScaler((0, 1))

            for spider_patch in tqdm(self.graphs, position=0, leave=True, desc=f"Normalizer Fitting: ", colour="white", ncols=80):
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

    def keepCurvaturesResolution(self, radius_to_keep=None):
        """
        Select the curvature resolution to keep, if the initial curvature feature is an array with shape (1x5) then, if I choose to keep resolution 2 and 4, it will be (1x2)
        @param radius_to_keep: (list) A list of resolution levels (0,4) to keep
        @return: the aggregate feature shape along axis 1 ( if shape 30x4 --> returns 4 )
        """
        if radius_to_keep is None:
            radius_to_keep = [0, 1, 2, 3, 4]

        if radius_to_keep == [0, 1, 2, 3, 4]:
            return self.graphs[0].ndata["local_depth"].shape

        for patch in tqdm(self.graphs, position=0, leave=True, desc=f"Keeping radius: ", colour="white", ncols=80):
            for feat_name in ["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"]:
                patch.ndata[feat_name] = patch.ndata[feat_name][:, radius_to_keep]
        return self.graphs[0].ndata["local_depth"].shape

    def aggregateNodeFeatures(self, feat_names=None):
        """
        @param feat_names: list of features key to aggregate in a single feature (called aggregate_feature)
        @return: the aggregate feature shape along axis 1 ( if shape 30x4 --> returns 4 )
        """
        if feat_names is None:
            feat_names = self.graphs[0]
            feat_names.remove("vertices")

        for patch in tqdm(self.graphs, position=0, leave=True, desc=f"Aggregating node features: ", colour="white", ncols=80):
            patch.ndata["aggregated_feats"] = patch.ndata[feat_names[0]]
            patch.ndata.pop(feat_names[0])
            for name in feat_names[1:]:
                if patch.node_attr_schemes()[name].shape == ():
                    patch.ndata["aggregated_feats"] = torch.hstack((patch.ndata["aggregated_feats"], torch.reshape(patch.ndata[name], (-1, 1))))
                    patch.ndata.pop(name)
                else:
                    patch.ndata["aggregated_feats"] = torch.hstack((patch.ndata["aggregated_feats"], patch.ndata[name]))
                    patch.ndata.pop(name)

        return self.graphs[0].ndata["aggregated_feats"].shape

    def aggregateEdgeFeatures(self, feat_names="all"):
        """
        @param feat_names: list of features key to aggregate in a single feature (called aggregate_feature)
        @return: the aggregate feature shape along axis 1 ( if shape 30x4 --> returns 4 )
        """
        if feat_names == "all":
            feat_names = self.getEdgeFeatsName()

        for patch in tqdm(self.graphs, position=0, leave=True, desc=f"Aggregating edge features: ", colour="white", ncols=80):
            patch.edata["weights"] = patch.edata[feat_names[0]]
            for name in feat_names[1:]:
                if patch.edge_attr_schemes()[name].shape == ():
                    patch.edata["weights"] = torch.hstack((patch.edata["weights"], torch.reshape(patch.edata[name], (-1, 1))))
                    patch.edata.pop(name)
                else:
                    patch.edata["weights"] = torch.hstack((patch.edata["weights"], patch.edata[name]))
                    patch.edata.pop(name)

        return self.graphs[0].edata["weights"].shape

    def removeNonAggregatedFeatures(self, to_keep=None):
        feats_name = self.getNodeFeatsName()
        feats_name.remove("aggregated_feats")
        if to_keep is not None:
            for feat_name in to_keep:
                feats_name.remove(feat_name)
        for patch in self.graphs:
            for name in feats_name:
                patch.ndata.pop(name)

    def selectGraphsByNumNodes(self, num_nodes):
        to_delete = []
        for id, graph in enumerate(self.graphs):
            if graph.num_nodes() != num_nodes:
                to_delete.append(id)
        to_delete = np.sort(to_delete)

        for i in range(len(to_delete) - 1, -1, -1):
            self.graphs.pop(to_delete[i])
            self.labels.pop(to_delete[i])

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
        self.seed_point_indices = loaded_dataset.seed_point_indices
        self._name = dataset_name
