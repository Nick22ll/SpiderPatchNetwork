import warnings
import gc

import dgl
import numpy as np
from dgl.data import DGLDataset
import os
import pickle as pkl
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from SpiderPatch.SpiderPatch import SpiderPatch


class SpiderPatchDataset(DGLDataset):
    def __init__(self, dataset_name="", graphs=None, labels=None):

        self.graphs = np.array(graphs)
        self.labels = np.array(labels)
        self.seed_point_indices = np.empty(0)
        self.node_normalizers = {}
        self.edge_normalizers = {}
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

    def getTrainTestMask(self, train_samples=200, percentage=False):
        rng = np.random.default_rng(22)
        train_indices = []
        test_indices = []
        for label in np.unique(self.labels):
            label_indices = np.where(self.labels == label)[0]
            if percentage:
                train_label_indices = rng.choice(label_indices, int((train_samples / 100) * len(label_indices)), replace=False)
            else:
                train_label_indices = rng.choice(label_indices, train_samples, replace=False)
            test_indices.extend(list(np.delete(label_indices, np.argwhere(np.isin(label_indices, train_label_indices)))))
            train_indices.extend(list(train_label_indices))
        return train_indices, test_indices

    def train_normalizers(self, fit_indices):
        feats_names = self.getNodeFeatsName()
        if "vertices" in feats_names:
            feats_names.remove("vertices")
        if "weight" in feats_names:
            feats_names.remove("weight")

        self.node_normalizers = {}
        for feature in feats_names:
            self.node_normalizers[feature] = MinMaxScaler((0, 1))

        for spider_patch in tqdm(self.graphs[fit_indices], position=0, leave=True, desc=f"Normalizer Fitting: ", colour="white", ncols=80):
            for feature in feats_names:
                if spider_patch.node_attr_schemes()[feature].shape == ():
                    self.node_normalizers[feature].partial_fit(spider_patch.ndata[feature].reshape((-1, 1)))
                else:
                    self.node_normalizers[feature].partial_fit(spider_patch.ndata[feature])

    def normalize(self, indices_to_normalize=None):
        feats_names = self.getNodeFeatsName()
        if "vertices" in feats_names:
            feats_names.remove("vertices")
        if "weights" in feats_names:
            feats_names.remove("weights")

        if indices_to_normalize is None:
            to_normalize = self.graphs
        else:
            to_normalize = self.graphs[indices_to_normalize]
        for spider_patch in tqdm(to_normalize, position=0, leave=True, desc=f"Normalizing: ", colour="white", ncols=80):
            for feature in feats_names:
                if spider_patch.node_attr_schemes()[feature].shape == ():
                    spider_patch.ndata[feature] = torch.tensor(self.node_normalizers[feature].transform(spider_patch.ndata[feature].reshape((-1, 1))), dtype=torch.float32)
                else:
                    spider_patch.ndata[feature] = torch.tensor(self.node_normalizers[feature].transform(spider_patch.ndata[feature]), dtype=torch.float32)

    def train_edge_normalizers(self, fit_indices):
        self.edge_normalizers = {}
        feats_names = self.getEdgeFeatsName()
        if "weights" in feats_names:
            feats_names.remove("weights")

        for feature in feats_names:
            self.edge_normalizers[feature] = MinMaxScaler((0, 1))

        for spider_patch in tqdm(self.graphs[fit_indices], position=0, leave=True, desc=f"Normalizer Fitting: ", colour="white", ncols=80):
            for feature in feats_names:
                if spider_patch.edge_attr_schemes()[feature].shape == ():
                    self.edge_normalizers[feature].partial_fit(spider_patch.edata[feature].reshape((-1, 1)))
                else:
                    self.edge_normalizers[feature].partial_fit(spider_patch.edata[feature])

    def normalize_edges(self, indices_to_normalize=None):
        feats_names = self.getEdgeFeatsName()

        if "weights" in feats_names:
            feats_names.remove("weights")

        if indices_to_normalize is None:
            to_normalize = self.graphs
        else:
            to_normalize = self.graphs[indices_to_normalize]

        for spider_patch in tqdm(self.graphs[to_normalize], position=0, leave=True, desc=f"Normalizing: ", colour="white", ncols=80):
            for feature in self.edge_normalizers.keys():
                if spider_patch.edge_attr_schemes()[feature].shape == ():
                    spider_patch.edata[feature] = torch.tensor(self.edge_normalizers[feature].transform(spider_patch.edata[feature].reshape((-1, 1))), dtype=torch.float32)
                else:
                    spider_patch.edata[feature] = torch.tensor(self.edge_normalizers[feature].transform(spider_patch.edata[feature]), dtype=torch.float32)

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

    def aggregateNodeFeatures(self, to_aggreg_feats=None, aggreg_name="aggregated_feats"):
        """
        @param aggreg_name:
        @param to_aggreg_feats: list of features key to aggregate in a single feature (called aggregate_feature)
        @return: the aggregate feature shape along axis 1 ( if shape 30x4 --> returns 4 )
        """
        if to_aggreg_feats is None:
            to_aggreg_feats = self.getNodeFeatsName()
            to_aggreg_feats.remove("vertices")

        for patch in tqdm(self.graphs, position=0, leave=True, desc=f"Aggregating Nodes: ", colour="white", ncols=80):
            if patch.node_attr_schemes()[to_aggreg_feats[0]].shape == ():
                patch.ndata[aggreg_name] = patch.ndata[to_aggreg_feats[0]].view(-1, 1)
            else:
                patch.ndata[aggreg_name] = patch.ndata[to_aggreg_feats[0]]
            patch.ndata.pop(to_aggreg_feats[0])
            for name in to_aggreg_feats[1:]:
                if patch.node_attr_schemes()[name].shape == ():
                    patch.ndata[aggreg_name] = torch.hstack((patch.ndata[aggreg_name], patch.ndata[name].view(-1, 1)))
                else:
                    patch.ndata[aggreg_name] = torch.hstack((patch.ndata[aggreg_name], patch.ndata[name]))
                patch.ndata.pop(name)

        return self.graphs[0].ndata[aggreg_name].shape

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
        for id, graph in tqdm(enumerate(self.graphs), position=0, leave=True, desc=f"Aggregating edge features: ", colour="white", ncols=80):
            if graph.num_nodes() != num_nodes:
                to_delete.append(id)
        to_delete = np.sort(to_delete)
        if to_delete != []:
            self.graphs = np.delete(self.graphs, to_delete)
            self.labels = np.delete(self.labels, to_delete)

    def removeClasses(self, class_to_remove):
        for label in class_to_remove:
            indices = np.where(self.labels == label)[0]
            self.graphs = np.delete(self.graphs, indices)
            self.labels = np.delete(self.labels, indices)

        for id, label in enumerate(np.unique(self.labels)):
            self.labels[self.labels == label] = id

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
