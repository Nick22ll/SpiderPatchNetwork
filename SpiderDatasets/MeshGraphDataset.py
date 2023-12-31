import gc
import multiprocessing
import os
import pickle
import pickle as pkl
import shutil
import warnings
from time import time, sleep

import numpy as np
import torch
from dgl.data import DGLDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from tqdm import tqdm

from MeshGraph.MeshGraph import MeshGraph, MeshGraphGeo
from OutliersUtils import IQROutliers, SDOutliers
import re


class MeshGraphDataset(DGLDataset):
    def __init__(self, dataset_name="", graphs=None, labels=None, sample_id=None, mesh_id=None):
        self.mesh_id = mesh_id
        self.sample_id = sample_id
        self.graphs = graphs
        self.labels = labels
        super().__init__(name=dataset_name)

    def process(self):
        pass

    def __getitem__(self, i):
        if len(i) > 1:
            return [(self.graphs[j], self.labels[j]) for j in i]
        else:
            return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

    def getNodeFeatsName(self):
        return list(self.graphs[0].ndata.keys())

    def getEdgeFeatsName(self):
        return list(self.graphs[0].edata.keys())

    def getSpiderPatchNodeFeatsNames(self):
        return self.graphs[0].patches[0].getNodeFeatsNames()

    def getSpiderPatchEdgeFeatsNames(self):
        return self.graphs[0].patches[0].getEdgeFeatsNames()

    def numClasses(self):
        if isinstance(self.labels, np.ndarray):
            return len(np.unique(self.labels))
        elif isinstance(self.labels, torch.Tensor):
            return len(torch.unique(self.labels))

    def to(self, device):
        self.graphs = np.array([self.graphs[idx].to(device) for idx in range(len(self.graphs))])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.labels = torch.tensor(self.labels, device=device, dtype=torch.int64, requires_grad=False)

    def aggregateSpiderPatchesNodeFeatures(self, to_aggreg_feats=None, aggreg_name="aggregated_feats"):
        """
        @param aggreg_name:
        @param to_aggreg_feats: list of features key to aggregate in a single feature (called aggregate_feature)
        @return: the aggregate feature shape along axis 1 ( if shape 30x4 --> returns 4 )
        """
        if to_aggreg_feats is None:
            to_aggreg_feats = self.getSpiderPatchNodeFeatsNames()
            to_aggreg_feats.remove("vertices")

        for mesh_graph in tqdm(self.graphs, position=0, leave=True, desc=f"Aggregating Nodes: ", colour="white", ncols=80):
            for patch in mesh_graph.patches:
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

        return self.graphs[0].patches[0].ndata[aggreg_name].shape

    def aggregateSpiderPatchEdgeFeatures(self, feat_names="all"):
        """
        @param feat_names: list of features key to aggregate in a single feature (called aggregate_feature)
        @return: the aggregate feature shape along axis 1 ( if shape 30x4 --> returns 4 )
        """
        if feat_names == "all":
            feat_names = self.getSpiderPatchEdgeFeatsNames()

        for mesh_graph in tqdm(self.graphs, position=0, leave=True, desc=f"Aggregating Edge: ", colour="white", ncols=80):
            for patch in mesh_graph.patches:
                patch.edata["weights"] = patch.edata[feat_names[0]]
                for name in feat_names[1:]:
                    if patch.edge_attr_schemes()[name].shape == ():
                        patch.edata["weights"] = torch.hstack((patch.edata["weights"], torch.reshape(patch.edata[name], (-1, 1))))
                        patch.edata.pop(name)
                    else:
                        patch.edata["weights"] = torch.hstack((patch.edata["weights"], patch.edata[name]))
                        patch.edata.pop(name)

        return self.graphs[0].patches[0].edata["weights"].shape

    def removeSpiderPatchByNumNodes(self, num_nodes):
        deleted_patches = 0
        for mesh_graph in tqdm(self.graphs, position=0, leave=True, desc=f"Removing Spider Patches: ", colour="white", ncols=80):
            to_delete = np.empty(0, dtype=int)
            for id, spider_patch in enumerate(mesh_graph.patches):
                if spider_patch.num_nodes() < num_nodes:
                    to_delete = np.append(to_delete, id)
            mesh_graph.patches = np.delete(mesh_graph.patches, to_delete)
            mesh_graph.remove_nodes(torch.tensor(to_delete, dtype=torch.int64))

            isolated_nodes = ((mesh_graph.in_degrees() == 0) & (mesh_graph.out_degrees() == 0)).nonzero().squeeze(1)
            mesh_graph.patches = np.delete(mesh_graph.patches, isolated_nodes)
            mesh_graph.remove_nodes(isolated_nodes)

            deleted_patches += len(to_delete) + len(isolated_nodes)
        print("SpiderPatches deleted: ", deleted_patches)

    def removeNonAggregatedFeatures(self, no_remove=[]):
        feats_name = self.getSpiderPatchNodeFeatsNames()
        for feat_name in no_remove:
            if feat_name in feats_name:
                feats_name.remove(feat_name)
        to_del = []
        for idx, feat_name in enumerate(feats_name):
            if "aggregated" in feat_name:
                to_del.append(idx)
        feats_name = np.delete(feats_name, to_del)
        for mesh_graph in tqdm(self.graphs, position=0, leave=True, desc=f"Removing: ", colour="white", ncols=80):
            for patch in mesh_graph.patches:
                for name in feats_name:
                    patch.ndata.pop(name)

    def removeClasses(self, class_to_remove):
        for label in class_to_remove:
            indices = np.where(self.labels == label)[0]
            self.graphs = np.delete(self.graphs, indices)
            self.labels = np.delete(self.labels, indices)
            self.mesh_id = np.delete(self.mesh_id, indices)
            self.sample_id = np.delete(self.sample_id, indices)

        for id, label in enumerate(np.unique(self.labels)):
            self.labels[self.labels == label] = id

    def keepCurvaturesResolution(self, radius_to_keep=None):
        """
        Select the curvature resolution to keep, if the initial curvature feature is an array with shape (1x5) then, if I choose to keep resolution 2 and 4, it will be (1x2)
        @param radius_to_keep: (list) A list of resolution levels (0,4) to keep
        @return: the aggregate feature shape along axis 1 ( if shape 30x4 --> returns 4 )
        """
        if radius_to_keep is None:
            radius_to_keep = [0, 1, 2, 3, 4]

        if radius_to_keep == [0, 1, 2, 3, 4]:
            return self.graphs[0].patches[0].ndata["local_depth"].shape

        for mesh_graph in tqdm(self.graphs, position=0, leave=True, desc=f"Keeping radius: ", colour="white", ncols=80):
            for patch in mesh_graph.patches:
                for feat_name in ["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"]:
                    patch.ndata[feat_name] = patch.ndata[feat_name][:, radius_to_keep]
        return self.graphs[0].patches[0].ndata["local_depth"].shape

    def shuffle(self):
        p = np.random.permutation(len(self.labels))
        self.graphs = self.graphs[p]
        self.labels = self.labels[p]
        self.mesh_id = self.mesh_id[p]
        self.sample_id = self.sample_id[p]

    def getTrainTestMask(self, train_meshes_per_class=10, percentage=False):
        rng = np.random.default_rng(717)
        train_indices = np.empty(0, dtype=int)
        test_indices = np.empty(0, dtype=int)
        for label in np.unique(self.labels):
            label_indices = np.where(self.labels == label)[0]
            uniques_mesh_id = np.unique(self.mesh_id[label_indices])
            if percentage:
                to_train = rng.choice(uniques_mesh_id, int((train_meshes_per_class / 100) * len(uniques_mesh_id)), replace=False)
            else:
                to_train = rng.choice(uniques_mesh_id, train_meshes_per_class, replace=False)

            for mesh_id in to_train:
                mesh_id_indices = np.where(self.mesh_id == mesh_id)[0]
                train_indices = np.hstack((train_indices, mesh_id_indices))

            for mesh_id in list(np.delete(uniques_mesh_id, np.argwhere(np.isin(uniques_mesh_id, to_train)))):
                mesh_id_indices = np.where(self.mesh_id == mesh_id)[0]
                test_indices = np.hstack((test_indices, mesh_id_indices))

        return train_indices, test_indices

    def getCrossValidationMask(self, train_meshes_per_class=11):  # TODO funzione fatta velocemente per l'articolo, non funziona per valori diversi da 11!!!!!!!!

        unique_labels, counts = np.unique(self.labels, return_counts=True)
        mesh_per_label = min(counts / (len(self.graphs) / len(np.unique(self.sample_id))))
        k_fold = int(mesh_per_label // (mesh_per_label - train_meshes_per_class))

        cross_validation_masks = {}
        for fold in range(k_fold):
            cross_validation_masks[fold] = {}
            train_indices = np.empty(0, dtype=int)
            test_indices = np.empty(0, dtype=int)
            for label in unique_labels:
                label_indices = np.where(self.labels == label)[0]
                uniques_mesh_id = np.unique(self.mesh_id[label_indices])
                to_test = uniques_mesh_id[fold]

                for mesh_id in [to_test]:
                    mesh_id_indices = np.where(self.mesh_id == mesh_id)[0]
                    test_indices = np.hstack((test_indices, mesh_id_indices))

                for mesh_id in list(np.delete(uniques_mesh_id, np.argwhere(np.isin(uniques_mesh_id, to_test)))):
                    mesh_id_indices = np.where(self.mesh_id == mesh_id)[0]
                    train_indices = np.hstack((train_indices, mesh_id_indices))
            cross_validation_masks[fold]["train_indices"] = train_indices
            cross_validation_masks[fold]["test_indices"] = test_indices

        return cross_validation_masks

    def getSHREC20CrossValidationMask(self):  # TODO funzione fatta velocemente per l'articolo
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        k_fold = 5
        test_meshes_per_fold = 4

        cross_validation_masks = {}
        for fold in range(k_fold):
            cross_validation_masks[fold] = {}
            train_indices = np.empty(0, dtype=int)
            test_indices = np.empty(0, dtype=int)
            for label in unique_labels:
                label_indices = np.where(self.labels == label)[0]
                uniques_mesh_id = np.unique(self.mesh_id[label_indices])
                to_test = uniques_mesh_id[fold * test_meshes_per_fold: (fold * test_meshes_per_fold + test_meshes_per_fold)]

                for mesh_id in to_test:
                    mesh_id_indices = np.where(self.mesh_id == mesh_id)[0]
                    test_indices = np.hstack((test_indices, mesh_id_indices))

                for mesh_id in list(np.delete(uniques_mesh_id, np.argwhere(np.isin(uniques_mesh_id, to_test)))):
                    mesh_id_indices = np.where(self.mesh_id == mesh_id)[0]
                    train_indices = np.hstack((train_indices, mesh_id_indices))
            cross_validation_masks[fold]["train_indices"] = train_indices
            cross_validation_masks[fold]["test_indices"] = test_indices

        return cross_validation_masks

    def normalize(self, fit_indices):
        feats_names = self.getSpiderPatchNodeFeatsNames()
        if "vertices" in feats_names:
            feats_names.remove("vertices")
        if "weight" in feats_names:
            feats_names.remove("weight")

        node_normalizers = {}
        for feature in feats_names:
            node_normalizers[feature] = MinMaxScaler((0, 1))

        for mesh_graph in tqdm(self.graphs[fit_indices], position=0, leave=True, desc=f"Normalizer Fitting: ", colour="white", ncols=80):
            for spider_patch in mesh_graph.patches:
                for feature in feats_names:
                    if spider_patch.node_attr_schemes()[feature].shape == ():
                        node_normalizers[feature].partial_fit(spider_patch.ndata[feature].reshape((-1, 1)))
                    else:
                        node_normalizers[feature].partial_fit(spider_patch.ndata[feature])

        for mesh_graph in tqdm(self.graphs, position=0, leave=True, desc=f"Normalizing: ", colour="white", ncols=80):
            for spider_patch in mesh_graph.patches:
                for feature in feats_names:
                    if spider_patch.node_attr_schemes()[feature].shape == ():
                        spider_patch.ndata[feature] = torch.tensor(node_normalizers[feature].transform(spider_patch.ndata[feature].reshape((-1, 1))), dtype=torch.float32)
                    else:
                        spider_patch.ndata[feature] = torch.tensor(node_normalizers[feature].transform(spider_patch.ndata[feature]), dtype=torch.float32)

        return node_normalizers

    def normalizeV2(self, fit_indices, mode, outlier_elim_mode=None):
        feats_names = self.getSpiderPatchNodeFeatsNames()
        if "vertices" in feats_names:
            feats_names.remove("vertices")

        node_normalizers = {}
        for feature in feats_names:
            if mode == "standardization":
                node_normalizers[feature] = StandardScaler()
            elif mode == "normalization":
                node_normalizers[feature] = MinMaxScaler((0, 1))
            elif mode == "robust":
                node_normalizers[feature] = RobustScaler()
            elif mode == "quantile":
                node_normalizers[feature] = QuantileTransformer(output_distribution="normal")
            else:
                raise ()

            train_data = []
            for mesh_graph in tqdm(self.graphs[fit_indices], position=0, leave=True, desc=f"Accumulating Data: ", colour="white", ncols=80):
                for spider_patch in mesh_graph.patches:
                    train_data.extend(spider_patch.ndata[feature].tolist())
            train_data = np.array(train_data)

            if mode == "robust" or mode == "quantile":
                node_normalizers[feature].fit(train_data)
                continue

            if outlier_elim_mode is None:
                for mesh_graph in tqdm(self.graphs[fit_indices], position=0, leave=True, desc=f"Normalizer Fitting: ", colour="white", ncols=80):
                    for spider_patch in mesh_graph.patches:
                        node_normalizers[feature].partial_fit(spider_patch.ndata[feature])
                continue

            for col in range(train_data.shape[1]):
                if outlier_elim_mode == "quantile":
                    lower, upper = IQROutliers(train_data[:, col])
                elif outlier_elim_mode == "standard":
                    lower, upper = SDOutliers(train_data[:, col])
                else:
                    raise ()

                for mesh_graph in tqdm(self.graphs[fit_indices], position=0, leave=True, desc=f"Eliminating Column {col} Outliers: ", colour="white", ncols=80):
                    for spider_patch in mesh_graph.patches:
                        upper_idx = np.where(spider_patch.ndata[feature][:, col] > upper)[0]
                        lower_idx = np.where(spider_patch.ndata[feature][:, col] < lower)[0]
                        if len(upper_idx) > 0:
                            spider_patch.ndata[feature][:, col][upper_idx] = upper
                        if len(lower_idx) > 0:
                            spider_patch.ndata[feature][:, col][lower_idx] = lower

            for mesh_graph in tqdm(self.graphs[fit_indices], position=0, leave=True, desc=f"normalizer Fitting: ", colour="white", ncols=80):
                for spider_patch in mesh_graph.patches:
                    node_normalizers[feature].partial_fit(spider_patch.ndata[feature])

        for mesh_graph in tqdm(self.graphs, position=0, leave=True, desc=f"Normalizing nodes features: ", colour="white", ncols=80):
            for spider_patch in mesh_graph.patches:
                for feature in feats_names:
                    spider_patch.ndata[feature] = torch.tensor(node_normalizers[feature].transform(spider_patch.ndata[feature]), dtype=torch.float32)

        return node_normalizers

    def normalize_nodes(self, node_normalizers):
        feats_names = self.getSpiderPatchNodeFeatsNames()
        if "vertices" in feats_names:
            feats_names.remove("vertices")

        for mesh_graph in tqdm(self.graphs, position=0, leave=True, desc=f"Normalizing nodes features: ", colour="white", ncols=80):
            for spider_patch in mesh_graph.patches:
                for feature in feats_names:
                    spider_patch.ndata[feature] = torch.tensor(node_normalizers[feature].transform(spider_patch.ndata[feature]), dtype=torch.float32)

        return node_normalizers

    def normalize_edge(self, fit_indices):
        edge_normalizers = {}
        feats_names = self.getSpiderPatchEdgeFeatsNames()
        for feature in feats_names:
            edge_normalizers[feature] = MinMaxScaler((0, 1))

        for mesh_graph in tqdm(self.graphs[fit_indices], position=0, leave=True, desc=f"Normalizer Fitting: ", colour="white", ncols=80):
            for spider_patch in mesh_graph.patches:
                for feature in feats_names:
                    if spider_patch.edge_attr_schemes()[feature].shape == ():
                        edge_normalizers[feature].partial_fit(spider_patch.edata[feature].reshape((-1, 1)))
                    else:
                        edge_normalizers[feature].partial_fit(spider_patch.edata[feature])

        for mesh_graph in tqdm(self.graphs, position=0, leave=True, desc=f"Normalizing: ", colour="white", ncols=80):
            for spider_patch in mesh_graph.patches:
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
        self.labels = np.array(loaded_dataset.labels, dtype=np.int64)
        self.sample_id = loaded_dataset.sample_id
        self.mesh_id = loaded_dataset.mesh_id
        self._name = dataset_name

    def fromRawPatches(self, load_path, graph_for_mesh=10, patch_for_graph=10, connection_number=0, only_resolution_level="all", feature_to_keep=None):
        """
        Generate a MeshGraphDataset from a PatchesDataset.

        @param load_path: (string) the path to the SpiderPatches dataset folder structure.
        @param only_resolution_level: (string) e.g. all, level_0, level_1, ecc... The resolution level of the mesh to use.
        @param graph_for_mesh: (int) number of MeshGraph to generate per Mesh.
        @param patch_for_graph: (int) number of Patch to use as nodes per MeshGraph
        @param connection_number: (int) number of neighbours per patch, leave zero for a fully connected graph
        @param feature_to_keep: (list) A list of node features keys to keep, if None keep all features
        @return: self
        """

        print(f"Loading MeshGraph Dataset from: {load_path}")
        import re
        rng = np.random.default_rng(22)
        self.graphs = np.empty(0)
        self.labels = np.empty(0, dtype=np.int64)
        self.sample_id = np.empty(0)
        self.mesh_id = np.empty(0)

        for label in tqdm(os.listdir(f"{load_path}"), position=0, desc=f"SpiderPatch Class Loading: ", colour="white", ncols=80):
            for sample_id in os.listdir(f"{load_path}/{label}"):
                if os.path.isdir(f"{load_path}/{label}/{sample_id}"):
                    for resolution_level in os.listdir(f"{load_path}/{label}/{sample_id}"):
                        if resolution_level.removeprefix("resolution_") != only_resolution_level and only_resolution_level != "all":
                            continue
                        patches_filename = os.listdir(f"{load_path}/{label}/{sample_id}/{resolution_level}")[0]
                        with open(f"{load_path}/{label}/{sample_id}/{resolution_level}/{patches_filename}", "rb") as pkl_file:
                            gc.disable()
                            patches = pkl.load(pkl_file)
                            gc.enable()
                            for _ in range(graph_for_mesh):
                                mesh_id = [int(s) for s in re.findall(r'\d+', patches_filename)]
                                self.graphs = np.append(self.graphs,
                                                        MeshGraph(rng.choice(patches, patch_for_graph, replace=False),
                                                                  sample_id=int(sample_id.removeprefix("id_")),
                                                                  mesh_id=int(mesh_id[0]),
                                                                  resolution_level=resolution_level.removeprefix("resolution_"),
                                                                  neighbours_number=connection_number,
                                                                  keep_feats_names=feature_to_keep))
                                self.labels = np.append(self.labels, int(label.removeprefix("class_")))
                                self.sample_id = np.append(self.sample_id, int(sample_id.removeprefix("id_")))
                                self.mesh_id = np.append(self.mesh_id, mesh_id)
                else:
                    patches_path = f"{load_path}/{label}/{sample_id}"
                    int_sample_id = None
                    resolution_level = None

                    mesh_id = [int(s) for s in re.findall(r'\d+', patches_path.split(sep="/")[-1])]

                    with open(patches_path, "rb") as pkl_file:
                        gc.disable()
                        patches = pkl.load(pkl_file)
                        gc.enable()
                        for _ in range(graph_for_mesh):
                            self.graphs = np.append(self.graphs,
                                                    MeshGraph(rng.choice(patches, patch_for_graph, replace=False),
                                                              sample_id=int_sample_id, mesh_id=int(mesh_id[0]),
                                                              resolution_level=resolution_level,
                                                              neighbours_number=connection_number,
                                                              keep_feats_names=feature_to_keep))
                            self.labels = np.append(self.labels, int(label.removeprefix("class_")))
                            self.sample_id = np.append(self.sample_id, int_sample_id)
                            self.mesh_id = np.append(self.mesh_id, mesh_id)

        # if resolution_level != "all":
        #     for label in tqdm(os.listdir(f"{load_path}"), position=0, desc=f"Mesh Class Loading: ", colour="white", ncols=80):
        #         for sample_id in os.listdir(f"{load_path}/{label}"):
        #             patches_filename = os.listdir(f"{load_path}/{label}/{sample_id}/resolution_{resolution_level}")[0]
        #             with open(f"{load_path}/{label}/{sample_id}/resolution_{resolution_level}/{patches_filename}", "rb") as pkl_file:
        #                 gc.disable()
        #                 patches = pkl.load(pkl_file)
        #                 gc.enable()
        #                 for _ in range(graph_for_mesh):
        #                     mesh_id = [int(s) for s in re.findall(r'\d+', patches_filename)]
        #                     self.graphs = np.append(self.graphs,
        #                                             MeshGraph(rng.choice(patches, patch_for_graph, replace=False), sample_id=int(sample_id.removeprefix("id_")), mesh_id=int(mesh_id[0]), resolution_level=resolution_level.removeprefix("resolution_"), neighbours_number=connection_number,
        #                                                       keep_feats_names=feature_to_keep))
        #                     self.labels = np.append(self.labels, int(label.removeprefix("class_")))
        #                     self.sample_id = np.append(self.sample_id, int(sample_id.removeprefix("id_")))
        #                     self.mesh_id = np.append(self.mesh_id, mesh_id)
        # else:
        #     for label in tqdm(os.listdir(f"{load_path}"), position=0, desc=f"Mesh Class Loading: ", colour="white", ncols=80):
        #         for sample_id in tqdm(os.listdir(f"{load_path}/{label}"), position=1, desc=f"Sample loading: ", colour="white", ncols=80, leave=False):
        #             for resolution_level in os.listdir(f"{load_path}/{label}/{sample_id}"):
        #                 patches_filename = os.listdir(f"{load_path}/{label}/{sample_id}/{resolution_level}")[0]
        #                 with open(f"{load_path}/{label}/{sample_id}/{resolution_level}/{patches_filename}", "rb") as pkl_file:
        #                     gc.disable()
        #                     patches = pkl.load(pkl_file)
        #                     gc.enable()
        #                     for _ in range(graph_for_mesh):
        #                         mesh_id = [int(s) for s in re.findall(r'\d+', patches_filename)]
        #                         self.graphs = np.append(self.graphs,
        #                                                 MeshGraph(rng.choice(patches, patch_for_graph, replace=False), sample_id=int(sample_id.removeprefix("id_")), mesh_id=int(mesh_id[0]), resolution_level=resolution_level.removeprefix("resolution_"), neighbours_number=connection_number,
        #                                                           keep_feats_names=feature_to_keep))
        #                         self.labels = np.append(self.labels, int(label.removeprefix("class_")))
        #                         self.sample_id = np.append(self.sample_id, int(sample_id.removeprefix("id_")))
        #                         self.mesh_id = np.append(self.mesh_id, mesh_id)

        super().__init__(name=self.name)
        return self

    def fromRawPatchesGeo(self, load_path, graph_for_mesh=10, patch_for_graph=10, connection_number=0, only_resolution_level="all", feature_to_keep=None):
        """
        Generate a MeshGraphDataset from a PatchesDataset.

        @param load_path: (string) the path to the SpiderPatches dataset folder structure.
        @param only_resolution_level: (string) e.g. all, level_0, level_1, ecc... The resolution level of the mesh to use.
        @param graph_for_mesh: (int) number of MeshGraph to generate per Mesh.
        @param patch_for_graph: (int) number of Patch to use as nodes per MeshGraph
        @param connection_number: (int) number of neighbours per patch, leave zero for a fully connected graph
        @param feature_to_keep: (list) A list of node features keys to keep, if None keep all features
        @return: self
        """

        print(f"Loading MeshGraph Dataset from: {load_path}")

        rng = np.random.default_rng(717)
        self.graphs = []
        self.labels = []
        self.sample_id = []
        self.mesh_id = []

        partial_dataset_names = []
        labels = os.listdir(f"{load_path}")
        NUM_THREADS = 6
        class_for_thread = int(len(labels) / NUM_THREADS) + 1

        pool = multiprocessing.Pool(processes=NUM_THREADS)
        for result in pool.starmap(extractPartial, [(labels[i * class_for_thread: (i * class_for_thread) + class_for_thread], load_path, rng, graph_for_mesh, patch_for_graph, connection_number, only_resolution_level, feature_to_keep, i) for i in range(NUM_THREADS)]):
            partial_dataset_names.append(result)

        for dataset_name in partial_dataset_names:
            partial_dataset = MeshGraphDataset(dataset_name=dataset_name)
            partial_dataset.load_from(f"Datasets/PartialMeshGraphs", dataset_name)
            self.graphs = np.append(self.graphs, partial_dataset.graphs)
            self.labels = np.append(self.labels, partial_dataset.labels)
            self.sample_id = np.append(self.sample_id, partial_dataset.sample_id)
            self.mesh_id = np.append(self.mesh_id, partial_dataset.mesh_id)

        super().__init__(name=self.name)

        shutil.rmtree(f"Datasets/PartialMeshGraphs")
        return


def extractPartial(labels_to_extract, load_path, rng, graph_for_mesh=10, patch_for_graph=10, connection_number=0, only_resolution_level="all", feature_to_keep=None, pos=0):
    partial_graphs = np.empty(0)
    partial_labels = np.empty(0, dtype=np.int64)
    partial_sample_id = np.empty(0)
    partial_mesh_id = np.empty(0)
    sleep(rng.random())
    for label in tqdm(labels_to_extract, position=pos, desc=f"SpiderPatch Class Loading: ", colour="white", ncols=80):
        for sample_id in os.listdir(f"{load_path}/{label}"):
            if os.path.isdir(f"{load_path}/{label}/{sample_id}"):
                for resolution_level in os.listdir(f"{load_path}/{label}/{sample_id}"):
                    if resolution_level.removeprefix("resolution_") != only_resolution_level and only_resolution_level != "all":
                        continue
                    patches_filename = os.listdir(f"{load_path}/{label}/{sample_id}/{resolution_level}")[0]
                    mesh_path = f'Datasets/Meshes/{os.path.basename(load_path).split("_")[0]}/{sample_id}/{resolution_level}/{patches_filename.replace("spiderPatches", "mesh")}'
                    with open(mesh_path, "rb") as mesh_file:
                        mesh = pickle.load(mesh_file)
                        mesh.to_networkX()

                    with open(f"{load_path}/{label}/{sample_id}/{resolution_level}/{patches_filename}", "rb") as pkl_file:
                        gc.disable()
                        patches = pkl.load(pkl_file)
                        gc.enable()
                        for _ in range(graph_for_mesh):
                            mesh_id = [int(s) for s in re.findall(r'\d+', patches_filename)]
                            partial_graphs = np.append(partial_graphs,
                                                       MeshGraphGeo(rng.choice(patches, patch_for_graph, replace=False),
                                                                    mesh,
                                                                    sample_id=int(sample_id.removeprefix("id_")),
                                                                    mesh_id=int(mesh_id[0]),
                                                                    resolution_level=resolution_level.removeprefix("resolution_"),
                                                                    neighbours_number=connection_number,
                                                                    keep_feats_names=feature_to_keep))
                            partial_labels = np.append(partial_labels, int(label.removeprefix("class_")))
                            partial_sample_id = np.append(partial_sample_id, int(sample_id.removeprefix("id_")))
                            partial_mesh_id = np.append(partial_mesh_id, mesh_id)
            else:
                patches_path = f"{load_path}/{label}/{sample_id}"
                mesh_path = f'Datasets/Meshes/{os.path.basename(load_path).split("_")[0]}/{label}/{sample_id.replace("concRing", "mesh")}'
                with open(mesh_path, "rb") as mesh_file:
                    mesh = pickle.load(mesh_file)
                    mesh.to_networkX()

                int_sample_id = None
                resolution_level = None

                mesh_id = [int(s) for s in re.findall(r'\d+', patches_path.split(sep="/")[-1])]

                with open(patches_path, "rb") as pkl_file:
                    gc.disable()
                    patches = pkl.load(pkl_file)
                    gc.enable()
                    for _ in range(graph_for_mesh):
                        partial_graphs = np.append(partial_graphs,
                                                   MeshGraphGeo(rng.choice(patches, patch_for_graph, replace=False),
                                                                mesh,
                                                                sample_id=int_sample_id, mesh_id=int(mesh_id[0]),
                                                                resolution_level=resolution_level,
                                                                neighbours_number=connection_number,
                                                                keep_feats_names=feature_to_keep))
                        partial_labels = np.append(partial_labels, int(label.removeprefix("class_")))
                        partial_sample_id = np.append(partial_sample_id, int_sample_id)
                        partial_mesh_id = np.append(partial_mesh_id, mesh_id)
    dataset_name = f"PartialDataset{pos}_{int(time())}"
    dataset = MeshGraphDataset(dataset_name, partial_graphs, partial_labels, partial_sample_id, partial_mesh_id)
    os.makedirs(f"Datasets/PartialMeshGraphs", exist_ok=True)
    dataset.save_to(f"Datasets/PartialMeshGraphs")
    return dataset_name

    def fromRawSuperPatches(self, load_path, graph_for_mesh=10, patch_for_graph=10, connection_number=0, feature_to_keep=None):
        """
        Generate a MeshGraphDataset from a PatchesDataset.

        @param load_path: (string) the path to the SpiderPatches dataset folder structure.
        @param resolution_level: (string) e.g. all, level_0, level_1, ecc... The resolution level of the mesh to use.
        @param graph_for_mesh: (int) number of MeshGraph to generate per Mesh.
        @param patch_for_graph: (int) number of Patch to use as nodes per MeshGraph
        @param connection_number: (int) number of neighbours per patch, leave zero for a fully connected graph
        @param feature_to_keep: (list) A list of node features keys to keep, if None keep all features
        @return: self
        """

        print(f"Loading MeshGraph Dataset from: {load_path}")
        import re
        rng = np.random.default_rng(22)
        self.graphs = []
        self.labels = []
        self.sample_id = []
        self.mesh_id = []
        for label in tqdm(os.listdir(f"{load_path}"), position=0, desc=f"Mesh Class Loading: ", colour="white", ncols=80):
            for sample_id in tqdm(os.listdir(f"{load_path}/{label}"), position=1, desc=f"Sample loading: ", colour="white", ncols=80, leave=False):
                patches_filename = os.listdir(f"{load_path}/{label}/{sample_id}")[0]
                with open(f"{load_path}/{label}/{sample_id}/{patches_filename}", "rb") as pkl_file:
                    gc.disable()
                    patches = pkl.load(pkl_file)
                    gc.enable()
                    for _ in range(graph_for_mesh):
                        mesh_id = [int(s) for s in re.findall(r'\d+', patches_filename)]
                        self.graphs.append(MeshGraph(rng.choice(patches, patch_for_graph, replace=False), sample_id=int(sample_id.removeprefix("id_")), mesh_id=int(mesh_id[0]), neighbours_number=connection_number, keep_feats_names=feature_to_keep))
                        self.labels.append(int(label.removeprefix("class_")))
                        self.sample_id.append(int(sample_id.removeprefix("id_")))
                        self.mesh_id.extend(mesh_id)

        super().__init__(name=self.name)
        return self
