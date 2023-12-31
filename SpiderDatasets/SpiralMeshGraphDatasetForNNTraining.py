import random
import os
import numpy as np
import torch
from tqdm import tqdm

from SpiderDatasets.SpiralMehGraphDataset import SpiralMeshGraphDataset


class SpiralMeshGraphDatasetForNNTraining:
    """
    A class that manages 3 MeshGraphDataset (the training, validation and test datasets) for AI training
    """

    def __init__(self):
        self.normalizers = None
        self.edge_normalizers = None
        self.test_dataset = None
        self.validation_dataset = None
        self.train_dataset = None
        self.name = ""

    def __len__(self):
        return len(self.train_dataset) + len(self.validation_dataset) + len(self.test_dataset)

    def split_dataset(self, dataset, seed=22, to_train=None, to_test=None):
        rng = np.random.default_rng(seed)
        data_train, label_train = [], []
        data_test, label_test = [], []
        data_validation, label_validation = [], []
        sample_idx_train, sample_idx_test, sample_idx_validation = [], [], []
        mesh_idx_train, mesh_idx_test, mesh_idx_validation = [], [], []
        unique_labels = np.unique(dataset.labels)

        if to_train is None and to_test is None:
            to_train, to_test = [], []
            for label in unique_labels:
                sample_idx = np.unique(np.array(dataset.sample_id)[np.where(dataset.labels == label)])
                train = rng.choice(list(sample_idx), int(len(sample_idx) * 0.80), replace=False)
                test = list(set(sample_idx) - set(train))
                validation = rng.choice(list(test), int(len(test) * 0.50), replace=False)
                test = list(set(test) - set(validation))
                to_train.extend(train)
                to_test.extend(test)
        elif to_train is None:
            to_train = []
            for label in unique_labels:
                sample_idx = np.unique(np.array(dataset.sample_id)[np.where(dataset.labels == label)])
                sample_idx = list(set(sample_idx) - set(to_test))
                train = rng.choice(list(sample_idx), int(len(sample_idx) * 0.90), replace=False)
                to_train.extend(train)
        elif to_test is None:
            to_test = []
            for label in unique_labels:
                sample_idx = np.unique(np.array(dataset.sample_id)[np.where(dataset.labels == label)])
                sample_idx = list(set(sample_idx) - set(to_train))
                test = rng.choice(list(sample_idx), int(len(sample_idx) * 0.50), replace=False)
                to_train.extend(test)

        to_train = set(to_train)
        to_test = set(to_test)

        for i, sample_idx in enumerate(dataset.sample_id):
            if sample_idx in to_train:
                data_train.append(dataset.graphs[i])
                label_train.append(dataset.labels[i])
                sample_idx_train.append(sample_idx)
                mesh_idx_train.append(dataset.mesh_id[i])
            elif sample_idx in to_test:
                data_test.append(dataset.graphs[i])
                label_test.append(dataset.labels[i])
                sample_idx_test.append(sample_idx)
                mesh_idx_test.append(dataset.mesh_id[i])
            else:
                data_validation.append(dataset.graphs[i])
                label_validation.append(dataset.labels[i])
                sample_idx_validation.append(sample_idx)
                mesh_idx_validation.append(dataset.mesh_id[i])

        combined = list(zip(data_train, label_train, sample_idx_train))
        rng.shuffle(combined)
        data_train[:], label_train[:], sample_idx_train[:] = zip(*combined)

        self.train_dataset = SpiralMeshGraphDataset(dataset_name=self.name + "_train", graphs=data_train, labels=label_train, sample_id=sample_idx_train, mesh_id=mesh_idx_train)
        self.test_dataset = SpiralMeshGraphDataset(dataset_name=self.name + "_test", graphs=data_test, labels=label_test, sample_id=sample_idx_test, mesh_id=mesh_idx_test)
        self.validation_dataset = SpiralMeshGraphDataset(dataset_name=self.name + "_val", graphs=data_validation, labels=label_validation, sample_id=sample_idx_validation, mesh_id=mesh_idx_validation)

    def getNodeFeatsNames(self):
        return self.train_dataset.graphs[0].getNodeFeatsNames()

    def getEdgeFeatsNames(self):
        return self.train_dataset.graphs[0].getEdgeFeatsNames()

    def getPatchNodeFeatsNames(self):
        return self.train_dataset.graphs[0].patches[0].getNodeFeatsNames()

    def getPatchEdgeFeatsNames(self):
        return self.train_dataset.graphs[0].patches[0].getEdgeFeatsNames()

    def numClasses(self, partition="all"):
        """
        @param partition: a string to select the partition dataset to transfer to device: "all" - all the dataset
                                                                                            "train" - train dataset
                                                                                            "val" - validation dataset
                                                                                            "test" - test dataset
        @return:
        """
        if partition == "all":
            return len(torch.unique(torch.hstack((self.train_dataset.labels, self.validation_dataset.labels, self.test_dataset.labels))))
        elif partition == "train":
            return len(torch.unique(self.train_dataset.labels))
        elif partition == "val":
            return len(torch.unique(self.validation_dataset.labels))
        elif partition == "test":
            return len(torch.unique(self.test_dataset))
        else:
            return None

    def to(self, device, partition="all"):
        """
        @param device:
        @param partition: a string to select the partition dataset to transfer to device: "all" - all the dataset
                                                                                            "train" - train dataset
                                                                                            "val" - validation dataset
                                                                                            "test" - test dataset
        @return: None
        """
        if partition == "train" or partition == "all":
            self.train_dataset.to(device)
        if partition == "val" or partition == "all":
            self.validation_dataset.to(device)
        if partition == "test" or partition == "all":
            self.test_dataset.to(device)

    def aggregateNodeFeatures(self, feat_names=None):
        """
        @param feat_names: list of features key to aggregate in a single feature (called aggregate_feature)
        @return: the aggregate feature shape along axis 1 ( if shape 30x4 --> returns 4 )
        """
        if feat_names is None:
            feat_names = self.getPatchNodeFeatsNames()
            feat_names.remove("vertices")

        for dataset in tqdm([self.train_dataset, self.validation_dataset, self.test_dataset]):
            dataset.aggregateNodeFeatures(feat_names)

    def aggregateEdgeFeatures(self, feat_names="all"):
        """
        @param feat_names: list of features key to aggregate in a single feature (called aggregate_feature)
        @return: the aggregate feature shape along axis 1 ( if shape 30x4 --> returns 4 )
        """
        if feat_names == "all":
            feat_names = self.getPatchEdgeFeatsNames()

        for dataset in [self.train_dataset, self.validation_dataset, self.test_dataset]:
            dataset.aggregateSpiderPatchEdgeFeatures(feat_names)

        return self.train_dataset.graphs[0].patches[0].edata["weights"].shape

    def removeNonAggregatedFeatures(self):
        for dataset in [self.train_dataset, self.validation_dataset, self.test_dataset]:
            dataset.removeNonAggregatedFeatures()

    def normalize(self):
        """
        Normalizes the training and validation datasets using their values and stores the normalizers for later use.
        @return: (dict) e.g. { "feat_name":sklearn.preprocessing.Scaler }, the normalizers calculated in the process.
        """
        self.name = self.name + "_Normalized"
        self.normalizers = self.train_dataset.normalize()
        return self.normalizers

    def normalize_test_dataset(self):
        """
        Uses the normalizers in self.normalizers to normalize the test dataset.
        @return:
        """
        if self.normalizers is not None:
            self.test_dataset.normalize(self.normalizers)

    def normalize_validation_dataset(self):
        """
        Uses the normalizers in self.normalizers to normalize the test dataset.
        @return:
        """
        if self.normalizers is not None:
            self.validation_dataset.normalize(self.normalizers)

    def normalize_edge(self):
        """
        Normalizes the training and validation datasets using their values and stores the normalizers for later use.
        @return: (dict) e.g. { "feat_name":sklearn.preprocessing.Scaler }, the normalizers calculated in the process.
        """
        self.edge_normalizers = self.train_dataset.normalize_edge()
        return self.edge_normalizers

    def normalize_test_dataset_edge(self):
        """
        Uses the normalizers in self.edge_normalizers to normalize the test dataset.
        @return:
        """
        if self.edge_normalizers is not None:
            self.test_dataset.normalize_edge(self.edge_normalizers)

    def normalize_validation_dataset_edge(self):
        """
        Uses the normalizers in self.edge_normalizers to normalize the test dataset.
        @return:
        """
        if self.edge_normalizers is not None:
            self.validation_dataset.normalize_edge(self.edge_normalizers)

    def save(self, path, train_name=None, validation_name=None, test_name=None):
        os.makedirs(path, exist_ok=True)
        if train_name is None:
            self.train_dataset.save_to(path)
        else:
            self.train_dataset._name = train_name
            self.train_dataset.save_to(path)
        if validation_name is None:
            self.validation_dataset.save_to(path)
        else:
            self.validation_dataset._name = validation_name
            self.validation_dataset.save_to(path)
        if test_name is None:
            self.test_dataset.save_to(path)
        else:
            self.test_dataset._name = test_name
            self.test_dataset.save_to(path)

    def load(self, path, name, train_name=None, validation_name=None, test_name=None):
        """
        Loads a MeshGraphDatasetForNNTraining instance from a dataset structured folder
        @param path: (string) The path to the dataset folder.
        @param name: (string) The base name of the dataset that follow the standard.
        @param train_name: (string) Use it if the training dataset doesn't follow the standard name system.
        @param validation_name:(string) Use it if the validation dataset doesn't follow the standard name system.
        @param test_name:(string) Use it if the test dataset doesn't follow the standard name system.
        @return:
        """
        self.train_dataset = SpiralMeshGraphDataset()
        self.validation_dataset = SpiralMeshGraphDataset()
        self.test_dataset = SpiralMeshGraphDataset()

        if train_name is None:
            self.train_dataset.load_from(path, name + "_train")
        else:
            self.train_dataset.load_from(path, train_name)

        if validation_name is None:
            self.validation_dataset.load_from(path, name + "_val")
        else:
            self.validation_dataset.load_from(path, validation_name)

        if test_name is None:
            try:
                self.test_dataset.load_from(path, name + "_test")
            except FileNotFoundError:
                print("Test Dataset NOT Found ---- trying without Normalized!")
                self.test_dataset.load_from(path, name.replace("_Normalized", "") + "_test")
        else:
            self.test_dataset.load_from(path, test_name)

    def loadFromSpiralMeshGraphDataset(self, dataset_path, dataset_name):
        """
        Loads a MeshGraphDatasetForNNTraining from a MeshGraphDataset file.
        @param dataset_path: (string) The path to the folder containing the dataset file.
        @param dataset_name: (string) The dataset file name.
        @return:
        """
        dataset = SpiralMeshGraphDataset()
        dataset.load_from(dataset_path, dataset_name)
        self.name = dataset.name
        self.split_dataset(dataset)
