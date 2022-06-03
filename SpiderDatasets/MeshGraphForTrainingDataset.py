import random
import os
import numpy as np
import torch
from SpiderDatasets.MeshGraphDataset import MeshGraphDataset


class MeshGraphDatasetForNNTraining:
    """
    A class that manages 3 MeshGraphDataset (the training, validation and test datasets) for AI training
    """

    def __init__(self):
        self.normalizers = None
        self.test_dataset = None
        self.validation_dataset = None
        self.train_dataset = None
        self.name = ""

    def __len__(self):
        return len(self.train_dataset) + len(self.validation_dataset) + len(self.test_dataset)

    def split_graphs(self, graphs, labels, mesh_id):
        a = id(graphs[0])
        random.seed(22)
        data_train, label_train = [], []
        data_test, label_test = [], []
        data_validation, label_validation = [], []
        train_indices, test_indices, validation_indices = [], [], []
        to_train, to_test = [], []
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mesh_idx = np.unique(np.array(mesh_id)[np.where(labels == label)])
            train = random.sample(list(mesh_idx), int(len(mesh_idx) * 0.80))
            test = list(set(mesh_idx) - set(train))
            validation = random.sample(list(test), int(len(test) * 0.50))
            test = list(set(test) - set(validation))
            to_train.extend(train)
            to_test.extend(test)

        to_train = set(to_train)
        to_test = set(to_test)

        for i, mesh in enumerate(mesh_id):
            if mesh in to_train:
                data_train.append(graphs[i])
                label_train.append(labels[i])
                train_indices.append(mesh)
            elif mesh in to_test:
                data_test.append(graphs[i])
                label_test.append(labels[i])
                test_indices.append(mesh)
            else:
                data_validation.append(graphs[i])
                label_validation.append(labels[i])
                validation_indices.append(mesh)

        combined = list(zip(data_train, label_train, train_indices))
        random.shuffle(combined)
        data_train[:], label_train[:], train_indices[:] = zip(*combined)

        for i, graph in enumerate(data_train):
            if id(graph) == a:
                b = i

        self.train_dataset = MeshGraphDataset(dataset_name=self.name + "_train", graphs=data_train, labels=label_train, mesh_id=train_indices)
        self.test_dataset = MeshGraphDataset(dataset_name=self.name + "_test", graphs=data_test, labels=label_test, mesh_id=test_indices)
        self.validation_dataset = MeshGraphDataset(dataset_name=self.name + "_val", graphs=data_validation, labels=label_validation, mesh_id=validation_indices)

    def getNodeFeatsNames(self):
        return self.train_dataset.graphs[0].getNodeFeatsNames()

    def getEdgeFeatsNames(self):
        return self.train_dataset.graphs[0].getEdgeFeatsNames()

    def getPatchNodeFeatsNames(self):
        return self.train_dataset.patches[0].getNodeFeatsNames()

    def getPatchEdgeFeatsNames(self):
        return self.train_dataset.patches[0].getEdgeFeatsNames()

    def numClasses(self, partition="all"):
        """
        :param partition: a string to select the partition dataset to transfer to device: "all" - all the dataset
                                                                                            "train" - train dataset
                                                                                            "val" - validation dataset
                                                                                            "test" - test dataset
        :return:
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
        :param device:
        :param partition: a string to select the partition dataset to transfer to device: "all" - all the dataset
                                                                                            "train" - train dataset
                                                                                            "val" - validation dataset
                                                                                            "test" - test dataset
        :return: None
        """
        if partition == "train" or partition == "all":
            self.train_dataset.to(device)
        if partition == "val" or partition == "all":
            self.validation_dataset.to(device)
        if partition == "test" or partition == "all":
            self.test_dataset.to(device)

    def aggregateNodeFeatures(self, feat_names="all"):
        """

        :param feat_names: list of features key to aggregate in a single feature (called aggregate_feature)
        :return: the aggregate feature shape along axis 1 ( if shape 30x4 --> returns 4 )
        """
        if feat_names == "all":
            feat_names = self.getNodeFeatsNames()

        for dataset in [self.train_dataset, self.validation_dataset, self.test_dataset]:
            for i in range(len(dataset)):
                dataset[i][0].ndata["aggregated_feats"] = dataset[i][0].ndata[feat_names[0]]
                for name in feat_names[1:]:
                    if dataset.graphs[0].node_attr_schemes()[name].shape == ():
                        dataset[i][0].ndata["aggregated_feats"] = torch.hstack((dataset[i][0].ndata["aggregated_feats"], torch.reshape(dataset[i][0].ndata[name], (-1, 1))))
                    else:
                        dataset[i][0].ndata["aggregated_feats"] = torch.hstack((dataset[i][0].ndata["aggregated_feats"], dataset[i][0].ndata[name]))
        return self.train_dataset[0][0].ndata["aggregated_feats"].shape

    def aggregateEdgeFeatures(self, feat_names="all"):
        """

        :param feat_names: list of features key to aggregate in a single feature (called aggregate_feature)
        :return: the aggregate feature shape along axis 1 ( if shape 30x4 --> returns 4 )
        """
        if feat_names == "all":
            feat_names = self.getEdgeFeatsNames()

        for dataset in [self.train_dataset, self.validation_dataset, self.test_dataset]:
            for i in range(len(dataset)):
                dataset[i][0].edata["weights"] = dataset[i][0].edata[feat_names[0]]
                for name in feat_names[1:]:
                    if dataset.graphs[0].edge_attr_schemes()[name].shape == ():
                        dataset[i][0].edata["weights"] = torch.hstack((dataset[i][0].edata["weights"], torch.reshape(dataset[i][0].edata[name], (-1, 1))))
                    else:
                        dataset[i][0].edata["weights"] = torch.hstack((dataset[i][0].edata["weights"], dataset[i][0].edata[name]))

        return self.train_dataset[0][0].edata["weights"].shape

    def normalize(self):
        """
        Normalizes the training and validation datasets using their values and stores the normalizers for later use.
        :return: (dict) e.g. { "feat_name":sklearn.preprocessing.Scaler }, the normalizers calculated in the process.
        """
        self.name = self.name + "_Normalized"
        self.normalizers = self.train_dataset.normalize()
        return self.normalizers

    def normalize_test_dataset(self):
        """
        Uses the normalizers in self.normalizers to normalize the test dataset.
        :return:
        """
        if self.normalizers is not None:
            self.test_dataset.normalize(self.normalizers)

    def normalize_validation_dataset(self):
        """
        Uses the normalizers in self.normalizers to normalize the test dataset.
        :return:
        """
        if self.normalizers is not None:
            self.validation_dataset.normalize(self.normalizers)

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
        :param path: (string) The path to the dataset folder.
        :param name: (string) The base name of the dataset that follow the standard.
        :param train_name: (string) Use it if the training dataset doesn't follow the standard name system.
        :param validation_name:(string) Use it if the validation dataset doesn't follow the standard name system.
        :param test_name:(string) Use it if the test dataset doesn't follow the standard name system.
        :return:
        """
        self.train_dataset = MeshGraphDataset()
        self.validation_dataset = MeshGraphDataset()
        self.test_dataset = MeshGraphDataset()

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

    def loadFromMeshGraphDataset(self, dataset_path, dataset_name):
        """
        Loads a MeshGraphDatasetForNNTraining from a MeshGraphDataset file.
        :param dataset_path: (string) The path to the folder containing the dataset file.
        :param dataset_name: (string) The dataset file name.
        :return:
        """
        dataset = MeshGraphDataset()
        dataset.load_from(dataset_path, dataset_name)
        self.name = dataset.name
        self.split_graphs(dataset.graphs, dataset.labels, dataset.mesh_id)