import warnings
from sklearn.model_selection import train_test_split
from dgl.data import DGLDataset
import os
import pickle as pkl
import torch
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler

from SHREC_Utils import subdivide_for_mesh





class PatchDataset(DGLDataset):
    def __init__(self, dataset_name="", graphs=None, labels=None):
        self.graphs = graphs
        self.labels = labels
        super().__init__(name=dataset_name)

    def fromDiskRawData(self, load_path, resolution_level="all", save_dir=None):
        self.graphs = []
        self.labels = []
        if resolution_level != "all":
            for mesh in subdivide_for_mesh():
                mesh_id, label = mesh[resolution_level]
                with open(f"{load_path}/{label}/patches{mesh_id}.pkl", "rb") as pkl_file:
                    patches = pkl.load(pkl_file)
                    self.graphs.extend(patches)
                    self.labels.extend([int(label)] * len(patches))
        else:
            for class_label in os.listdir(load_path):
                for graph_pkl in os.listdir(f"{load_path}/{class_label}"):
                    with open(f"{load_path}/{class_label}/{graph_pkl}", "rb") as pkl_file:
                        patches = pkl.load(pkl_file)
                        self.graphs.extend(patches)
                        self.labels.extend([int(class_label)] * len(patches))
        super().__init__(name=self.name, save_dir=save_dir)

    def process(self):
        # Convert the label list to tensor for saving.
        if self.labels is not None:
            self.labels = torch.LongTensor(self.labels)
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
        for idx in range(len(self.graphs)):
            self.graphs[idx] = self.graphs[idx].to(device)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.labels = torch.tensor(self.labels, device=device, dtype=torch.int64)

    def normalize(self):  # TODO normalizzare solo sui dati di training
        # Normalize node data
        node_normalizers = {}
        for feature in self.getNodeFeatsName():
            node_normalizers[feature] = StandardScaler()

        """
        edge_normalizers = {}
        for feature in self.getEdgeFeatsNames():
            edge_normalizers[feature] = MaxAbsScaler()#MinMaxScaler((1, 2))
        """

        for i in range(len(self.graphs)):
            graph = self.graphs[i]
            for feature in self.getNodeFeatsName():
                if graph.node_attr_schemes()[feature].shape == ():
                    node_normalizers[feature].partial_fit(graph.ndata[feature].reshape((-1, 1)))
                else:
                    node_normalizers[feature].partial_fit(graph.ndata[feature])
            """
            for feature in self.getEdgeFeatsNames():
                if graph.edge_attr_schemes()[feature].shape == ():
                    edge_normalizers[feature].partial_fit(graph.edata[feature].reshape((-1, 1)))
                else:
                    edge_normalizers[feature].partial_fit(graph.edata[feature])
            """
        for graph in self.graphs:
            for feature in self.getNodeFeatsName():
                if graph.node_attr_schemes()[feature].shape == ():
                    graph.ndata[feature] = torch.tensor(node_normalizers[feature].transform(graph.ndata[feature].reshape((-1, 1))), dtype=torch.float32)
                else:
                    graph.ndata[feature] = torch.tensor(node_normalizers[feature].transform(graph.ndata[feature]), dtype=torch.float32)
            """
            for feature in self.getEdgeFeatsNames():
                if graph.edge_attr_schemes()[feature].shape == ():
                    graph.edata[feature] = torch.tensor(edge_normalizers[feature].transform(graph.edata[feature].reshape((-1, 1))), dtype=torch.float32)
                else:
                    graph.edata[feature] = torch.tensor(edge_normalizers[feature].transform(graph.edata[feature]), dtype=torch.float32)
            """

    def save_to(self, save_path=None):
        os.makedirs(save_path, exist_ok=True)

        if save_path is not None:
            path = f"{save_path}/{self.name}.pkl"
        else:
            path = f"{self.save_path}/{self.name}.pkl"

        with open(path, "wb") as dataset_file:
            pkl.dump(self, dataset_file)

    def load_from(self, load_path=None, dataset_name=None):
        if load_path is not None:
            path = f"{load_path}/{self.name}.pkl"
        else:
            path = f"{self.save_path}/{self.name}.pkl"

        with open(path, "rb") as dataset_file:
            loaded_dataset = pkl.load(dataset_file)
        self.graphs = loaded_dataset.graphs
        self.labels = loaded_dataset.labels


class PatchDatasetForNNTraining:
    def __init__(self, dgl_dataset_path=None, dataset_name=""):
        self.normalized = False
        self.test_dataset = None
        self.validation_dataset = None
        self.train_dataset = None
        self.name = dataset_name
        if dgl_dataset_path is not None:
            self.loadFromDGLDataset(dgl_dataset_path=dgl_dataset_path, dgl_dataset_name=dataset_name)

    def loadFromDGLDataset(self, dgl_dataset_path, dgl_dataset_name):
        dataset = PatchDataset(dataset_name=dgl_dataset_name)
        dataset.load_from(dgl_dataset_path)
        self.name = dataset.name
        self.split_graphs(dataset.graphs, dataset.labels)

    def split_graphs(self, graphs, labels):
        data_train, data_test, label_train, label_test = train_test_split(graphs, labels, train_size=0.7, test_size=0.3, random_state=22, shuffle=True)
        self.train_dataset = PatchDataset(dataset_name=self.name + "_train", graphs=data_train, labels=label_train)
        data_validation, data_test, label_validation, label_test = train_test_split(data_test, label_test, train_size=0.30, test_size=0.70, random_state=23, shuffle=True)
        self.validation_dataset = PatchDataset(dataset_name=self.name + "_val", graphs=data_validation, labels=label_validation)
        self.test_dataset = PatchDataset(dataset_name=self.name + "_test", graphs=data_test, labels=label_test)

    def loadFromRawPatchesDataset(self, patch_radius, rings_number, point_per_ring, Normalized=False):
        if Normalized:
            self.name = f"SHREC17_R{patch_radius}_RI{rings_number}_P{point_per_ring}_Normalized"
        else:
            self.name = f"SHREC17_R{patch_radius}_RI{rings_number}_P{point_per_ring}"

        self.normalized = Normalized

        self.base_path = "./Datasets/Patches/" + self.name
        os.makedirs(self.base_path, exist_ok=True)
        graphs = []
        labels = []

        for label_class in os.listdir(self.base_path):
            for patch_graph_pkl in os.listdir(f"{self.base_path}/{label_class}"):
                with open(f"{self.base_path}/{label_class}/{patch_graph_pkl}", "rb") as pkl_file:
                    patches = pkl.load(pkl_file)
                    graphs.extend(patches)
                    labels.extend([int(label_class)] * len(patches))

        # Split the dataset in TRAINING (70%), VALIDATION(10%) and TEST(20%)
        self.split_graphs(graphs, labels)

    def __len__(self):
        return len(self.train_dataset) + len(self.validation_dataset) + len(self.test_dataset)

    def getNodeFeatsNames(self):
        return self.train_dataset.graphs[0].getNodeFeatsNames()

    def getEdgeFeatsNames(self):
        return self.train_dataset.graphs[0].getEdgeFeatsNames()

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