import itertools
import os
import pickle as pkl
import random
import re
import sys
import warnings
from time import time

import open3d as o3d
import numpy as np
import torch
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from scipy.special import softmax
from sklearn import metrics
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from Networks.BLOCKCONVNetworks import BlockMeshNetwork
from PlotUtils import save_confusion_matrix, plot_training_statistics, plot_grad_flow, plot_voting_confidences
from SpiderDatasets.MeshGraphDataset import MeshGraphDataset


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_name = "SHREC17_R5_R4_P6_CSIRSv2Spiral_MGRAPHS50_SPIDER50_CONN5_RES3"

    features = np.array(["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"])
    features_to_keep = [3, 4]
    radius_to_keep = [0, 1, 2, 3, 4]

    dataset = MeshGraphDataset(dataset_name=dataset_name)
    dataset.load_from(f"../Datasets/MeshGraphs", dataset_name)

    dataset.keepCurvaturesResolution(radius_to_keep)
    dataset.aggregateSpiderPatchesNodeFeatures(features[features_to_keep])
    dataset.aggregateSpiderPatchEdgeFeatures()
    dataset.removeNonAggregatedFeatures(["vertices"])

    train_mask, test_mask = dataset.getTrainTestMask(10, percentage=False)
    class_num = len(np.unique(dataset.labels[train_mask]))
    dataset.normalize(train_mask)
    dataset.normalize_edge(train_mask)
    print(dataset.graphs[0].patches[0].node_attr_schemes())

    feat_dim = dataset.graphs[0].patches[0].ndata["aggregated_feats"].shape[1]
    readout_dim = feat_dim * dataset.graphs[0].patches[0].num_nodes()
    hidden_dim = 64

    epochs = 50

    model = BlockMeshNetwork(feat_dim, hidden_dim=hidden_dim, block_dim=5, out_feats=class_num, dropout=0, block_reduction="average", mesh_graph_edge_weights=True)
    model.load("U:\AssegnoDiRicerca\PythonProject\TrainedModels\Train_SHREC17_R5_R4_P6_CSIRSv2Spiral_MGRAPHS50_SPIDER50_CONN5_RES3_READY_BlockCONVMeshNetwork_K2LD_AVERAGEBLOCKMODEL_GOODPARAM\MeshNetworkBestAcc/network.pt")
    visualizeErrorsMeshNetworkBLOCK(model, dataset, test_mask, 5, "cpu")


def visualizeErrorsMeshNetworkBLOCK(model, dataset, test_mask, block_size, device):
    model.eval()
    predicted_labels = np.empty(0)
    loss_confidences = np.empty((0, dataset.numClasses()))
    with torch.no_grad():
        for i in range(0, len(dataset.graphs[test_mask]), block_size):
            block = [(dataset.graphs[test_mask][i + j], dataset.graphs[test_mask][i + j].patches) for j in range(block_size)]
            mesh_graphs = [tup[0] for tup in block]
            patches_list = [tup[1] for tup in block]
            pred = model(mesh_graphs, patches_list, device)
            pred = pred.cpu()
            predicted_labels = np.hstack((predicted_labels, pred.argmax(dim=0)))  # Take the highest value in the predicted classes vector
            loss_confidences = np.vstack((loss_confidences, pred.detach().numpy()))
            if predicted_labels[-1] != dataset.labels[test_mask][i]:
                print(f"Mesh misclassified with class {predicted_labels[-1]}, should be {int(dataset.labels[test_mask][i])}!")
                with open(f"../Datasets/Meshes/SHREC17/class_{int(dataset.labels[test_mask][i])}/id_{dataset.graphs[test_mask][i].sample_id}/resolution_{dataset.graphs[test_mask][i].resolution_level}/mesh{dataset.graphs[test_mask][i].mesh_id}.pkl", "rb") as file:
                    mesh = pkl.load(file)
                to_draw = mesh.drawWithMeshGraphs(mesh_graphs, return_to_draw=True)
                misclassified_id = os.listdir(f"../Datasets/Meshes/SHREC17/class_{int(predicted_labels[-1])}")[:1][0]
                mesh_filename = os.listdir(f"../Datasets/Meshes/SHREC17/class_{int(predicted_labels[-1])}/{misclassified_id}/resolution_level_3")[0]
                with open(f"../Datasets/Meshes/SHREC17/class_{int(predicted_labels[-1])}/{misclassified_id}/resolution_level_3/{mesh_filename}", "rb") as file:
                    misclassified_mesh = pkl.load(file)
                to_draw.append(misclassified_mesh.draw(return_to_draw=True).translate((mesh.oriented_bounding_box["extent"][0], mesh.oriented_bounding_box["extent"][1], 0)))
                o3d.visualization.draw_geometries(to_draw, mesh_show_back_face=True)
        model.train()
        true_mesh_labels = torch.tensor([dataset.labels[test_mask][i] for i in range(0, len(dataset.graphs[test_mask]), block_size)])
        return compute_scores(true_mesh_labels, predicted_labels, loss_confidences)
