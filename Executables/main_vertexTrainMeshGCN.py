import os
import pickle as pkl
import random
import sys
import warnings
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm

from Networks.BLOCKCONVNetworks import BlockMeshNetworkPEARAR, BlockMeshNetworkPEARUR
from PlotUtils import save_confusion_matrix, plot_training_statistics, plot_grad_flow
from SpiderDatasets.MeshGraphDataset import MeshGraphDataset


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###### TRAIN DI UN MODELLO SU 4 RISOLUZIONI DI UN DATASET  #########

    dataset_name = "SHREC17_R0.05_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER75_CONN5_RES3_READY"

    dataset = MeshGraphDataset(dataset_name=dataset_name)
    dataset.load_from(f"../Datasets/MeshGraphs", dataset_name)

    # set_node_weights(dataset)

    print(dataset.graphs[0].patches[0].node_attr_schemes())
    # dataset.removeClasses([0, 1, 2, 4, 10, 11, 12])
    # normalize_vertices(dataset)
    # dataset.aggregateSpiderPatchesNodeFeatures(["vertices"])
    # dataset.aggregateSpiderPatchEdgeFeatures()
    # dataset.removeNonAggregatedFeatures(["weight"])

    train_mask, test_mask = dataset.getTrainTestMask(10, percentage=False)
    class_num = len(np.unique(dataset.labels[train_mask]))
    dataset.normalize(train_mask)
    dataset.normalize_edge(train_mask)
    print(dataset.graphs[0].patches[0].node_attr_schemes())
    dataset.to(device)

    feat_dim = dataset.graphs[0].patches[0].ndata["aggregated_feats"].shape[1]

    epochs = 50
    block_dim = 5

    model = BlockMeshNetworkPEARUR(feat_dim, block_dim=block_dim, out_feats=class_num, dropout=0, block_reduction="average", mesh_graph_edge_weights=True)
    model.load("U:\AssegnoDiRicerca\PythonProject\TrainedModels\Train_SHREC17_R0.05_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER75_CONN5_RES3_BlockMeshNetworkPEARUR_VERTEX_AVG5BLOCK_PEARUR\MeshNetworkBestAcc/network.pt")
    model.to(device)
    trainMeshNetworkBLOCK(model, dataset, train_mask, test_mask, block_dim, device, epochs, f"{dataset_name}_{model.name}_VERTEX_AVG5BLOCK_PEARUR", debug=False)


def normalize_vertices(dataset):
    last_mesh_id = -1
    for i, graph in tqdm(enumerate(dataset.graphs)):
        class_num = int(dataset.labels[i])
        sample_id = graph.sample_id
        mesh_id = graph.mesh_id
        resolution_level = graph.resolution_level
        if last_mesh_id != mesh_id:
            last_mesh_id = mesh_id
            with open(f"../Datasets/Meshes/SHREC17/class_{class_num}/id_{sample_id}/resolution_{resolution_level}/mesh{mesh_id}.pkl", "rb") as file:
                mesh = pkl.load(file)
            centroid, m = pc_normalize_parameters(mesh.vertices)
        for patch in graph.patches:
            patch.ndata["vertices"] = (patch.ndata["vertices"] - centroid) / m


def pc_normalize_parameters(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    return centroid, m


def trainMeshNetworkBLOCK(model, dataset, train_mask, test_mask, block_size, device, epochs=1000, train_name="", debug=True):
    if not debug:
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.emit_nvtx(enabled=False)
        torch.autograd.profiler.profile(enabled=False)

    model.train()
    best_acc = 0
    best_loss = 1000

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=10e-4)

    start = time()
    train_losses = []
    mesh_val_accuracies = []
    mesh_val_losses = []

    for epoch in range(epochs):
        train_losses.append(0)
        np.random.shuffle(train_mask.reshape((-1, 5)))
        for i in tqdm(range(0, len(dataset.graphs[train_mask]), block_size), position=0, leave=False, desc=f"Epoch {epoch + 1}: ", colour="white", ncols=80):
            block = [(dataset.graphs[train_mask][i + j], dataset.graphs[train_mask][i + j].patches) for j in range(block_size)]
            random.shuffle(block)
            mesh_graphs = [tup[0] for tup in block]
            patches_list = [tup[1] for tup in block]
            optimizer.zero_grad()
            pred = model(mesh_graphs, patches_list, device)
            loss_running = F.cross_entropy(torch.flatten(pred), dataset.labels[train_mask][i])
            train_losses[-1] += loss_running.item()
            loss_running.backward()
            optimizer.step()
        plot_grad_flow(model.named_parameters())

        train_losses[-1] /= len(dataset.graphs[train_mask])
        print(f"Epoch {epoch + 1}/{epochs} ({int(time() - start)}s):"
              f" Epoch Loss={train_losses[-1]:.3f}")

        mesh_class_acc, mesh_class_loss, mesh_class_cm = testMeshNetworkBLOCK(model=model, dataset=dataset, test_mask=test_mask, block_size=block_size, device=device)
        mesh_val_accuracies.append(mesh_class_acc)
        mesh_val_losses.append(mesh_class_loss)

        print(f"Validation Test\n"
              f"Mesh Graph Acc. : {mesh_class_acc}\n"
              f"Mesh GraphLoss.: {mesh_class_loss}")

        if mesh_class_acc > best_acc:
            best_acc = mesh_class_acc
            model.save(f"../TrainedModels/Train_{train_name}/MeshNetworkBestAcc")
            save_confusion_matrix(mesh_class_cm, f"../TrainedModels/Train_{train_name}/MeshNetworkBestAcc", "MeshGraphConfusionMatrix.png")

            with open(f'../TrainedModels/Train_{train_name}/MeshNetworkBestAcc/bestAcc.txt', 'w') as f:
                f.write(f'Acc: {np.trunc(best_acc * 10000) / 100}\n'
                        f'Epoch:{epoch}\n'
                        f'Loss:{mesh_class_loss}')

        if mesh_class_loss < best_loss:
            best_loss = mesh_class_loss
            # model.save(f"../TrainedModels/Train_{train_name}/MeshNetworkBestLoss")
            save_confusion_matrix(mesh_class_cm, f"../TrainedModels/Train_{train_name}/MeshNetworkBestLoss", "MeshGraphConfusionMatrix.png")

        plot_training_statistics(f"../TrainedModels/Train_{train_name}", f"Train_MeshGraph_statistics.png", title=f"Best Acc: {np.trunc(best_acc * 10000) / 100}, Loss: {best_loss}", epochs=range(epoch + 1), losses=train_losses, val_epochs=range(epoch + 1), val_accuracies=mesh_val_accuracies,
                                 val_losses=mesh_val_losses)


def testMeshNetworkBLOCK(model, dataset, test_mask, block_size, device):
    model.eval()
    with torch.no_grad():
        predicted_labels = np.empty(0)
        loss_confidences = np.empty((0, dataset.numClasses()))
        for i in range(0, len(dataset.graphs[test_mask]), block_size):
            block = [(dataset.graphs[test_mask][i + j], dataset.graphs[test_mask][i + j].patches) for j in range(block_size)]
            mesh_graphs = [tup[0] for tup in block]
            patches_list = [tup[1] for tup in block]
            pred = model(mesh_graphs, patches_list, device)
            pred = pred.cpu()
            predicted_labels = np.hstack((predicted_labels, pred.argmax(dim=0)))  # Take the highest value in the predicted classes vector
            loss_confidences = np.vstack((loss_confidences, pred.detach().numpy()))
        model.train()
        true_mesh_labels = torch.tensor([dataset.labels[test_mask][i] for i in range(0, len(dataset.graphs[test_mask]), block_size)])
        return compute_scores(true_mesh_labels, predicted_labels, loss_confidences)


def visualizeErrorsMeshNetworkBLOCK(model, dataset, test_mask, block_size, device):
    model.eval()
    predicted_labels = np.empty(0)
    loss_confidences = np.empty((0, dataset.numClasses()))
    for i in range(0, len(dataset.graphs[test_mask]), block_size):
        block = [(dataset.graphs[test_mask][i + j], dataset.graphs[test_mask][i + j].patches) for j in range(block_size)]
        mesh_graphs = [tup[0] for tup in block]
        patches_list = [tup[1] for tup in block]
        pred = model(mesh_graphs, patches_list, device)
        pred = pred.cpu()
        predicted_labels = np.hstack((predicted_labels, pred.argmax(dim=1)))  # Take the highest value in the predicted classes vector
        loss_confidences = np.vstack((loss_confidences, pred.detach().numpy()))
        if predicted_labels[-1] != dataset.labels[test_mask][i]:
            print(f"Mesh misclassified with class {predicted_labels[-1]}, should be {int(dataset.labels[test_mask][i])}!")
            with open(f"../Datasets/Meshes/SHREC17/class_{int(dataset.labels[test_mask][i])}/id_{dataset.graphs[test_mask][i].sample_id}/resolution_{dataset.graphs[test_mask][i].resolution_level}/mesh{dataset.graphs[test_mask][i].mesh_id}.pkl", "rb") as file:
                mesh = pkl.load(file)
            mesh.drawWithMeshGraphs(mesh_graphs)
    model.train()
    true_mesh_labels = torch.tensor([dataset.labels[test_mask][i] for i in range(0, len(dataset.graphs[test_mask]), block_size)])
    return compute_scores(true_mesh_labels, predicted_labels, loss_confidences)


def compute_scores(true_labels, pred_labels, loss_predicted):
    # Computation of accuracy metrics
    dataset_labels_cpu = true_labels.cpu().numpy()
    accuracy = np.equal(pred_labels, dataset_labels_cpu).sum() / len(true_labels)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        confusion_matrix = metrics.confusion_matrix(true_labels.tolist(), pred_labels.tolist(), normalize="true")
    final_labels = torch.tensor(dataset_labels_cpu, dtype=torch.int64)
    loss = F.cross_entropy(torch.tensor(loss_predicted), final_labels)
    return accuracy, loss.item(), confusion_matrix


if __name__ == "__main__":
    main()
