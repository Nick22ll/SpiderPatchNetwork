import itertools
import os
import pickle as pkl
import random
import re
import sys
import warnings
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from scipy.special import softmax
from sklearn import metrics
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from Networks.CONVNetworks import MeshNetworkARAR, MeshNetworkPEARAR, MeshNetworkPEUR, MeshNetworkURUR, MeshNetworkSRAR, AverageMeshNetworkPEARAR, AverageMeshNetworkPEARUR
from PlotUtils import save_confusion_matrix, plot_training_statistics, plot_grad_flow, plot_voting_confidences
from SpiderDatasets.MeshGraphDataset import MeshGraphDataset


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###### TRAIN DI UN MODELLO SU 4 RISOLUZIONI DI UN DATASET  #########

    dataset_name = "SHREC17_R10_R8_P10_CSIRSv2Spiral_MGRAPHS50_SPIDER25_CONN2_RES3"
    # dataset_name = "SHREC17_R10_R4_P6_CSIRSv2Spiral_MGRAPHS5_SPIDER50_CONN5_RES3"

    features = np.array(["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"])
    features_to_keep = [3, 4]
    radius_to_keep = [0, 1, 2, 3, 4]

    dataset = MeshGraphDataset(dataset_name=dataset_name)
    dataset.load_from(f"../Datasets/MeshGraphs", dataset_name)

    # config = re.search('_(.*?)_', dataset_name).group(1)
    # rings = int(re.search('_R(\d)_P', dataset_name).group(1))
    # points = int(re.search('_P(\d)_C', dataset_name).group(1))

    print(dataset.graphs[0].patches[0].node_attr_schemes())
    # dataset.removeClasses([0, 1, 2, 4, 10, 11, 12])
    # dataset.removeSpiderPatchByNumNodes((rings * points) + 1)
    dataset.keepCurvaturesResolution(radius_to_keep)
    # dataset.aggregateSpiderPatchesNodeFeatures(["vertices"])
    dataset.aggregateSpiderPatchesNodeFeatures(features[features_to_keep])
    dataset.aggregateSpiderPatchEdgeFeatures()
    dataset.removeNonAggregatedFeatures()

    train_mask, test_mask = dataset.getTrainTestMask(10, percentage=False)
    class_num = len(np.unique(dataset.labels[train_mask]))
    dataset.normalize(train_mask)
    dataset.normalize_edge(train_mask)
    print(dataset.graphs[0].patches[0].node_attr_schemes())
    dataset.to(device)

    feat_dim = dataset.graphs[0].patches[0].ndata["aggregated_feats"].shape[1]
    readout_dim = feat_dim * dataset.graphs[0].patches[0].num_nodes()
    hidden_dim = 32

    epochs = 50

    # model = AverageMeshNetworkPEARUR(feat_dim, readout_dim=feat_dim, hidden_dim=hidden_dim, block_dim=5, out_feats=class_num, dropout=0, mesh_graph_edge_weights=True)
    # model.to(device)
    # trainMeshNetworkNEW(model, dataset, train_mask, test_mask, 5, device, epochs, f"{dataset_name}_{model.name}_K2LD_AVERAGEBLOCKMODEL")
    #

    model = AverageMeshNetworkPEARAR(feat_dim, readout_dim=feat_dim * 2, hidden_dim=hidden_dim, block_dim=5, out_feats=class_num, dropout=0, mesh_graph_edge_weights=True)
    model.to(device)
    trainMeshNetworkNEW(model, dataset, train_mask, test_mask, 5, device, epochs, f"{dataset_name}_{model.name}_K2LD_AVERAGEBLOCKMODEL")

    model = MeshNetworkPEARAR(feat_dim, readout_dim=feat_dim * 2, hidden_dim=hidden_dim, out_feats=class_num, dropout=0, mesh_graph_edge_weights=True)
    model.to(device)
    trainMeshNetwork(model, dataset, train_mask, test_mask, device, epochs, f"{dataset_name}_{model.name}_K2LD_SIMPLEMODEL")

    # model = MeshNetworkARAR(feat_dim, hidden_dim=hidden_dim, out_feats=class_num, dropout=0, mesh_graph_edge_weights=True)
    # model.to(device)
    # trainMeshNetwork(model, dataset, train_mask, test_mask, device, epochs, f"{dataset_name}_{model.name}_K2LD_SIMPLEMODEL")

    # model = MeshNetworkPEUR(feat_dim, readout_dim=feat_dim * 2, hidden_dim=feat_dim * 4, out_feats=class_num, dropout=0, mesh_graph_edge_weights=True)
    # model.to(device)
    # trainMeshNetwork(model, dataset, train_mask, test_mask, device, epochs, f"{dataset_name}_{model.name}_K2LD_SIMPLEMODEL")
    # #
    # model = MeshNetworkURUR(feat_dim, readout_dim=feat_dim * 2, hidden_dim=hidden_dim, out_feats=class_num, dropout=0, mesh_graph_edge_weights=True)
    # model.to(device)
    # trainMeshNetwork(model, dataset, train_mask, test_mask, device, epochs, f"{dataset_name}_{model.name}_K2LD_SIMPLEMODEL")

    # model = MeshNetworkPEUR(feat_dim, feat_dim * 2, hidden_dim=hidden_dim, out_feats=class_num, dropout=0, mesh_graph_edge_weights=True)
    # model.to(device)
    # trainMeshNetwork(model, dataset, train_mask, test_mask, device, epochs, f"{dataset_name}_{model.name}_prova")

    # model = MeshNetworkSRUR(readout_dim, hidden_dim=hidden_dim, out_feats=class_num, dropout=0.25, mesh_graph_edge_weights=False)
    # model.to(device)
    # trainMeshNetwork(model, dataset, train_mask, test_mask, device, epochs, f"{dataset_name}_{model.name}_prova")

    # model = MeshNetworkSRPR(readout_dim, hidden_dim=hidden_dim, out_feats=class_num, dropout=0.25, mesh_graph_edge_weights=False)
    # model.to(device)
    # trainMeshNetwork(model, dataset, train_mask, test_mask, device, epochs, f"{dataset_name}_{model.name}")
    #
    # model = MeshNetworkSRAR(readout_dim, hidden_dim=hidden_dim, out_feats=class_num, dropout=0.25, mesh_graph_edge_weights=False)
    # model.to(device)
    # trainMeshNetwork(model, dataset, train_mask, test_mask, device, epochs, f"{dataset_name}_{model.name}")
    #
    # model = GATMeshNetworkSRUR(readout_dim, hidden_dim=hidden_dim, out_feats=class_num, dropout=0.25, mesh_graph_edge_weights=False)
    # model.to(device)
    # trainMeshNetwork(model, dataset, train_mask, test_mask, device, epochs, f"{dataset_name}_{model.name}")
    #
    # hidden_dim = 950
    #
    # model = GATMeshNetworkSRPR(readout_dim, hidden_dim=hidden_dim, out_feats=class_num, dropout=0.25, mesh_graph_edge_weights=False)
    # model.to(device)
    # trainMeshNetwork(model, dataset, train_mask, test_mask, device, epochs, f"{dataset_name}_{model.name}")

    # summary(model)  # , input_data=[(dataset.graphs[0], dataset.graphs[0].patches, device)], col_names=["input_size","output_size","num_params"], depth=4


def add_edge_weight(mesh_graph_dataset):
    for mesh_graph in mesh_graph_dataset.graphs:
        weights = []
        for edge_id in range(int(len(mesh_graph.edges()[0]) / 2)):
            start = mesh_graph.patches[mesh_graph.edges()[0][edge_id]].seed_point
            end = mesh_graph.patches[mesh_graph.edges()[1][edge_id]].seed_point
            distance = np.linalg.norm(end - start)
            weights.append(1 - (1 / distance))
        weights = np.concatenate((weights, weights))
        mesh_graph.edata["weights"] = torch.tensor(weights, dtype=torch.float32)


def trainMeshNetwork(model, dataset, train_mask, test_mask, device, epochs=1000, train_name=""):
    model.train()
    best_acc = 0
    best_mesh_acc = 0
    best_loss = 1000

    # dataloader = GraphDataLoader(dataset[train_mask], batch_size=1, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=10e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.40), int(epochs * 0.85), int(epochs * 0.95)], gamma=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, threshold=0.01)
    start = time()
    train_losses = []
    val_accuracies = []
    mesh_val_accuracies = []
    val_losses = []
    for epoch in range(epochs):
        train_losses.append(0)
        np.random.shuffle(train_mask)
        dataloader = GraphDataLoader(dataset[train_mask], batch_size=1, drop_last=False)
        sampler = 0
        for graph, label in tqdm(dataloader, position=0, leave=False, desc=f"Epoch {epoch + 1}: ", colour="white", ncols=80):
            optimizer.zero_grad()
            pred = model(graph, dataset.graphs[train_mask][sampler].patches, device)
            loss_running = F.cross_entropy(pred, label)
            train_losses[-1] += loss_running.item()
            loss_running.backward()
            optimizer.step()
            sampler += 1
        plot_grad_flow(model.named_parameters())

        train_losses[-1] /= len(dataset.graphs[train_mask])
        print(f"Epoch {epoch + 1}/{epochs} ({int(time() - start)}s):"
              f" Epoch Loss={train_losses[-1]:.3f}")

        mesh_graph_classification_statistics, mesh_classification_statistics = testMeshNetwork(model=model, dataset=dataset, test_mask=test_mask, device=device)
        mesh_graph_classification_acc, mesh_graph_classification_loss, mesh_graph_classification_cm = mesh_graph_classification_statistics
        mesh_class_acc, mesh_class_cm, mesh_class_data = mesh_classification_statistics

        val_accuracies.append(mesh_graph_classification_acc)
        mesh_val_accuracies.append(mesh_class_acc)
        val_losses.append(mesh_graph_classification_loss)
        print(f"Validation Test\n"
              f"Mesh Graph Acc. : {mesh_graph_classification_acc}\n"
              f"Mesh GraphLoss.: {mesh_graph_classification_loss}\n"
              f"Mesh Acc.: {mesh_class_acc}")

        if mesh_graph_classification_acc > best_acc:
            best_acc = mesh_graph_classification_acc
            # model.save(f"../TrainedModels/Train_{train_name}/MeshNetworkBestAcc")
            save_confusion_matrix(mesh_graph_classification_cm, f"../TrainedModels/Train_{train_name}/MeshNetworkBestAcc", "MeshGraphConfusionMatrix.png")
            save_confusion_matrix(mesh_class_cm, f"../TrainedModels/Train_{train_name}/MeshNetworkBestAcc", "MeshConfusionMatrix.png")
            plot_voting_confidences(mesh_class_data, f"../TrainedModels/Train_{train_name}/MeshNetworkBestAcc", f"Confidence_statistics.png")

        if mesh_graph_classification_loss < best_loss:
            best_loss = mesh_graph_classification_loss
            # model.save(f"../TrainedModels/Train_{train_name}/MeshNetworkBestLoss")
            save_confusion_matrix(mesh_graph_classification_cm, f"../TrainedModels/Train_{train_name}/MeshNetworkBestLoss", "MeshGraphConfusionMatrix.png")
            save_confusion_matrix(mesh_class_cm, f"../TrainedModels/Train_{train_name}/MeshNetworkBestLoss", "MeshConfusionMatrix.png")
            plot_voting_confidences(mesh_class_data, f"../TrainedModels/Train_{train_name}/MeshNetworkBestLoss", f"Confidence_statistics.png")

        if mesh_class_acc > best_mesh_acc:
            best_mesh_acc = mesh_class_acc

        # scheduler.step(loss)
        scheduler.step()
        plot_training_statistics(f"../TrainedModels/Train_{train_name}", f"Train_MeshGraph_statistics.png", title=f"Best Acc: {np.trunc(best_acc * 10000) / 100}, Loss: {best_loss}", epochs=range(epoch + 1), losses=train_losses, val_epochs=range(epoch + 1), val_accuracies=val_accuracies,
                                 val_losses=val_losses)
        plot_training_statistics(f"../TrainedModels/Train_{train_name}", f"Train_Mesh_statistics.png", title=f"Best Acc: {np.trunc(best_mesh_acc * 10000) / 100}", epochs=range(epoch + 1), losses=train_losses, val_accuracies=mesh_val_accuracies, val_epochs=range(epoch + 1))


def trainMeshNetworkNEW(model, dataset, train_mask, test_mask, block_size, device, epochs=1000, train_name=""):
    model.train()
    best_acc = 0
    best_loss = 1000

    # dataloader = GraphDataLoader(dataset[train_mask], batch_size=1, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=10e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.40), int(epochs * 0.85), int(epochs * 0.95)], gamma=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, threshold=0.01)
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

        mesh_class_acc, mesh_class_loss, mesh_class_cm = testMeshNetworkNEW(model=model, dataset=dataset, test_mask=test_mask, block_size=block_size, device=device)
        mesh_val_accuracies.append(mesh_class_acc)
        mesh_val_losses.append(mesh_class_loss)

        print(f"Validation Test\n"
              f"Mesh Graph Acc. : {mesh_class_acc}\n"
              f"Mesh GraphLoss.: {mesh_class_loss}")

        if mesh_class_acc > best_acc:
            best_acc = mesh_class_acc
            # model.save(f"../TrainedModels/Train_{train_name}/MeshNetworkBestAcc")
            save_confusion_matrix(mesh_class_cm, f"../TrainedModels/Train_{train_name}/MeshNetworkBestAcc", "MeshGraphConfusionMatrix.png")
        if mesh_class_loss < best_loss:
            best_loss = mesh_class_loss
            # model.save(f"../TrainedModels/Train_{train_name}/MeshNetworkBestLoss")
            save_confusion_matrix(mesh_class_cm, f"../TrainedModels/Train_{train_name}/MeshNetworkBestLoss", "MeshGraphConfusionMatrix.png")

        # scheduler.step(loss)
        scheduler.step()
        plot_training_statistics(f"../TrainedModels/Train_{train_name}", f"Train_MeshGraph_statistics.png", title=f"Best Acc: {np.trunc(best_acc * 10000) / 100}, Loss: {best_loss}", epochs=range(epoch + 1), losses=train_losses, val_epochs=range(epoch + 1), val_accuracies=mesh_val_accuracies,
                                 val_losses=mesh_val_losses)


def testMeshNetwork(model, dataset, test_mask, device):
    model.eval()
    predicted_labels = np.empty(0)
    loss_confidences = np.empty((0, dataset.numClasses()))
    dataloader = GraphDataLoader(dataset[test_mask], batch_size=1, drop_last=False)
    sampler = 0
    for graph, label in dataloader:
        pred = model(graph, dataset.graphs[test_mask][sampler].patches, device)
        pred = pred.cpu()
        predicted_labels = np.hstack((predicted_labels, pred.argmax(dim=1)))  # Take the highest value in the predicted classes vector
        loss_confidences = np.vstack((loss_confidences, pred.detach().numpy()))
        sampler += 1
    model.train()
    mesh_ids = dataset.mesh_id[test_mask]
    true_mesh_labels = dataset.labels[test_mask]
    mesh_predicted_labels = {}
    for id in np.unique(mesh_ids):
        mesh_predicted_labels[int(id)] = {}
        mesh_predicted_labels[int(id)]["pred"] = []
        mesh_predicted_labels[int(id)]["true"] = int(true_mesh_labels[mesh_ids == id][0].cpu())
        mesh_confidences = loss_confidences[mesh_ids == id]
        mesh_confidences = softmax(mesh_confidences, axis=1)
        mesh_confidences = np.mean(mesh_confidences, axis=0)

        top3_labels = np.argpartition(mesh_confidences, -3)[-3:]  # Indices not sorted
        top3_labels = top3_labels[np.argsort(mesh_confidences[top3_labels])][::-1]  # Indices sorted by value from largest to smallest
        for idx, pred_label in enumerate(top3_labels):
            mesh_predicted_labels[int(id)]["pred"].append((pred_label, mesh_confidences[top3_labels[idx]]))

    return compute_scores(dataset.labels[test_mask], predicted_labels, loss_confidences), compute_voting_scores(mesh_predicted_labels)


def testMeshNetworkNEW(model, dataset, test_mask, block_size, device):
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


def compute_voting_scores(mesh_results):
    dict_to_plot = {"data": {}, "colors": []}
    pred_labels = []
    true_labels = []
    color_sample = {0: "red", 1: "green", 2: "blue"}
    for mesh_id, internal_dict in mesh_results.items():
        true_labels.append(internal_dict["true"])
        pred_labels.append(internal_dict["pred"][0][0])
        for confidence_level, confidence in enumerate(internal_dict["pred"]):
            dict_to_plot["data"][f"{true_labels[-1]}_top{confidence_level}"] = confidence
            dict_to_plot["colors"].append(color_sample[confidence_level])

    # Computation of accuracy metrics
    accuracy = np.equal(pred_labels, true_labels).sum() / len(true_labels)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        confusion_matrix = metrics.confusion_matrix(true_labels, pred_labels, normalize="true")
    return accuracy, confusion_matrix, dict_to_plot


def gridSearch(model, train_dataset, test_dataset, device):
    params = {
        'internal_hidden_dim': [64, 128, 256],
        'readout_dim': [16, 32, 64],
        'hidden_dim': [32, 64, 128],
        'dropout': [0],  # , 0.25, 0.5],
        'patch_batch': [25, 10],
    }

    dataloader = GraphDataLoader(train_dataset, {"batch_size": 1, "drop_last": False})
    grid_results = []
    for batch_size in params["patch_batch"]:
        for internal_hidden_dim in params["internal_hidden_dim"]:
            for readout_dim in params["readout_dim"]:
                for hidden_dim in params["hidden_dim"]:
                    for dropout in params["dropout"]:
                        model = model(pr_conv_hd=internal_hidden_dim, pr_readout_dim=readout_dim, mr_hidden_dim=hidden_dim, dropout=dropout, patch_batch=batch_size, patch_feat_dim=train_dataset.numClasses()).to(device)
                        model.train()

                        optimizer = torch.optim.Adam(model.parameters(), lr=10e-5, weight_decay=10e-4)

                        best_acc = 0
                        best_loss = 1000
                        start = time()
                        train_losses = []
                        for _ in tqdm(range(50), position=0, leave=False, desc=f"Run internal_hidden_dim: {internal_hidden_dim}, readout_dim: {readout_dim}, hidden_dim: {hidden_dim}, dropout: {dropout}, patch_batch: {batch_size}", colour="white"):
                            sampler = 0
                            train_losses.append(0)
                            for graph, label in dataloader:
                                optimizer.zero_grad()
                                pred, embedding = model(graph, train_dataset.graphs[sampler].patches, device)
                                loss_running = F.cross_entropy(pred, label)
                                train_losses[-1] += loss_running.item()
                                loss_running.backward()
                                optimizer.step()
                                sampler += 1
                            train_losses[-1] /= len(train_dataset.graphs)
                            acc, loss, _ = testMeshNetwork(model=model, dataset=test_dataset, device=device)

                            if acc > best_acc:
                                best_acc = acc

                            if loss < best_loss:
                                best_loss = loss

                        run_results = {
                            'internal_hidden_dim': internal_hidden_dim,
                            'readout_dim': readout_dim,
                            'hidden_dim': hidden_dim,
                            'dropout': dropout,
                            'patch_batch': batch_size,
                            "accuracy": best_acc,
                            "loss": best_loss,
                            "train_time": f"{time() - start}s"
                        }
                        grid_results.append(run_results)
                        print(run_results)
                        with open("grid_search_results.pkl", "wb") as grid_file:
                            pkl.dump(grid_results, grid_file, protocol=-1)


if __name__ == "__main__":
    main()
