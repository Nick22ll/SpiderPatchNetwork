import warnings
from time import time
import pickle as pkl
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from PlotUtils import plot_confusion_matrix, save_confusion_matrix, plot_training_statistics, plot_embeddings, plot_model_parameters_comparison, plot_grad_flow, print_weights, print_weights_difference
from sklearn import metrics
import torch.nn.functional as F
from SpiderDatasets.MeshGraphForTrainingDataset import MeshGraphDatasetForNNTraining
from SpiderPatch.CONVNetworks import *
from tqdm import tqdm, trange
from dgl.dataloading import GraphDataLoader


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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###### TRAIN DI UN MODELLO SU 4 RISOLUZIONI DI UN DATASET  #########
    dataset = MeshGraphDatasetForNNTraining()
    # dataset.load(f"Datasets/MeshGraphsForTraining/SHREC17_R10_RI6_P6_PATCH20_SAMPLE50_3CONN_ONLY2CURV/SHREC17_R10_RI6_P6_PATCH20_SAMPLE50_3CONN_ONLY2CURV_all_Normalized", f"SHREC17_R10_RI6_P6_PATCH20_SAMPLE50_3CONN_ONLY2CURV_all")
    # dataset.load(f"Datasets/MeshGraphsForTraining/SHREC17_R10_RI6_P6_PATCH50_SAMPLE30_CONN5/SHREC17_R10_RI6_P6_PATCH50_SAMPLE30_CONN5_level_0_Normalized", f"SHREC17_R10_RI6_P6_PATCH50_SAMPLE30_CONN5_level_0")
    # dataset.load(f"Datasets/MeshGraphsForTraining/SHREC17_R10_RI6_P6_PATCH50_SAMPLE30_CONN5_ONLY2CURV/SHREC17_R10_RI6_P6_PATCH50_SAMPLE30_CONN5_ONLY2CURV_all_Normalized", f"SHREC17_R10_RI6_P6_PATCH50_SAMPLE30_CONN5_ONLY2CURV_all")
    # dataset.load(f"Datasets/MeshGraphsForTraining/SHREC17_R10_RI6_P6_PATCH50_SAMPLE30_CONN10_ONLYCURV/SHREC17_R10_RI6_P6_PATCH50_SAMPLE30_CONN10_ONLYCURV_all_Normalized", f"SHREC17_R10_RI6_P6_PATCH50_SAMPLE30_CONN10_ONLYCURV_all")
    # dataset.load(f"Datasets/MeshGraphsForTraining/SHREC17_R10_RI6_P6_PATCH50_SAMPLE30_CONN10_ONLYCURV/SHREC17_R10_RI6_P6_PATCH50_SAMPLE30_CONN10_ONLYCURV_level_1_Normalized", f"SHREC17_R10_RI6_P6_PATCH50_SAMPLE30_CONN10_ONLYCURV_level_1")
    # dataset.load(f"Datasets/MeshGraphsForTraining/SHREC17_R10_RI3_P4_PATCH30_SAMPLE100_FC_ONLYCURV/SHREC17_R10_RI3_P4_PATCH30_SAMPLE100_FC_ONLYCURV_level_1_Normalized", f"SHREC17_R10_RI3_P4_PATCH30_SAMPLE100_FC_ONLYCURV_level_1")
    # dataset.load(f"Datasets/MeshGraphsForTraining/SHREC17_R10_RI3_P4_PATCH50_SAMPLE30_CONN20_ONLYCURV/SHREC17_R10_RI3_P4_PATCH50_SAMPLE30_CONN20_ONLYCURV_all_Normalized", f"SHREC17_R10_RI3_P4_PATCH50_SAMPLE30_CONN20_ONLYCURV_all")
    # dataset.load(f"Datasets/MeshGraphsForTraining/SHREC17_R10_RI6_P6_PATCH50_SAMPLE30_CONN10/SHREC17_R10_RI6_P6_PATCH50_SAMPLE30_CONN10_level_0_Normalized", f"SHREC17_R10_RI6_P6_PATCH50_SAMPLE30_CONN10_level_0")
    # dataset.load(f"Datasets/MeshGraphsForTraining/SHREC17_R10_RI6_P6_PATCH25_SAMPLE20_CONN3_ONLY2CURV/SHREC17_R10_RI6_P6_PATCH25_SAMPLE20_CONN3_ONLY2CURV_all_Normalized", f"SHREC17_R10_RI6_P6_PATCH25_SAMPLE20_CONN3_ONLY2CURV_all")
    # dataset.load(f"Datasets/MeshGraphsForTraining/
    # s/MeshGraphsForTraining/SHREC17_R10_RI6_P6_PATCH20_SAMPLE10_FC/SHREC17_R10_RI6_P6_PATCH20_SAMPLE10_FC_{level}_Normalized", f"SHREC17_R10_RI6_P6_PATCH20_SAMPLE10_FC_{level}")
    # dataset.load(f"Datasets/MeshGraphsForTraining/SHREC17_R10_RI3_P4_PATCH30_SAMPLE20_FC_LRF/SHREC17_R10_RI3_P4_PATCH30_SAMPLE20_FC_LRF_all_Normalized", f"SHREC17_R10_RI3_P4_PATCH30_SAMPLE20_FC_LRF_all")
    dataset.load(f"Datasets/MeshGraphsForTraining/SHREC17_R10_RI6_P6_SUPERPATCH10_SAMPLE20/SHREC17_R10_RI6_P6_SUPERPATCH10_SAMPLE20_FC_Normalized", f"SHREC17_R10_RI6_P6_SUPERPATCH10_SAMPLE20_FC")
    add_edge_weight(dataset.train_dataset)
    add_edge_weight(dataset.validation_dataset)
    add_edge_weight(dataset.test_dataset)
    dataset.normalize_edge()
    dataset.normalize_test_dataset_edge()
    dataset.normalize_validation_dataset_edge()
    print(dataset.train_dataset.graphs[0].patches[0].node_attr_schemes())
    dataset.aggregateNodeFeatures(["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"])

    dataset.aggregateEdgeFeatures()
    dataset.removeNonAggregatedFeatures()
    dataset.train_dataset.graphs[0].draw()
    print(dataset.train_dataset.graphs[0].patches[0].node_attr_schemes())
    indices = []
    if 1 == 0:
        for i, graph in enumerate(dataset.validation_dataset.graphs):
            if graph.resolution_level != "level_0":
                indices.append(i)
        indices = sorted(indices, reverse=True)
        for idx in indices:
            if idx < len(dataset.validation_dataset.graphs):
                dataset.validation_dataset.graphs.pop(idx)
                dataset.validation_dataset.labels.pop(idx)
                dataset.validation_dataset.sample_id.pop(idx)
                dataset.validation_dataset.mesh_id.pop(idx)

    dataset.to(device)

    model = MeshNetwork(patch_feat_dim=dataset.train_dataset.graphs[0].patches[0].ndata["aggregated_feats"].shape[1], internal_hidden_dim=1024, readout_dim=64, hidden_dim=2048, out_feats=15, dropout=0.5, patch_batch=10, patch_edge_weights=True, mesh_graph_edge_weights=True)
    model.to(device)

    trainMeshNetwork(model, dataset, device, 150, f"SHREC17_R10_RI6_P6_SUPERPATCH10_SAMPLE20_FC_ONLYCURV_BIASWEIGHT_NORMEDGEWEIGHTS_X4", dataset.validation_dataset)

    # acc, loss, cm = testMeshNetwork(model=model, dataset=test_dataset, device=device)
    # print(f"Validation Test\n"
    #       f"Acc. : {acc}\n"
    #       f"Loss.: {loss}")

    # gridSearch(train_dataset=dataset.train_dataset, test_dataset=test_dataset, device=device)


def trainMeshNetwork(model, dataset, device, epochs=1000, train_name="", test_dataset=None):
    model.train()
    best_acc = 0
    best_loss = 1000

    dataloader = GraphDataLoader(dataset.train_dataset, batch_size=1, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=10e-4)
    # optimizer = torch.optim.Adam([{"params": model.patch_reader.parameters(), "lr": 10e-4}, {"params": model.mesh_reader.parameters(), "lr": 10e-5}], weight_decay=10e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.65), int(epochs * 0.90)], gamma=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, threshold=0.01)
    start = time()
    train_losses = []
    val_accuracies = []
    val_losses = []
    for epoch in range(epochs):
        sampler = 0
        train_losses.append(0)
        for id, group in enumerate(optimizer.state_dict()['param_groups']):
            print(f"Learning rate of group {id}: {group['lr']}")
        for graph, label in tqdm(dataloader, position=0, leave=False, desc=f"Epoch {epoch + 1}: ", colour="white", ncols=80):
            optimizer.zero_grad()
            pred, embedding = model(graph, dataset.train_dataset.graphs[sampler].patches, device)
            loss_running = F.cross_entropy(pred, label)
            train_losses[-1] += loss_running.item()
            loss_running.backward()
            optimizer.step()
            sampler += 1
            # if sampler % 1000 == 0:
            # plot_grad_flow(model.named_parameters())
        # if epoch % 20 == 0:
        #     plot_embeddings(model, dataset.train_dataset, device)
        train_losses[-1] /= len(dataset.train_dataset.graphs)
        print(f"Epoch {epoch + 1}/{epochs} ({int(time() - start)}s):"
              f" Epoch Loss={train_losses[-1]:.3f}")

        if test_dataset is None:
            acc, loss, cm = testMeshNetwork(model=model, dataset=dataset.validation_dataset, device=device)
        else:
            acc, loss, cm = testMeshNetwork(model=model, dataset=test_dataset, device=device)
        val_accuracies.append(acc)
        val_losses.append(loss)
        print(f"Validation Test\n"
              f"Acc. : {acc}\n"
              f"Loss.: {loss}")

        if acc > best_acc:
            best_acc = acc
            model.save(f"TrainedModels/Train_{train_name}/MeshNetworkBestAcc")
            save_confusion_matrix(cm, f"TrainedModels/Train_{train_name}/MeshNetworkBestAcc/ConfusionMatrixEpoch{epoch + 1}.png")

        if loss < best_loss:
            best_loss = loss
            model.save(f"TrainedModels/Train_{train_name}/MeshNetworkBestLoss")
            save_confusion_matrix(cm, f"TrainedModels/Train_{train_name}/MeshNetworkBestLoss/ConfusionMatrixEpoch{epoch + 1}.png")

        # scheduler.step(loss)
        scheduler.step()
        plot_training_statistics(f"TrainedModels/Train_{train_name}/Train_{train_name}_statistics.png", title=f"Best Acc: {np.trunc(best_acc * 10000) / 100}, Loss: {best_loss}", epochs=range(epoch + 1), losses=train_losses, val_accuracies=val_accuracies,
                                 val_losses=val_losses)


def testMeshNetwork(model, dataset, device):
    model.eval()
    correct_prediction_number = np.empty(0)
    loss_predicted = np.empty((0, dataset.numClasses()))
    dataloader = GraphDataLoader(dataset, batch_size=1, drop_last=False)
    sampler = 0
    for graph, label in dataloader:
        pred = model(graph, dataset.graphs[sampler].patches, device)[0]
        pred = pred.cpu()
        sampler += 1
        correct_prediction_number = np.hstack((correct_prediction_number, pred.argmax(dim=1)))  # Take the highest value in the predicted classes vector
        loss_predicted = np.vstack((loss_predicted, pred.detach().numpy()))
    model.train()
    return compute_scores(dataset, correct_prediction_number, loss_predicted)


def compute_scores(dataset, pred_labels, loss_predicted):
    # Computation of accuracy metrics
    dataset_labels_cpu = dataset.labels.cpu().numpy()
    accuracy = np.equal(pred_labels, dataset_labels_cpu).sum() / len(dataset.graphs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        confusion_matrix = metrics.confusion_matrix(dataset.labels.tolist(), pred_labels.tolist(), normalize="true")
    final_labels = torch.tensor(dataset_labels_cpu, dtype=torch.int64)
    loss = F.cross_entropy(torch.tensor(loss_predicted), final_labels)
    return accuracy, loss.item(), confusion_matrix


def gridSearch(train_dataset, test_dataset, device):
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
                        model = MeshNetwork(internal_hidden_dim=internal_hidden_dim, readout_dim=readout_dim, hidden_dim=hidden_dim, dropout=dropout, patch_batch=batch_size, patch_feat_dim=train_dataset.numClasses()).to(device)
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
