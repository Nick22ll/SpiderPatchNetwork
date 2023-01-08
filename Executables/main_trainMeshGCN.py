import os
import pickle as pkl
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
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from Networks.GATNetworks import SimplestNetwork, SimplestURNetwork, GATJumpARNetwork, GATCriterion, GATJumpARNetworkBatch
from PlotUtils import save_confusion_matrix, plot_training_statistics, plot_grad_flow, plot_voting_confidences
from SpiderDatasets.MeshGraphDataset import MeshGraphDataset
from Utils import AverageMeter, ProgressMeter, accuracy


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_name = "SHREC17_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS5_SPIDER20_CONN5_RES3"

    features = np.array(["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"])
    features_to_keep = [0, 1, 2, 3, 4]
    radius_to_keep = [0, 1, 2, 3, 4]

    dataset = MeshGraphDataset(dataset_name=dataset_name)
    dataset.load_from(f"../Datasets/MeshGraphs", dataset_name)

    config = re.search('_(.*?)_', dataset_name).group(1)
    rings = int(re.search('_R(\d)_P', dataset_name).group(1))
    points = int(re.search('_P(\d)_C', dataset_name).group(1))

    print(dataset.graphs[0].patches[0].node_attr_schemes())
    # dataset.removeClasses([5, 10, 11, 12, 13])
    # dataset.removeSpiderPatchByNumNodes((rings * points) + 1)
    dataset.keepCurvaturesResolution(radius_to_keep)
    # dataset.aggregateSpiderPatchesNodeFeatures(["vertices"])
    dataset.aggregateSpiderPatchesNodeFeatures(features[features_to_keep], "aggregated_feats")
    dataset.aggregateSpiderPatchesNodeFeatures(["weights", "rings", "points"], "aggregated_weights")
    dataset.aggregateSpiderPatchEdgeFeatures()
    dataset.removeNonAggregatedFeatures()

    train_mask, test_mask = dataset.getTrainTestMask(10, percentage=False)
    class_num = len(np.unique(dataset.labels[train_mask]))

    mode = "normalization"
    elim_mode = None

    dataset.normalizeV2(train_mask, mode, elim_mode)
    # dataset.normalize_edge(train_mask)
    # plot_data_distributions(dataset, train_mask)

    print(dataset.graphs[0].patches[0].node_attr_schemes())
    dataset.to(device)

    feat_dim = dataset.graphs[0].patches[0].ndata["aggregated_feats"].shape[1]
    weights_dim = dataset.graphs[0].patches[0].ndata["aggregated_weights"].shape[1]
    epochs = 100

    model = GATJumpARNetworkBatch(feat_dim, out_feats=class_num, use_node_weights=True, weights_in_channels=weights_dim, dropout=0, use_SP_triplet=False)
    model.to(device)
    trainMeshNetworkWithBatch(model, dataset, train_mask, test_mask, device, epochs, train_name=f"{dataset_name}_{model.name}_{mode}_TRIPLET")

    # model = BlockMeshNetwork(feat_dim, hidden_dim=hidden_dim, block_dim=5, out_feats=class_num, dropout=0, block_reduction="average", mesh_graph_edge_weights=True)
    # model.to(device)
    # trainBLOCKMeshNetwork(model, dataset, train_mask, test_mask, 5, device, epochs=epochs, train_name=f"{dataset_name}_{model.name}_ALLFEATS_{mode}", optimizer_option="all", debug=False)


def trainMeshNetwork(model, dataset, train_mask, test_mask, device, epochs=1000, train_name="", optimizer_option="all"):
    PATH = f"../TrainedModels/Train_{train_name}"

    torch.autograd.set_detect_anomaly(True)
    rng = np.random.default_rng(22)
    model.train()
    best_acc = 0
    best_mesh_acc = 0
    best_loss = 1000

    BATCH_SIZE = 1

    # dataloader = GraphDataLoader(dataset[train_mask], batch_size=1, drop_last=False)
    if optimizer_option == "all":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.025, weight_decay=10e-4)

    elif optimizer_option == "patch":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.patch_embedder.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.patch_embedder.parameters(), lr=1e-4, weight_decay=10e-4)

    elif optimizer_option == "mesh":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.mesh_embedder.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.mesh_embedder.parameters(), lr=1e-4, weight_decay=10e-4)
    else:
        raise ()

    criterion = GATCriterion()
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.40), int(epochs * 0.85), int(epochs * 0.95)], gamma=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, threshold=0.01)

    start = time()
    train_losses = []
    val_accuracies = []
    mesh_val_accuracies = []
    val_losses = []

    for epoch in range(epochs):
        rng.shuffle(train_mask)
        dataloader = GraphDataLoader(dataset[train_mask], batch_size=BATCH_SIZE, drop_last=False)

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.6f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        progress = ProgressMeter(
            len(dataloader),
            [batch_time, losses, top1, top3],
            prefix='Training: ')

        train_losses.append(0)
        end = time()

        for sampler, (graph, label) in enumerate(tqdm(dataloader, position=0, leave=False, desc=f"Epoch {epoch + 1}: ", colour="white", ncols=80)):
            with torch.no_grad():
                positive_mask = (dataset.labels[train_mask] == label).cpu()
                positive_mask[sampler] = False
                positive_pool = dataset.graphs[train_mask][positive_mask]
                positive_sample = rng.choice(positive_pool)
                _, MG_positive_embedding, _, _, _ = model(positive_sample, positive_sample.patches, device)

                negative_mask = torch.logical_not(positive_mask)
                negative_mask[sampler] = False
                negative_pool = dataset.graphs[train_mask][negative_mask]
                negative_sample = rng.choice(negative_pool)
                _, MG_negative_embedding, _, _, _ = model(negative_sample, negative_sample.patches, device)

            optimizer.zero_grad()
            pred_class, MG_embedding, SP_embeddings, min_indices, max_indices = model(graph, dataset.graphs[train_mask][sampler].patches, device)

            batch_loss = criterion(pred_class, label, MG_embedding, MG_positive_embedding, MG_negative_embedding, SP_embeddings, min_indices, max_indices)
            train_losses[-1] += batch_loss.item()

            acc1, acc3 = accuracy(pred_class, label, topk=(1, 3))
            losses.update(batch_loss.item(), BATCH_SIZE)
            top1.update(acc1.item(), BATCH_SIZE)
            top3.update(acc3.item(), BATCH_SIZE)

            batch_loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

            if sampler % 100 == 0:
                tqdm.write(progress.get_string(sampler))

        plot_grad_flow(model.named_parameters())

        train_losses[-1] /= len(dataset.graphs[train_mask])
        print(f"Epoch {epoch + 1}/{epochs} ({int(time() - start)}s):"
              f" Epoch Loss={train_losses[-1]:.3f}")

        mesh_graph_classification_statistics, mesh_classification_statistics = testMeshNetwork(model=model, dataset=dataset, test_mask=test_mask, criterion=nn.CrossEntropyLoss(), batch_size=BATCH_SIZE, device=device)
        mgraph_acc1, mgraph_acc3, mgraph_loss, mgraph_cm = mesh_graph_classification_statistics
        mesh_acc, mesh_cm, mesh_data = mesh_classification_statistics

        val_accuracies.append(mgraph_acc1)
        mesh_val_accuracies.append(mesh_acc)
        val_losses.append(mgraph_loss)
        print(f"Validation Test\n"
              f"Mesh Graph Acc. : {mgraph_acc1}\n"
              f"Mesh Graph Loss.: {mgraph_loss}\n"
              f"Mesh Acc.: {mesh_acc}")

        if mgraph_acc1 > best_acc:
            best_acc = mgraph_acc1
            model.save(f"{PATH}/MeshNetworkBestAcc")
            save_confusion_matrix(mgraph_cm, f"{PATH}/MeshNetworkBestAcc", "MeshGraphConfusionMatrix.png")
            save_confusion_matrix(mesh_cm, f"{PATH}/MeshNetworkBestAcc", "MeshConfusionMatrix.png")
            plot_voting_confidences(mesh_data, f"{PATH}/MeshNetworkBestAcc", f"Confidence_statistics.png")
            with open(f'{PATH}/MeshNetworkBestAcc/bestAcc.txt', 'w') as f:
                f.write(f'Acc: {np.trunc(best_acc * 10000) / 100}\n'
                        f'Epoch:{epoch}\n'
                        f'Loss:{mgraph_loss}')

        if mgraph_loss < best_loss:
            best_loss = mgraph_loss
            # model.save(f"{PATH/MeshNetworkBestLoss")
            save_confusion_matrix(mgraph_cm, f"{PATH}/MeshNetworkBestLoss", "MeshGraphConfusionMatrix.png")
            save_confusion_matrix(mesh_cm, f"{PATH}/MeshNetworkBestLoss", "MeshConfusionMatrix.png")
            plot_voting_confidences(mesh_data, f"{PATH}/MeshNetworkBestLoss", f"Confidence_statistics.png")

        if mesh_acc > best_mesh_acc:
            best_mesh_acc = mesh_acc

        # scheduler.step(loss)
        scheduler.step()
        plot_training_statistics(f"{PATH}", f"Train_MeshGraph_statistics.png", title=f"Best Acc: {np.trunc(best_acc) / 100}, Loss: {best_loss}", epochs=range(epoch + 1), losses=train_losses, val_epochs=range(epoch + 1), val_accuracies=np.array(val_accuracies) / 100,
                                 val_losses=val_losses)
        plot_training_statistics(f"{PATH}", f"Train_Mesh_statistics.png", title=f"Best Acc: {best_mesh_acc}", epochs=range(epoch + 1), losses=train_losses, val_accuracies=np.array(mesh_val_accuracies) / 100, val_epochs=range(epoch + 1))


def trainMeshNetworkWithBatch(model, dataset, train_mask, test_mask, device, epochs=1000, train_name="", optimizer_option="all"):
    PATH = f"../TrainedModels/Train_{train_name}"

    torch.autograd.set_detect_anomaly(True)
    rng = np.random.default_rng(22)
    model.train()
    best_acc = 0
    best_mesh_acc = 0
    best_loss = 1000

    BATCH_SIZE = 64

    if optimizer_option == "all":
        optimizer = torch.optim.AdamW(model.parameters())

    elif optimizer_option == "patch":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.patch_embedder.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(model.patch_embedder.parameters())

    elif optimizer_option == "mesh":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.mesh_embedder.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(model.mesh_embedder.parameters())
    else:
        raise ()

    # criterion = GATCriterion()
    criterion = nn.CrossEntropyLoss()
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.40), int(epochs * 0.85), int(epochs * 0.95)], gamma=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, threshold=0.01)

    start = time()
    train_losses = []
    val_accuracies = []
    mesh_val_accuracies = []
    val_losses = []

    for epoch in range(epochs):
        rng.shuffle(train_mask)
        dataloader = GraphDataLoader(dataset[train_mask], batch_size=BATCH_SIZE, drop_last=False)

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.6f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        progress = ProgressMeter(
            len(dataloader),
            [batch_time, losses, top1, top3],
            prefix='Training: ')

        train_losses.append(0)
        end = time()

        for sampler, (graph, label) in enumerate(tqdm(dataloader, position=0, leave=False, desc=f"Epoch {epoch + 1}: ", colour="white", ncols=80)):
            patch_sampler = sampler + np.arange(0, len(label))
            patches = np.empty(0, dtype=object)
            for idx in patch_sampler:
                patches = np.hstack((patches, dataset.graphs[train_mask[idx]].patches))

            optimizer.zero_grad()
            pred_class, MG_embedding, SP_embeddings, SP_positive, SP_negative = model(graph, patches, device)

            batch_loss = criterion(pred_class, label)
            # batch_loss = criterion(pred_class, label, MG_embedding, SP_embeddings, SP_positive, SP_negative)

            acc1, acc3 = accuracy(pred_class, label, topk=(1, 3))
            losses.update(batch_loss.item(), BATCH_SIZE)
            top1.update(acc1.item(), BATCH_SIZE)
            top3.update(acc3.item(), BATCH_SIZE)

            batch_loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

            if sampler % (len(dataloader) // 5) == 0:
                tqdm.write(progress.get_string(sampler))

        plot_grad_flow(model.named_parameters())

        print(f"Epoch {epoch + 1}/{epochs} ({int(time() - start)}s):"
              f" Epoch Loss={losses.avg:.3f}")

        mesh_graph_classification_statistics, mesh_classification_statistics = testMeshNetworkBatch(model=model, dataset=dataset, test_mask=test_mask, criterion=criterion, batch_size=BATCH_SIZE, device=device)
        mgraph_acc1, mgraph_acc3, mgraph_loss, mgraph_cm = mesh_graph_classification_statistics
        mesh_acc, mesh_cm, mesh_data = mesh_classification_statistics

        val_accuracies.append(mgraph_acc1)
        mesh_val_accuracies.append(mesh_acc)
        val_losses.append(mgraph_loss)
        print(f"Validation Test\n"
              f"Mesh Graph Acc. : {mgraph_acc1}\n"
              f"Mesh Graph Loss.: {mgraph_loss}\n"
              f"Mesh Acc.: {mesh_acc}")

        if mgraph_acc1 > best_acc:
            best_acc = mgraph_acc1
            model.save(f"{PATH}/MeshNetworkBestAcc")
            save_confusion_matrix(mgraph_cm, f"{PATH}/MeshNetworkBestAcc", "MeshGraphConfusionMatrix.png")
            save_confusion_matrix(mesh_cm, f"{PATH}/MeshNetworkBestAcc", "MeshConfusionMatrix.png")
            plot_voting_confidences(mesh_data, f"{PATH}/MeshNetworkBestAcc", f"Confidence_statistics.png")
            with open(f'{PATH}/MeshNetworkBestAcc/bestAcc.txt', 'w') as f:
                f.write(f'Acc: {np.trunc(best_acc * 10000) / 100}\n'
                        f'Epoch:{epoch}\n'
                        f'Loss:{mgraph_loss}')

        if mgraph_loss < best_loss:
            best_loss = mgraph_loss
            # model.save(f"{PATH/MeshNetworkBestLoss")
            save_confusion_matrix(mgraph_cm, f"{PATH}/MeshNetworkBestLoss", "MeshGraphConfusionMatrix.png")
            save_confusion_matrix(mesh_cm, f"{PATH}/MeshNetworkBestLoss", "MeshConfusionMatrix.png")
            plot_voting_confidences(mesh_data, f"{PATH}/MeshNetworkBestLoss", f"Confidence_statistics.png")

        if mesh_acc > best_mesh_acc:
            best_mesh_acc = mesh_acc

        # scheduler.step(loss)
        scheduler.step()
        plot_training_statistics(f"{PATH}", f"Train_MeshGraph_statistics.png", title=f"Best Acc: {np.trunc(best_acc) / 100}, Loss: {best_loss}", epochs=range(epoch + 1), losses=train_losses, val_epochs=range(epoch + 1), val_accuracies=np.array(val_accuracies) / 100,
                                 val_losses=val_losses)
        plot_training_statistics(f"{PATH}", f"Train_Mesh_statistics.png", title=f"Best Acc: {best_mesh_acc}", epochs=range(epoch + 1), losses=train_losses, val_accuracies=np.array(mesh_val_accuracies) / 100, val_epochs=range(epoch + 1))


def trainBLOCKMeshNetwork(model, dataset, train_mask, test_mask, block_size, device, epochs=1000, train_name="", debug=True, optimizer_option="all"):
    rng = np.random.default_rng(22)

    if not debug:
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.emit_nvtx(enabled=False)
        torch.autograd.profiler.profile(enabled=False)

    model.train()
    best_acc = 0
    best_loss = 1000

    if optimizer_option == "all":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=10e-4)

    elif optimizer_option == "patch":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.patch_embedder.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.patch_embedder.parameters(), lr=1e-4, weight_decay=10e-4)

    elif optimizer_option == "mesh":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.mesh_embedder.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.mesh_embedder.parameters(), lr=1e-4, weight_decay=10e-4)
    else:
        raise ()

    # scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.70), int(epochs * 0.85), int(epochs * 0.95)], gamma=0.1)

    start = time()
    train_losses = []
    mesh_val_accuracies = []
    mesh_val_losses = []

    for epoch in range(epochs):
        train_losses.append(0)
        rng.shuffle(train_mask.reshape((-1, 5)))
        for i in tqdm(range(0, len(dataset.graphs[train_mask]), block_size), position=0, leave=False, desc=f"Epoch {epoch + 1}: ", colour="white", ncols=80):
            block = [(dataset.graphs[train_mask][i + j], dataset.graphs[train_mask][i + j].patches) for j in range(block_size)]
            rng.shuffle(block)
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

        # scheduler.step(train_losses[-1])
        # scheduler.step()
        plot_training_statistics(f"../TrainedModels/Train_{train_name}", f"Train_MeshGraph_statistics.png", title=f"Best Acc: {np.trunc(best_acc * 10000) / 100}, Loss: {best_loss}", epochs=range(epoch + 1), losses=train_losses, val_epochs=range(epoch + 1), val_accuracies=mesh_val_accuracies,
                                 val_losses=mesh_val_losses)


def trainBLOCKMeshNetworkAIO(model, dataset, train_mask, test_mask, block_size, device, epochs=None, train_name="", debug=True):
    rng = np.random.default_rng(22)

    if epochs is None:
        epochs = [25, 30, 35, 40]

    if not debug:
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.emit_nvtx(enabled=False)
        torch.autograd.profiler.profile(enabled=False)

    model.train()
    best_acc = 0
    best_loss = 1000
    train_losses = []
    mesh_val_accuracies = []
    mesh_val_losses = []
    start = time()

    for epoch in range(epochs[-1]):
        if epoch < epochs[0]:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=10e-4)
        elif epochs[0] <= epoch < epochs[1]:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.patch_embedder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.patch_embedder.parameters(), lr=1e-4, weight_decay=10e-4)

        elif epochs[1] <= epoch < epochs[2]:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.mesh_embedder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.mesh_embedder.parameters(), lr=1e-4, weight_decay=10e-4)
        else:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=10e-4)

        train_losses.append(0)
        rng.shuffle(train_mask.reshape((-1, 5)))
        for i in tqdm(range(0, len(dataset.graphs[train_mask]), block_size), position=0, leave=False, desc=f"Epoch {epoch + 1}: ", colour="white", ncols=80):
            block = [(dataset.graphs[train_mask][i + j], dataset.graphs[train_mask][i + j].patches) for j in range(block_size)]
            rng.shuffle(block)
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
        print(f"Epoch {epoch + 1}/{epochs[-1]} ({int(time() - start)}s):"
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

            with open(f'../TrainedModels/Train_{train_name}/MeshNetworkBestAcc/bestAcc.txt', 'w') as f:
                f.write(f'Acc: {np.trunc(best_acc * 10000) / 100}\n'
                        f'Epoch:{epoch}\n'
                        f'Loss:{mesh_class_loss}')

        plot_training_statistics(f"../TrainedModels/Train_{train_name}", f"Train_MeshGraph_statistics.png", title=f"Best Acc: {np.trunc(best_acc * 10000) / 100}, Loss: {best_loss}", epochs=range(epoch + 1), losses=train_losses, val_epochs=range(epoch + 1), val_accuracies=mesh_val_accuracies,
                                 val_losses=mesh_val_losses)


def testMeshNetwork(model, dataset, test_mask, criterion, batch_size, device):
    model.eval()
    predicted_labels = np.empty(0)
    loss_confidences = np.empty((0, dataset.numClasses()))
    dataloader = GraphDataLoader(dataset[test_mask], batch_size=batch_size, drop_last=False)

    losses = AverageMeter('Loss', ':.6f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')

    with torch.no_grad():
        for sampler, (graph, label) in enumerate(dataloader):
            pred = model(graph, dataset.graphs[test_mask][sampler].patches, device)
            loss = criterion(pred, label)

            acc1, acc3 = accuracy(pred, label, topk=(1, 3))
            # predicted_labels = np.hstack((predicted_labels, pred.cpu().argmax(dim=1)))
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

            pred = pred.cpu()
            predicted_labels = np.hstack((predicted_labels, pred.argmax(dim=1)))  # Take the highest value in the predicted classes vector
            loss_confidences = np.vstack((loss_confidences, pred.detach().numpy()))

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

    confusion_matrix = metrics.confusion_matrix(dataset.labels[test_mask].cpu(), predicted_labels.tolist(), normalize="true")
    model.train()
    graph_statistics = (top1.avg, top3.avg, losses.avg, confusion_matrix)
    return graph_statistics, compute_voting_scores(mesh_predicted_labels)


def testMeshNetworkBatch(model, dataset, test_mask, criterion, batch_size, device):
    model.eval()
    predicted_labels = np.empty(0)
    loss_confidences = np.empty((0, dataset.numClasses()))
    dataloader = GraphDataLoader(dataset[test_mask], batch_size=batch_size, drop_last=False)

    losses = AverageMeter('Loss', ':.6f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')

    with torch.no_grad():
        for sampler, (graph, label) in enumerate(dataloader):
            patch_sampler = sampler + np.arange(0, len(label))
            patches = np.empty(0, dtype=object)
            for idx in patch_sampler:
                patches = np.hstack((patches, dataset.graphs[test_mask[idx]].patches))
            pred, MG_embedding, SP_embeddings, SP_positive, SP_negative = model(graph, patches, device)

            if isinstance(criterion, GATCriterion):
                loss = criterion(pred, label, MG_embedding, SP_embeddings, SP_positive, SP_negative)
            else:
                loss = criterion(pred, label)

            acc1, acc3 = accuracy(pred, label, topk=(1, 3))
            # predicted_labels = np.hstack((predicted_labels, pred.cpu().argmax(dim=1)))
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

            pred = pred.cpu()
            predicted_labels = np.hstack((predicted_labels, pred.argmax(dim=1)))  # Take the highest value in the predicted classes vector
            loss_confidences = np.vstack((loss_confidences, pred.detach().numpy()))

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

    confusion_matrix = metrics.confusion_matrix(dataset.labels[test_mask].cpu(), predicted_labels.tolist(), normalize="true")
    model.train()
    graph_statistics = (top1.avg, top3.avg, losses.avg, confusion_matrix)
    return graph_statistics, compute_voting_scores(mesh_predicted_labels)


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
