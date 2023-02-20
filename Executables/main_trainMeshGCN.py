import os
import re
import shutil
import sys
import uuid
import warnings
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from scipy.special import softmax
from sklearn import metrics
from torch import nn
from tqdm import tqdm

from Networks.CONVMeshGraphModules import CONVMGEmbedder
from Networks.CONVSpiderPatchModules import CONVSPEmbedder
from Networks.GATMeshGraphModules import GATMGEmbedder
from Networks.GATNetworks import TestNetwork
from Networks.GATSpiderPatchModules import GATSPEmbedder
from Networks.Losses import CETripletMG, TripletMGSP, TripletMG
from PlotUtils import save_confusion_matrix, plot_training_statistics, plot_grad_flow
from SpiderDatasets.MeshGraphDataset import MeshGraphDataset
from Utils import AverageMeter, ProgressMeter, accuracy


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for name in ["SHREC17_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES0", "SHREC17_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES1", "SHREC17_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES2", "SHREC17_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES3"]:

        dataset_name = name

        features = np.array(["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"])
        features_to_keep = [0, 1, 2, 3, 4]
        radius_to_keep = [0, 1, 2, 3, 4]

        experiment_dict = {}

        for i in range(len(features)):
            if i in features_to_keep:
                experiment_dict[features[i]] = [True]
            else:
                experiment_dict[features[i]] = [False]

        for j in range(len(features)):
            if j in radius_to_keep:
                experiment_dict[f"features_radius{j}"] = [True]
            else:
                experiment_dict[f"features_radius{j}"] = [False]

        dataset = MeshGraphDataset(dataset_name=dataset_name)
        dataset.load_from(f"../Datasets/MeshGraphs", dataset_name)

        sp_radius = re.search('_R(.*?)_', dataset_name).group(1)
        rings = int(re.search('_R(\d)_P', dataset_name).group(1))
        points = int(re.search('_P(\d)_C', dataset_name).group(1))

        experiment_dict["SP_radius"] = [sp_radius]
        experiment_dict["SP_rings"] = [rings]
        experiment_dict["SP_points"] = [points]
        experiment_dict["dataset_MG_num"] = [int(re.search('MGRAPHS(\d*)_', dataset_name).group(1))]
        experiment_dict["SP_per_MG"] = [int(re.search('_SPIDER(\d*)_', dataset_name).group(1))]
        experiment_dict["MG_connectivity"] = [int(re.search('_CONN(\d*)_', dataset_name).group(1))]

        experiment_dict["mesh_resolution"] = [re.search('_RES(\d*)', dataset_name).group(1)]
        experiment_dict["dataset_name"] = [dataset_name]

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

        experiment_dict["normalization_mode"] = mode
        experiment_dict["normalization_elim_mode"] = elim_mode if elim_mode is not None else "None"

        dataset.normalizeV2(train_mask, mode, elim_mode)
        # dataset.normalize_edge(train_mask)
        # plot_data_distributions(dataset, train_mask)

        print(dataset.graphs[0].patches[0].node_attr_schemes())

        feat_in_channels = dataset.graphs[0].patches[0].ndata["aggregated_feats"].shape[1]
        weights_in_channels = dataset.graphs[0].patches[0].ndata["aggregated_weights"].shape[1]
        epochs = 70

        network_parameters = {}

        ####  SPIDER PATCH PARAMETERS  ####
        network_parameters["SP"] = {}
        network_parameters["SP"]["module"] = GATSPEmbedder  # [ CONVSPEmbedder, GATSPEmbedder , GATWeightedSP ]
        network_parameters["SP"]["readout_function"] = "AR"  # [ "AR" , "UR" ]
        network_parameters["SP"]["jumping_mode"] = "cat"  # [ None, "lstm", "max", "cat"]
        network_parameters["SP"]["layers_num"] = 4
        network_parameters["SP"]["dropout"] = 0

        # GAT params
        network_parameters["SP"]["residual"] = True  # bool
        network_parameters["SP"]["exp_heads"] = False  # bool

        # Node Weigher params
        network_parameters["SP"]["weigher_mode"] = "attn_weights+feats"  # [ "sp_weights", "attn_weights",  "attn_weights+feats" , None ]

        ####  MESH GRAPH PARAMETERS  ####
        network_parameters["MG"] = {}
        network_parameters["MG"]["module"] = GATMGEmbedder  # [ CONVMGEmbedder, GATMGEmbedder]
        network_parameters["MG"]["readout_function"] = "AR"  # [ "AR" , "UR" ]
        network_parameters["MG"]["jumping_mode"] = "cat"  # [ None, "lstm", "max", "cat"]
        network_parameters["MG"]["layers_num"] = 3
        network_parameters["MG"]["dropout"] = 0
        network_parameters["MG"]["SP_batch_size"] = 512

        # GAT params
        network_parameters["MG"]["residual"] = True  # bool
        network_parameters["MG"]["exp_heads"] = False  # bool

        for structure in ["MG", "SP"]:
            for key, value in network_parameters[structure].items():
                experiment_dict[f"{structure}_{key}"] = [value]

        model = TestNetwork(feat_in_channels, weights_in_channels, class_num, network_parameters=network_parameters, use_SP_triplet=False)
        model.to(device)

        #### TRAINING PARAMETERS  ####
        experiment_dict["MG_batch_size"] = 128
        experiment_dict["criterion"] = CETripletMG  # [nn.CrossEntropyLoss, TripletMG, CETripletMG]
        trainNetwork(model, dataset, train_mask, test_mask, device, experiment_dict, epochs)


def trainNetwork(model, dataset, train_mask, test_mask, device, experiment_dict, epochs=1000, optimizer_option="all"):
    rng = np.random.default_rng(717)

    # Added to not use 100% of CPU during the calculus of SpiderPatch embeddings in the network (The copy of a tensor is made in parallel in pytorch)
    torch.set_num_threads(6)

    RESULTS_TABLE_PATH = "../TrainingResults"
    RESULTS_TABLE_NAME = "ExperimentsTable"
    RESULTS_TABLE_BACKUP = "../TrainingResults/TableBackups"
    RESULTS_SAVE_PATH = "../TrainingResults/Experiments"

    EXPERIMENT_NAME = datetime.now().strftime('%d%m%Y-%H%M%S')

    if os.path.exists(f"{RESULTS_TABLE_PATH}/{RESULTS_TABLE_NAME}.csv"):
        os.makedirs(f"{RESULTS_TABLE_BACKUP}", exist_ok=True)
        shutil.copy2(f"{RESULTS_TABLE_PATH}/{RESULTS_TABLE_NAME}.csv", f"{RESULTS_TABLE_BACKUP}/TableBackup{EXPERIMENT_NAME}.csv")
        results_df = pd.read_csv(f"{RESULTS_TABLE_PATH}/{RESULTS_TABLE_NAME}.csv", index_col=0)
    else:
        results_df = pd.DataFrame()

    best_acc = 0
    best_loss = 1000

    BATCH_SIZE = experiment_dict["MG_batch_size"]
    CRITERION = experiment_dict["criterion"]()

    # scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.40), int(epochs * 0.85), int(epochs * 0.95)], gamma=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, threshold=0.01)

    if optimizer_option == "all":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

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

    start = time()
    train_losses = []
    val_accuracies = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_mask)
        dataloader = GraphDataLoader(dataset[train_mask], batch_size=BATCH_SIZE, drop_last=False, shuffle=False)

        batch_time = AverageMeter('Time', ':6.3f')

        if isinstance(CRITERION, TripletMGSP):
            loss_meters = None  # TODO implementare
        elif isinstance(CRITERION, CETripletMG):
            loss_meters = [AverageMeter('CE+TRI_Loss', ':.6f'), AverageMeter('CE_Loss', ':.6f'), AverageMeter('MGTRI_Loss', ':.6f')]
        elif isinstance(CRITERION, TripletMG):
            loss_meters = [AverageMeter('TRI_Loss', ':.6f')]
        else:
            loss_meters = [AverageMeter('CE_Loss', ':.6f')]

        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        progress = ProgressMeter(
            len(dataloader),
            [batch_time] + loss_meters + [top1, top3],
            prefix='Training: ')

        end = time()

        for sampler, (batched_MG, labels) in enumerate(tqdm(dataloader, position=0, leave=False, desc=f"Epoch {epoch + 1}: ", colour="white", ncols=80)):

            # To GPU memory
            batched_MG = batched_MG.to(device)
            labels = labels.to(device)

            # Prepare the batched SpiderPatches
            SP_sampler = np.arange((sampler * BATCH_SIZE), (sampler * BATCH_SIZE) + len(labels))
            spider_patches = [sp.to(device) for idx in SP_sampler for sp in dataset.graphs[train_mask][idx].patches]

            # Optimization step
            optimizer.zero_grad()

            if isinstance(CRITERION, TripletMGSP):
                pred_class, MG_embedding, SP_embeddings, SP_positive, SP_negative = model(batched_MG, spider_patches, device)
                batch_loss = CRITERION(pred_class, labels, MG_embedding, SP_embeddings, SP_positive, SP_negative)
            elif isinstance(CRITERION, CETripletMG):
                pred_class, MG_embedding = model(batched_MG, spider_patches, device)
                batch_loss = CRITERION(pred_class, labels, MG_embedding)
            elif isinstance(CRITERION, TripletMG):
                pred_class, MG_embedding = model(batched_MG, spider_patches, device)
                batch_loss = [CRITERION(labels, MG_embedding)]
            else:
                pred_class, _ = model(batched_MG, spider_patches, device)
                batch_loss = [CRITERION(pred_class, labels)]

            acc1, acc3 = accuracy(pred_class, labels, topk=(1, 3))
            for idx, loss_meter in enumerate(loss_meters):
                loss_meter.update(batch_loss[idx].item(), batched_MG.batch_size)
            top1.update(acc1.item(), batched_MG.batch_size)
            top3.update(acc3.item(), batched_MG.batch_size)

            batch_loss[0].backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

            if sampler % 10 == 0:
                tqdm.write(progress.get_string(sampler))

        # plot_grad_flow(model.named_parameters())
        train_losses.append(loss_meters[0].avg)

        experiment_dict["elapsed_epochs"] = epoch + 1
        experiment_dict["training_loss"] = train_losses[-1]

        to_write = f"Epoch {epoch + 1}/{epochs} ({int(time() - start)}s) -->"
        for loss_meter in loss_meters:
            to_write += f"{loss_meter.name}: {loss_meter.avg:.3f}\t"
        tqdm.write(to_write)

        mesh_graph_classification_statistics = testNetwork(model=model, dataset=dataset, test_mask=test_mask, criterion=CRITERION, batch_size=BATCH_SIZE, device=device)
        test_acc1, test_acc3, test_losses, cm = mesh_graph_classification_statistics

        val_accuracies.append(test_acc1)
        val_losses.append(test_losses[0].avg)

        to_write = f"Validation Test\nMesh Graph Acc. : {test_acc1}"
        for loss_meter in test_losses:
            to_write += f"   {loss_meter.name}: {loss_meter.avg:.3f}"
        tqdm.write(to_write)

        if test_acc1 > best_acc:
            best_acc = test_acc1
            experiment_dict["best_val_acc"] = best_acc
            experiment_dict["best_val_acc_epoch"] = epoch
            # model.save(f'{RESULTS_SAVE_PATH}/{EXPERIMENT_NAME}/MeshNetworkBestAcc')
            save_confusion_matrix(cm, f'{RESULTS_SAVE_PATH}/{EXPERIMENT_NAME}/MeshNetworkBestAcc', "MeshGraphConfusionMatrix.png")
            with open(f'{RESULTS_SAVE_PATH}/{EXPERIMENT_NAME}/MeshNetworkBestAcc/bestAcc.txt', 'w') as f:
                f.write(f'Epoch:{epoch}\n' + to_write)

        if test_losses[0].avg < best_loss:
            best_loss = test_losses[0].avg
            experiment_dict["best_val_loss"] = best_loss
            experiment_dict["best_val_loss_epoch"] = epoch
            # model.save(f"{RESULTS_SAVE_PATH}/{EXPERIMENT_NAME}/MeshNetworkBestLoss")
            save_confusion_matrix(cm, f"{RESULTS_SAVE_PATH}/{EXPERIMENT_NAME}/MeshNetworkBestLoss", "MeshGraphConfusionMatrix.png")
            with open(f"{RESULTS_SAVE_PATH}/{EXPERIMENT_NAME}/MeshNetworkBestLoss/bestLoss.txt", 'w') as f:
                f.write(f'Epoch:{epoch}\n' + to_write)

        # scheduler.step(loss)
        # scheduler.step()
        plot_training_statistics(f"{RESULTS_SAVE_PATH}/{EXPERIMENT_NAME}", f"Train_MeshGraph_statistics.png", title=f"Best Acc: {np.trunc(best_acc) / 100}, Loss: {best_loss}", epochs=range(epoch + 1), losses=train_losses, val_epochs=range(epoch + 1), val_accuracies=np.array(val_accuracies) / 100,
                                 val_losses=val_losses)

        df = pd.DataFrame.from_dict(experiment_dict)
        df.index = [EXPERIMENT_NAME]

        if epoch == 0:
            results_df = pd.concat([results_df, df])
        else:
            results_df.loc[EXPERIMENT_NAME] = df.loc[EXPERIMENT_NAME]

        results_df.to_csv(f"{RESULTS_TABLE_PATH}/{RESULTS_TABLE_NAME}.csv", index=True)
        results_df.to_excel(f"{RESULTS_TABLE_PATH}/{RESULTS_TABLE_NAME}.xlsx", index=True)


def testNetwork(model, dataset, test_mask, criterion, batch_size, device):
    model.eval()
    predicted_labels = np.empty(0)
    loss_confidences = np.empty((0, dataset.numClasses()))
    dataloader = GraphDataLoader(dataset[test_mask], batch_size=batch_size, drop_last=False, shuffle=False)

    if isinstance(criterion, TripletMGSP):
        loss_meters = None  # TODO implementare
    elif isinstance(criterion, CETripletMG):
        loss_meters = [AverageMeter('CE+TRI_Loss', ':.6f'), AverageMeter('CE_Loss', ':.6f'), AverageMeter('MGTRI_Loss', ':.6f')]
    elif isinstance(criterion, TripletMG):
        loss_meters = [AverageMeter('TRI_Loss', ':.6f')]
    else:
        loss_meters = [AverageMeter('CE_Loss', ':.6f')]

    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')

    with torch.no_grad():
        for sampler, (batched_MG, labels) in enumerate(dataloader):
            # To GPU memory
            batched_MG = batched_MG.to(device)
            labels = labels.to(device)

            # Prepare the batched SpiderPatches
            SP_sampler = np.arange((sampler * batch_size), (sampler * batch_size) + len(labels))
            spider_patches = [sp.to(device) for idx in SP_sampler for sp in dataset.graphs[test_mask][idx].patches]

            if isinstance(criterion, TripletMGSP):
                pred, MG_embedding, SP_embeddings, SP_positive, SP_negative = model(batched_MG, spider_patches, device)
                batched_loss = criterion(pred, labels, MG_embedding, SP_embeddings, SP_positive, SP_negative)
            elif isinstance(criterion, CETripletMG):
                pred, MG_embedding = model(batched_MG, spider_patches, device)
                batched_loss = criterion(pred, labels, MG_embedding)
            elif isinstance(criterion, TripletMG):
                pred, MG_embedding = model(batched_MG, spider_patches, device)
                batched_loss = [criterion(labels, MG_embedding)]
            else:
                pred, _ = model(batched_MG, spider_patches, device)
                batched_loss = [criterion(pred, labels)]

            acc1, acc3 = accuracy(pred, labels, topk=(1, 3))
            # predicted_labels = np.hstack((predicted_labels, pred.cpu().argmax(dim=1)))
            for idx, loss_meter in enumerate(loss_meters):
                loss_meter.update(batched_loss[idx].item(), batched_MG.batch_size)
            top1.update(acc1.item(), batched_MG.batch_size)
            top3.update(acc3.item(), batched_MG.batch_size)

            pred = pred.cpu()
            predicted_labels = np.hstack((predicted_labels, pred.argmax(dim=1)))  # Take the highest value in the predicted classes vector
            loss_confidences = np.vstack((loss_confidences, pred.detach().numpy()))

    confusion_matrix = metrics.confusion_matrix(dataset.labels[test_mask], predicted_labels.tolist(), normalize="true")
    graph_statistics = (top1.avg, top3.avg, loss_meters, confusion_matrix)
    model.train()
    return graph_statistics


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


def compute_voting_scores(dataset, test_mask, loss_confidences):
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

    dict_to_plot = {"data": {}, "colors": []}
    pred_labels = []
    true_labels = []
    color_sample = {0: "red", 1: "green", 2: "blue"}
    for mesh_id, internal_dict in mesh_predicted_labels.items():
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


# def trainNetworkWithVoting(model, dataset, train_mask, test_mask, device, epochs=1000, train_name="", optimizer_option="all"):
#     PATH = f"../TrainedModels/Train_{train_name}"
#
#     rng = np.random.default_rng(717)
#     model.train()
#
#     best_acc = 0
#     best_mesh_acc = 0
#     best_loss = 1000
#
#     BATCH_SIZE = 32
#
#     if optimizer_option == "all":
#         optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
#
#     elif optimizer_option == "patch":
#         for param in model.parameters():
#             param.requires_grad = False
#         for param in model.patch_embedder.parameters():
#             param.requires_grad = True
#         optimizer = torch.optim.AdamW(model.patch_embedder.parameters())
#
#     elif optimizer_option == "mesh":
#         for param in model.parameters():
#             param.requires_grad = False
#         for param in model.mesh_embedder.parameters():
#             param.requires_grad = True
#         optimizer = torch.optim.AdamW(model.mesh_embedder.parameters())
#     else:
#         raise ()
#
#     criterion = GATTripletMG()
#     # criterion = GATCriterion()
#     # criterion = nn.CrossEntropyLoss()
#     scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.40), int(epochs * 0.85), int(epochs * 0.95)], gamma=0.1)
#     # scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, threshold=0.01)
#
#     start = time()
#     train_losses = []
#     val_accuracies = []
#     mesh_val_accuracies = []
#     val_losses = []
#
#     for epoch in range(epochs):
#         rng.shuffle(train_mask)
#         dataloader = GraphDataLoader(dataset[train_mask], batch_size=BATCH_SIZE, drop_last=False, shuffle=False)
#
#         batch_time = AverageMeter('Time', ':6.3f')
#         losses = AverageMeter('Loss', ':.6f')
#         top1 = AverageMeter('Acc@1', ':6.2f')
#         top3 = AverageMeter('Acc@3', ':6.2f')
#         progress = ProgressMeter(
#             len(dataloader),
#             [batch_time, losses, top1, top3],
#             prefix='Training: ')
#
#         train_losses.append(0)
#         end = time()
#
#         for sampler, (graph, label) in enumerate(tqdm(dataloader, position=0, leave=False, desc=f"Epoch {epoch + 1}: ", colour="white", ncols=80)):
#             patch_sampler = np.arange((sampler * BATCH_SIZE), (sampler * BATCH_SIZE) + len(label))
#             patches = np.empty(0, dtype=object)
#             for idx in patch_sampler:
#                 patches = np.hstack((patches, dataset.graphs[train_mask][idx].patches))
#             optimizer.zero_grad()
#             pred_class, MG_embedding = model(graph, patches, device)
#             # pred_class, MG_embedding, SP_embeddings, SP_positive, SP_negative = model(graph, patches, device)
#
#             # batch_loss = criterion(pred_class, label)
#             batch_loss = criterion(pred_class, label, MG_embedding)
#             # batch_loss = criterion(pred_class, label, MG_embedding, SP_embeddings, SP_positive, SP_negative)
#
#             acc1, acc3 = accuracy(pred_class, label, topk=(1, 3))
#             losses.update(batch_loss.item(), graph.batch_size)
#             top1.update(acc1.item(), graph.batch_size)
#             top3.update(acc3.item(), graph.batch_size)
#
#             batch_loss.backward()
#             optimizer.step()
#
#             # measure elapsed time
#             batch_time.update(time() - end)
#             end = time()
#
#             if sampler % (len(dataloader) // 5) == 0:
#                 tqdm.write(progress.get_string(sampler))
#
#         plot_grad_flow(model.named_parameters())
#
#         print(f"Epoch {epoch + 1}/{epochs} ({int(time() - start)}s):"
#               f" Epoch Loss={losses.avg:.3f}")
#
#         mesh_graph_classification_statistics, mesh_classification_statistics = testNetwork(model=model, dataset=dataset, test_mask=test_mask, criterion=criterion, batch_size=BATCH_SIZE, device=device)
#         mgraph_acc1, mgraph_acc3, mgraph_loss, mgraph_cm = mesh_graph_classification_statistics
#         mesh_acc, mesh_cm, mesh_data = mesh_classification_statistics
#
#         val_accuracies.append(mgraph_acc1)
#         mesh_val_accuracies.append(mesh_acc)
#         val_losses.append(mgraph_loss)
#         print(f"Validation Test\n"
#               f"Mesh Graph Acc. : {mgraph_acc1}\n"
#               f"Mesh Graph Loss.: {mgraph_loss}\n"
#               f"Mesh Acc.: {mesh_acc}")
#
#         if mgraph_acc1 > best_acc:
#             best_acc = mgraph_acc1
#             model.save(f"{PATH}/MeshNetworkBestAcc")
#             save_confusion_matrix(mgraph_cm, f"{PATH}/MeshNetworkBestAcc", "MeshGraphConfusionMatrix.png")
#             save_confusion_matrix(mesh_cm, f"{PATH}/MeshNetworkBestAcc", "MeshConfusionMatrix.png")
#             plot_voting_confidences(mesh_data, f"{PATH}/MeshNetworkBestAcc", f"Confidence_statistics.png")
#             with open(f'{PATH}/MeshNetworkBestAcc/bestAcc.txt', 'w') as f:
#                 f.write(f'Acc: {np.trunc(best_acc * 10000) / 100}\n'
#                         f'Epoch:{epoch}\n'
#                         f'Loss:{mgraph_loss}')
#
#         if mgraph_loss < best_loss:
#             best_loss = mgraph_loss
#             # model.save(f"{PATH/MeshNetworkBestLoss")
#             save_confusion_matrix(mgraph_cm, f"{PATH}/MeshNetworkBestLoss", "MeshGraphConfusionMatrix.png")
#             save_confusion_matrix(mesh_cm, f"{PATH}/MeshNetworkBestLoss", "MeshConfusionMatrix.png")
#             plot_voting_confidences(mesh_data, f"{PATH}/MeshNetworkBestLoss", f"Confidence_statistics.png")
#
#         if mesh_acc > best_mesh_acc:
#             best_mesh_acc = mesh_acc
#
#         # scheduler.step(loss)
#         scheduler.step()
#         plot_training_statistics(f"{PATH}", f"Train_MeshGraph_statistics.png", title=f"Best Acc: {np.trunc(best_acc) / 100}, Loss: {best_loss}", epochs=range(epoch + 1), losses=train_losses, val_epochs=range(epoch + 1), val_accuracies=np.array(val_accuracies) / 100,
#                                  val_losses=val_losses)
#         plot_training_statistics(f"{PATH}", f"Train_Mesh_statistics.png", title=f"Best Acc: {best_mesh_acc}", epochs=range(epoch + 1), losses=train_losses, val_accuracies=np.array(mesh_val_accuracies) / 100, val_epochs=range(epoch + 1))
#


if __name__ == "__main__":
    main()
