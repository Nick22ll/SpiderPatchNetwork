import gc
import multiprocessing
import re
import sys

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from Executables.main_DatasetGenerator import generateMeshGraphDatasetFromPatches
from SpiderDatasets.RetrievalDataset import RetrievalDataset, generate_mesh, generate_labels
from SpiderDatasets.SpiderPatchDataset import SpiderPatchDataset
from Networks.MLP import MLP
import warnings
from time import time
import pickle as pkl
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from PlotUtils import save_confusion_matrix, plot_training_statistics
from sklearn import metrics
import torch.nn.functional as F
from Networks.CONVNetworks import *
from tqdm import tqdm
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import GridSearchCV

from sklearn import svm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, roc_auc_score, f1_score

from SpiderPatch.SpiderPatch import SpiderPatch


def set_node_weights(dataset):
    for patch in dataset.graphs:
        weights = []
        for vertex in patch.ndata["vertices"]:
            weights.append(1 - np.linalg.norm(vertex - patch.seed_point))
        patch.ndata["weight"] = torch.tensor(np.array(weights).reshape((-1, 1)), dtype=torch.float32)
    dataset.save_to(f"Datasets")


def retrieve(dataset, model):
    # with open("Mesh/egyptFaceDenseBC.pkl", "rb") as mesh_file:
    #     mesh = pkl.load(mesh_file)
    # label_colors = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1], 3: [0, 0, 0]}
    # point_cloud = np.empty((0, 3))
    # colors = np.empty((0, 3))
    # for id, graph in enumerate(dataset.graphs):
    #     point_cloud = np.vstack((point_cloud, graph.seed_point))
    #     colors = np.vstack((colors, label_colors[dataset.labels[id]]))
    # mesh.drawWithPointCloud(point_cloud,colors)

    model.eval()

    correct_prediction_number = np.empty(0)
    loss_predicted = np.empty((0, dataset.numClasses()))
    dataloader = GraphDataLoader(dataset, batch_size=100, drop_last=False)
    for graph, _ in tqdm(dataloader):
        pred = model(graph)  # , graph.edata["weights"] , graph.ndata["aggregated_feats"]
        pred = pred.cpu()
        correct_prediction_number = np.hstack((correct_prediction_number, pred.argmax(dim=1)))  # Take the highest value in the predicted classes vector
        loss_predicted = np.vstack((loss_predicted, pred.detach().numpy()))
    return compute_scores(dataset, correct_prediction_number, loss_predicted)


def drawTrainResultsOnMesh(mesh, model, X_test, y_test):  # TODO da fare
    label_colors = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1], 3: [0, 0, 0]}
    # point_cloud = np.empty((0, 3))
    # colors = np.empty((0, 3))
    # for id, graph in enumerate(dataset.graphs):
    #     point_cloud = np.vstack((point_cloud, graph.seed_point))
    #     colors = np.vstack((colors, label_colors[correct_prediction_number[id]]))


def trainRetrieve(model, train_dataset, test_dataset, epochs=500, train_name="", loss_precision=0.01, learning_rate=0.00001):
    os.makedirs(f"../Retrieval/TrainedModels/{train_name}", exist_ok=True)
    model.train()
    best_acc = 0
    best_loss = 1000
    dataloader = GraphDataLoader(train_dataset, batch_size=20, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.60), int(epochs * 0.85)], gamma=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, threshold=1e-4)
    start = time()
    train_losses, val_epochs, val_accuracies, val_losses = [], [], [], []
    for epoch in range(epochs):
        train_losses.append(0)
        for graph, label in tqdm(dataloader, position=0, leave=False, desc=f"Epoch {epoch + 1}: ", colour="white", ncols=80):
            optimizer.zero_grad()
            pred = model(graph)  # , graph.edata["weights"] , graph.ndata["aggregated_feats"]
            loss_running = F.cross_entropy(pred, label)
            train_losses[-1] += loss_running.item()
            loss_running.backward()
            optimizer.step()
        train_losses[-1] /= len(train_dataset.graphs)
        print(f"Epoch {epoch + 1}/{epochs} ({int(time() - start)}s):"
              f" Epoch Loss={train_losses[-1]:.3f}")

        if train_losses[-1] < loss_precision or epoch == epochs - 1:
            acc, loss, cm = retrieve(test_dataset, model)
            val_accuracies.append(acc)
            val_losses.append(loss)
            val_epochs.append(epoch)
            print(f"Validation Test\n"
                  f"Acc. : {acc}\n"
                  f"Loss.: {loss}")

            if acc > best_acc:
                best_acc = acc
                os.makedirs(f"../Retrieval/TrainedModels/{train_name}/MeshNetworkBestAcc", exist_ok=True)
                model.save(f"../Retrieval/TrainedModels/{train_name}/MeshNetworkBestAcc/MLP.pt")
                save_confusion_matrix(cm, f"../Retrieval/TrainedModels/{train_name}/MeshNetworkBestAcc/ConfusionMatrixEpoch.png")

            if loss < best_loss:
                best_loss = loss
                os.makedirs(f"../Retrieval/TrainedModels/{train_name}/MeshNetworkBestLoss", exist_ok=True)
                model.save(f"../Retrieval/TrainedModels/{train_name}/MeshNetworkBestLoss/MLP.pt")
                save_confusion_matrix(cm, f"../Retrieval/TrainedModels/{train_name}/MeshNetworkBestLoss/ConfusionMatrix.png")
        # scheduler.step(train_losses[-1])
        scheduler.step()
        plot_training_statistics(path=f"../Retrieval/TrainedModels/{train_name}", filename=f"{train_name}_statistics.png", title=f"Best Acc: {np.trunc(best_acc * 10000) / 100}, Loss: {best_loss}", epochs=range(epoch + 1), val_epochs=val_epochs, losses=train_losses, val_accuracies=val_accuracies,
                                 val_losses=val_losses)


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


def trainMLP(dataset, loss_precision):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_hidden_dim = 512

    # train_samples = 5000
    # train_mask, test_mask = dataset.getTrainTestMask(train_samples)
    # train_dataset = SpiderPatchDataset(dataset.name, graphs=dataset[train_mask][0], labels=dataset[train_mask][1])
    # normalizers = train_dataset.normalize()
    # test_dataset = SpiderPatchDataset(dataset.name, graphs=dataset[test_mask][0], labels=dataset[test_mask][1])
    # test_dataset.normalize(normalizers)
    #
    # num_class = len(np.unique(train_dataset.labels))
    # epochs = 150
    #
    # train_dataset.to(device)
    # test_dataset.to(device)

    # NETWORK MODEL
    # model = PatchReader1ConvLayer(len(radius_to_keep * len(features_to_keep)), hidden_dim=256, out_feats=4, dropout=0.2)
    # model = PatchReader2ConvLayerRetrieve(len(radius_to_keep) * len(features_to_keep), 256, 64, 64 * dataset.graphs[0].num_nodes(), 2**12, 4, 0.1)
    # model = MLP(train_dataset.graphs[0].ndata["aggregated_feats"].shape[1] * train_dataset.graphs[0].num_nodes(), 256, num_class, 0)
    # model.to(device)
    #
    # trainRetrieve(model, train_dataset, test_dataset, epochs, f"Training_MLP_HID256_ALLCURV_ALLRAD_SAMPLE{train_samples}_DROP0_{dataset.name}")
    #
    # model = MLP(train_dataset.graphs[0].ndata["aggregated_feats"].shape[1] * train_dataset.graphs[0].num_nodes(), 512, num_class, 0)
    # model.to(device)
    #
    # trainRetrieve(model, train_dataset, test_dataset, epochs, f"Training_MLP_HID512_ALLCURV_ALLRAD_SAMPLE{train_samples}_DROP0_{dataset.name}")

    # model = MLP(train_dataset.graphs[0].ndata["aggregated_feats"].shape[1] * train_dataset.graphs[0].num_nodes(), 1024, num_class, 0.2)
    # model.to(device)
    #
    # trainRetrieve(model, train_dataset, test_dataset, epochs, f"Training_MLP_HID1024_ALLCURV_ALLRAD_SAMPLE{train_samples}_DROP20_{dataset.name}")

    # model = MLP(train_dataset.graphs[0].ndata["aggregated_feats"].shape[1] * train_dataset.graphs[0].num_nodes(), hidden_dim, num_class, 0)
    # model.to(device)
    #
    # trainRetrieve(model, train_dataset, test_dataset, epochs, f"Training_MLP_HID{hidden_dim}_ALLCURV_ALLRAD_SAMPLE{train_samples}_DROP0_{dataset.name}", loss_precision)

    train_samples = 30
    train_mask, test_mask = dataset.getTrainTestMask(train_samples, percentage=True)
    train_dataset = SpiderPatchDataset(dataset.name, graphs=dataset[train_mask][0], labels=dataset[train_mask][1])
    normalizers = train_dataset.normalize()
    test_dataset = SpiderPatchDataset(dataset.name, graphs=dataset[test_mask][0], labels=dataset[test_mask][1])
    test_dataset.normalize(normalizers)

    num_class = len(np.unique(train_dataset.labels))
    epochs = 150

    train_dataset.to(device)
    test_dataset.to(device)

    # # NETWORK MODEL
    # # model = PatchReader1ConvLayer(len(radius_to_keep * len(features_to_keep)), hidden_dim=256, out_feats=4, dropout=0.2)
    # # model = PatchReader2ConvLayerRetrieve(len(radius_to_keep) * len(features_to_keep), 256, 64, 64 * dataset.graphs[0].num_nodes(), 2**12, 4, 0.1)
    #
    model = MLP(train_dataset.graphs[0].ndata["aggregated_feats"].shape[1] * train_dataset.graphs[0].num_nodes(), base_hidden_dim, num_class, 0)

    model.to(device)

    trainRetrieve(model, train_dataset, test_dataset, epochs, f"Training_MLP_HID{base_hidden_dim}_ALLCURV_ALLRAD_SAMPLE{train_samples}P_DROP0_{dataset.name}_ONLYLINEAR")

    model = MLP(train_dataset.graphs[0].ndata["aggregated_feats"].shape[1] * train_dataset.graphs[0].num_nodes(), base_hidden_dim * 2, num_class, 0)
    model.to(device)

    trainRetrieve(model, train_dataset, test_dataset, epochs, f"Training_MLP_HID{base_hidden_dim * 2}_ALLCURV_ALLRAD_SAMPLE{train_samples}P_DROP0_{dataset.name}_ONLYLINEAR", loss_precision=loss_precision)

    model = MLP(train_dataset.graphs[0].ndata["aggregated_feats"].shape[1] * train_dataset.graphs[0].num_nodes(), base_hidden_dim * 4, num_class, 0.2)
    model.to(device)

    trainRetrieve(model, train_dataset, test_dataset, epochs, f"Training_MLP_HID{base_hidden_dim * 4}_ALLCURV_ALLRAD_SAMPLE{train_samples}P_DROP20_{dataset.name}_ONLYLINEAR")


def trainSVC(dataset, train_mask, test_mask):
    tmp_graph = dataset[train_mask[0]][0]
    num_features = tmp_graph.ndata["aggregated_feats"].shape[1] * tmp_graph.num_nodes()

    X_train = np.ascontiguousarray(np.empty((dataset.graphs[train_mask].shape[0], tmp_graph.ndata["aggregated_feats"].shape[1] * tmp_graph.num_nodes()), dtype=np.float32))
    y_train = np.empty(dataset.graphs[train_mask].shape[0])
    for id, graph in tqdm(enumerate(dataset.graphs[train_mask]), position=0, leave=True, desc=f"Creating training contiguos array: ", colour="white", ncols=80):
        X_train[id] = graph.ndata["aggregated_feats"].reshape((1, -1))
        y_train[id] = dataset.labels[train_mask[id]]

    X_test = np.ascontiguousarray(np.empty((dataset.graphs[test_mask].shape[0], num_features)), dtype=np.float32)
    y_test = np.empty(dataset.graphs[test_mask].shape[0])
    for id, graph in tqdm(enumerate(dataset.graphs[test_mask]), position=0, leave=True, desc=f"Creating test contiguos array: ", colour="white", ncols=80):
        X_test[id] = graph.ndata["aggregated_feats"].reshape((1, -1))
        y_test[id] = dataset.labels[test_mask[id]]

    gc.collect()

    param_grid = {'C': [10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf'], "probability": [True]}
    # grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3, n_jobs=3, cv=5)
    # grid.fit(X_train, y_train)
    # print(grid.best_estimator_)
    # print(grid.best_params_)
    #
    # sv_classifier = grid.best_estimator_
    sv_classifier = svm.SVC(C=1000, gamma=1, kernel="rbf", probability=True)
    sv_classifier.fit(X_train, y_train)
    predicted_pattern = sv_classifier.predict(X_test)
    scores = {
        "overall_acc": accuracy_score(y_test, predicted_pattern),
        "balanced_acc": balanced_accuracy_score(y_test, predicted_pattern),
        "precision": precision_score(y_test, predicted_pattern, average="weighted"),
        "recall": recall_score(y_test, predicted_pattern, average="weighted"),
        "f1_score": f1_score(y_test, predicted_pattern, average="weighted"),
    }

    try:
        if len(np.unique(y_train)) <= 2:
            scores["AUC"] = roc_auc_score(y_test, sv_classifier.predict_proba(X_test)[:, 1], average="weighted")
        else:
            scores["AUC"] = roc_auc_score(y_test, sv_classifier.predict_proba(X_test), multi_class=sv_classifier.decision_function_shape, average="weighted")

    except ValueError:
        y_test[0] = not y_test[0]
        scores["AUC"] = roc_auc_score(y_test, predicted_pattern, average="weighted")
    print(scores)
    confusion_matrix = metrics.confusion_matrix(y_test.tolist(), predicted_pattern.tolist(), normalize="true")

    os.makedirs(f"../Retrieval/TrainedModels/Training_SVC_{dataset.name}", exist_ok=True)
    pkl.dump(sv_classifier, open(f"../Retrieval/TrainedModels/Training_SVC_{dataset.name}/SVC.pkl", 'wb'))
    save_confusion_matrix(confusion_matrix, f"../Retrieval/TrainedModels/Training_SVC_{dataset.name}/ConfusionMatrixEpoch.png")


def trainRandomForest(dataset, train_mask, test_mask):
    tmp_graph = dataset[train_mask[0]][0]
    num_features = tmp_graph.ndata["aggregated_feats"].shape[1] * tmp_graph.num_nodes()

    X_train = np.ascontiguousarray(np.empty((0, tmp_graph.ndata["aggregated_feats"].shape[1] * tmp_graph.num_nodes()), dtype=np.float32))
    y_train = np.empty(0)
    for id, graph in enumerate(dataset.graphs[train_mask]):
        X_train = np.vstack((X_train, graph.ndata["aggregated_feats"].reshape((1, -1))))
        y_train = np.append(y_train, dataset.labels[train_mask[id]])

    X_test = np.ascontiguousarray(np.empty((0, num_features)), dtype=np.float32)
    y_test = np.empty(0)
    for id, graph in enumerate(dataset.graphs[test_mask]):
        X_test = np.vstack((X_test, graph.ndata["aggregated_feats"].reshape((1, -1))))
        y_test = np.append(y_test, dataset.labels[test_mask[id]])

    gc.collect()

    param_grid = {
        "criterion": ["gini", "entropy"],
        'max_depth': [None, 100],
        'max_features': ["sqrt", "log2"],
        'min_samples_leaf': [1, 3],
        'min_samples_split': [8, 12],
        'n_estimators': [100, 200],
        'class_weight': ["balanced"],
    }

    grid = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=2, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_estimator_)
    print(grid.best_params_)

    sv_classifier = grid.best_estimator_
    predicted_pattern = sv_classifier.predict(X_test)
    scores = {
        "overall_acc": accuracy_score(y_test, predicted_pattern),
        "balanced_acc": balanced_accuracy_score(y_test, predicted_pattern),
        "precision": precision_score(y_test, predicted_pattern, average="weighted"),
        "recall": recall_score(y_test, predicted_pattern, average="weighted"),
        "f1_score": f1_score(y_test, predicted_pattern, average="weighted"),
    }
    print(scores)
    confusion_matrix = metrics.confusion_matrix(y_test.tolist(), predicted_pattern.tolist(), normalize="true")

    os.makedirs(f"../Retrieval/TrainedModels/Training_RandomForest_{dataset.name}", exist_ok=True)
    pkl.dump(sv_classifier, open(f"../Retrieval/TrainedModels/Training_RandomForest_{dataset.name}/RF.pkl", 'wb'))
    save_confusion_matrix(confusion_matrix, f"../Retrieval/TrainedModels/Training_RandomForest_{dataset.name}/ConfusionMatrixEpoch.png")


def trainExtraRandomForest(dataset, train_mask, test_mask):
    tmp_graph = dataset[train_mask[0]][0]
    num_features = tmp_graph.ndata["aggregated_feats"].shape[1] * tmp_graph.num_nodes()

    X_train = np.ascontiguousarray(np.empty((dataset.graphs[train_mask].shape[0], tmp_graph.ndata["aggregated_feats"].shape[1] * tmp_graph.num_nodes()), dtype=np.float32))
    y_train = np.empty(dataset.graphs[train_mask].shape[0])
    for id, graph in enumerate(dataset.graphs[train_mask]):
        X_train[id] = graph.ndata["aggregated_feats"].reshape((1, -1))
        y_train[id] = dataset.labels[train_mask[id]]

    X_test = np.ascontiguousarray(np.empty((dataset.graphs[test_mask].shape[0], num_features)), dtype=np.float32)
    y_test = dataset.graphs[test_mask].shape[0]
    for id, graph in enumerate(dataset.graphs[test_mask]):
        X_test[id] = graph.ndata["aggregated_feats"].reshape((1, -1))
        y_test[id] = dataset.labels[test_mask[id]]

    gc.collect()

    param_grid = {
        "criterion": ["gini", "entropy"],
        'max_depth': [None, 100],
        'max_features': ["sqrt", "log2"],
        'min_samples_leaf': [1, 3],
        'min_samples_split': [8, 12],
        'n_estimators': [100, 200],
        'class_weight': ["balanced"],
    }

    grid = GridSearchCV(ExtraTreesClassifier(), param_grid, refit=True, verbose=2, cv=3, n_jobs=6)
    grid.fit(X_train, y_train)
    print(grid.best_estimator_)
    print(grid.best_params_)

    sv_classifier = grid.best_estimator_
    predicted_pattern = sv_classifier.predict(X_test)
    scores = {
        "overall_acc": accuracy_score(y_test, predicted_pattern),
        "balanced_acc": balanced_accuracy_score(y_test, predicted_pattern),
        "precision": precision_score(y_test, predicted_pattern, average="weighted"),
        "recall": recall_score(y_test, predicted_pattern, average="weighted"),
        "f1_score": f1_score(y_test, predicted_pattern, average="weighted"),
    }
    print(scores)
    confusion_matrix = metrics.confusion_matrix(y_test.tolist(), predicted_pattern.tolist(), normalize="true")

    os.makedirs(f"../Retrieval/TrainedModels/Training_ExtraTree_{dataset.name}", exist_ok=True)
    pkl.dump(sv_classifier, open(f"../Retrieval/TrainedModels/Training_ExtraTree_{dataset.name}/RF.pkl", 'wb'))
    save_confusion_matrix(confusion_matrix, f"../Retrieval/TrainedModels/Training_ExtraTree_{dataset.name}/ConfusionMatrixEpoch.png")


def fuse_datasets(d_name_list, merged_dataset_name):
    merged_dataset = RetrievalDataset(dataset_name=merged_dataset_name)
    merged_dataset.load_from(f"../Retrieval/Datasets", d_name_list[0])
    merged_dataset.mesh_id = d_name_list
    incremental_label = np.max(np.unique(merged_dataset.labels))
    for d_name in d_name_list[1:]:
        dataset = RetrievalDataset(dataset_name=d_name)
        dataset.load_from(f"../Retrieval/Datasets", d_name)
        for label in np.unique(dataset.labels):
            label_indices = np.where(dataset.labels == label)[0]
            merged_dataset.graphs = np.append(merged_dataset.graphs, dataset.graphs[label_indices])
            if len(merged_dataset.seed_point_indices) != 0 and len(dataset.seed_point_indices) != 0:
                merged_dataset.seed_point_indices = np.append(merged_dataset.seed_point_indices, dataset.seed_point_indices[label_indices])
            if label == 0:
                merged_dataset.labels = np.append(merged_dataset.labels, dataset.labels[label_indices])
            else:
                incremental_label += 1
                merged_dataset.labels = np.append(merged_dataset.labels, np.tile(incremental_label, (len(label_indices))))

    merged_dataset._name = merged_dataset_name
    merged_dataset.save_to("../Retrieval/Datasets")
    return merged_dataset


def parallel_func(mesh, labels, relative_radius=False):
    radius = 1
    rings = 4
    points = 6

    if relative_radius:
        dataset = RetrievalDataset(dataset_name=f"{mesh}_RR{radius}R{rings}P{points}_Spiral", mesh_id=mesh)
    else:
        dataset = RetrievalDataset(dataset_name=f"{mesh}_R{radius}R{rings}P{points}_Spiral", mesh_id=mesh)

    dataset.generate(f"../Retrieval/Mesh/{mesh}.pkl", f"../Retrieval/Labels/{labels}.pkl", radius, rings, points, relative_radius=relative_radius)
    dataset.save_to("../Retrieval/Datasets")


def generateSHREC17SpiderDataset(path, dataset_name, keep_resolution_level, only_homogeneous=False):
    radius = int(re.search('_R(.*\d)R', dataset_name).group(1).replace("R", ""))
    graphs = np.empty(0)
    seed_point_indices = np.empty(0)
    labels = np.empty(0)
    mesh_id = np.empty(0)
    for class_label in tqdm(os.listdir(path)):
        if only_homogeneous and int(class_label.replace("class_", "")) not in [3, 5]:  # [1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14] FIRST HOMOGENEUS    [3, 5, 7, 8, 11, 12, 13, 14]  SECOND HOMOGENEUS
            continue
        label = int(class_label.replace("class_", ""))
        for id in tqdm(os.listdir(f"{path}/{class_label}")):
            for resolution_level in os.listdir(f"{path}/{class_label}/{id}"):
                if resolution_level != keep_resolution_level:
                    continue
                filename = os.listdir(f"{path}/{class_label}/{id}/{resolution_level}")[0]
                with open(f"../Datasets/Meshes/SHREC17/{class_label}/{id}/{resolution_level}/{filename.replace('concRing', 'mesh')}", "rb") as file:
                    mesh = pkl.load(file)

                with open(f"{path}/{class_label}/{id}/{resolution_level}/{filename}", "rb") as file:
                    concRings = pkl.load(file)

                for concentric_ring in tqdm(concRings):
                    if not concentric_ring.firstValidRings(len(concentric_ring.rings)):
                        continue
                    if np.linalg.norm(concentric_ring[-1][0] - concentric_ring[-1][int(len(concentric_ring[0]) / 2)]) < radius * 1.60:
                        continue
                    try:
                        graphs = np.append(graphs, SpiderPatch(concentric_ring, mesh, concentric_ring.seed_point, seed_point_idx=False))
                    except dgl.DGLError:
                        continue
                    labels = np.append(labels, label)
                    seed_point_indices = np.append(seed_point_indices, concentric_ring.seed_point)
                    mesh_id = np.append(mesh_id, int(re.sub(r"\D", "", filename)))
    if only_homogeneous:
        for id, el in enumerate([3, 5]):  # [1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14] FIRST HOMOGENEUS    [3, 5, 7, 8, 11, 12, 13, 14]  SECOND HOMOGENEUS
            labels[labels == el] = id
        dataset = RetrievalDataset(dataset_name=dataset_name + "_HOMOGENEOUS3", spiral_spider_patches=graphs, labels=labels, mesh_id=mesh_id)
    else:
        dataset = RetrievalDataset(dataset_name=dataset_name, spiral_spider_patches=graphs, labels=labels, mesh_id=mesh_id)
    dataset.save_to(f"../Retrieval/Datasets")


if __name__ == "__main__":
    # Change the scripts working directory to the script's own directory
    os.chdir(os.path.dirname(sys.argv[0]))

    # generateSHREC17SpiderDataset("../Datasets/ConcentricRings/SHREC17_RR10_R4_P6_CSIRSv2Spiral", "SHREC17_RR10R4P6_CSIRSv2Spiral", only_homogeneous=True, keep_resolution_level = "resolution_level_3")
    # for labels_filename in os.listdir("../Retrieval/Labels")[4:]:
    #     with open(f"../Retrieval/Datasets/{labels_filename.replace('.pkl','BC_R1R4P6_Spiral.pkl')}", "rb") as labels_file:
    #         print(labels_file)
    #         dataset = pkl.load(labels_file)
    #     with open(f"../Retrieval/Mesh/{labels_filename.replace('.pkl','BC.pkl')}", "rb") as labels_file:
    #         print(labels_file)
    #         mesh = pkl.load(labels_file)
    #     for label in np.unique(dataset.labels):
    #         indices = np.where(dataset.labels == label)[0]
    #         lista = []
    #         for idx in indices:
    #             lista.append(dataset.graphs[idx].ndata["vertices"][0])
    #         mesh.drawWithPointCloud(lista)

    # mesh_name = "surface7"
    # generate_mesh(path = f"U:\AssegnoDiRicerca\MeshDataset\SHREC18\shrec_retrieval_tortorici/{mesh_name}.off", name = f"{mesh_name}BC", curvature="BC")
    # generate_labels(mesh_name=f"{mesh_name}BC", labels_path=f"U:\AssegnoDiRicerca\MeshDataset\SHREC18\shrec_retrieval_tortorici\Labels/{mesh_name}.mat")
    # generate_mesh(path=f"U:\AssegnoDiRicerca\MeshDataset\SHREC18\shrec_retrieval_tortorici/{mesh_name}.off", name=f"{mesh_name}LC", curvature="LC")

    # radius = 10
    # rings = 4
    # points = 6
    # mesh = "egyptFaceDenseBC"
    # labels = "egyptFaceDense"
    # print(f"Generating: {mesh}_RR{radius}R{rings}P{points}_Spiral")
    # dataset = RetrievalDataset(dataset_name=f"{mesh}_RR{radius}R{rings}P{points}_Spiral", mesh_id=mesh)
    # dataset.generate(f"../Retrieval/Mesh/{mesh}.pkl", f"../Retrieval/Labels/{labels}.pkl", radius, rings, points, relative_radius=True)
    #
    # # PARALLEL DATASET GENERATION
    # mesh_list = ["surface4BC", "surface5BC", "surface6BC", "surface7BC"]
    # labels_list = ["surface4", "surface5", "surface6", "surface7"]
    #
    # thread_num = 4
    # pool = multiprocessing.Pool(processes=thread_num)
    # pool.starmap(parallel_func, [(mesh_list[i], labels_list[i], True) for i in range(thread_num)])

    # dataset_name_list = ["surface4BC_R1R4P6_Spiral", "surface5BC_R1R4P6_Spiral", "surface6BC_R1R4P6_Spiral", "surface7BC_R1R4P6_Spiral"]
    # dataset = fuse_datasets(dataset_name_list, "MergedSurfaceBC_R1R4P6_Spiral")
    # dataset.save_to("../Retrieval/Datasets")

    dataset_name = "MergedSurfaceBC_R1R4P6_Spiral"
    features = np.array(["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"])
    features_to_keep = [0, 1, 2, 3, 4]
    radius_to_keep = [0, 1, 2, 3, 4]
    #
    # # TRAIN DATASET
    config = re.search('_(.*?)_', dataset_name).group(1)
    rings = int(re.search('R(\d)P', dataset_name).group(1))
    points = int(re.search('P(.*\d)_', dataset_name).group(1))
    dataset = RetrievalDataset(dataset_name=dataset_name)
    dataset.load_from(f"../Retrieval/Datasets", dataset_name)
    dataset.selectGraphsByNumNodes((rings * points) + 1)
    dataset.keepCurvaturesResolution(radius_to_keep)
    dataset.aggregateNodeFeatures(features[features_to_keep])
    dataset.aggregateEdgeFeatures()
    dataset.removeNonAggregatedFeatures(to_keep=["weight"])
    # # dataset._name = dataset_name + "_READY"
    # # dataset.save_to("../Retrieval/Datasets")
    # #
    trainMLP(dataset, 0.005)
    # #
    # # train_mask, test_mask = dataset.getTrainTestMask(600, percentage=False)
    # # dataset.normalize(train_mask)
    # # print(dataset.graphs[0].node_attr_schemes())
    # # trainSVC(dataset, train_mask, test_mask)
    # # # trainRandomForest(dataset, train_mask, test_mask)
    # # # trainExtraRandomForest(dataset, train_mask, test_mask)
