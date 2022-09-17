import random
import scipy.io
from CSIRS.CSIRS import CSIRSv2, CSIRSv2Arbitrary, CSIRS, CSIRSv2Spiral
from operator import itemgetter

from Mesh.Mesh import Mesh
from SpiderDatasets.SpiderPatchDataset import SpiderPatchDataset
from Networks.MLP import MLP
from SpiderPatch.SpiderPatch import SpiderPatchLRF, SpiderPatch
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
from Networks.CONVNetworks import *
from tqdm import tqdm, trange
from dgl.dataloading import GraphDataLoader


def generate_mesh(name):
    mesh = Mesh()
    mesh.loadFromMeshFile("U:\AssegnoDiRicerca\MeshDataset\SHREC18\shrec_retrieval_tortorici/egyptFaceDense.off")
    mesh.computeCurvaturesTemp()
    mesh.save(f"U:\AssegnoDiRicerca\PythonProject\Executables\Mesh\egyptFaceDense{name}.pkl")


def generate_labels(mesh_name):
    mat = scipy.io.loadmat(f'U:\AssegnoDiRicerca\MeshDataset\SHREC18\shrec_retrieval_tortorici\Labels/{mesh_name}.mat')
    face_labels = mat["label"].flatten()
    with open(f"U:\AssegnoDiRicerca\PythonProject\Executables\Mesh\{mesh_name}.pkl", "rb") as mesh_file:
        mesh = pkl.load(mesh_file)
    vertex_labels = []
    for face_list in mesh.vertex_faces:
        vertex_labels.append(np.argmax(np.bincount(face_labels[face_list])))
    for id, elem in enumerate(np.unique(vertex_labels)):
        vertex_labels = [id if x == elem else x for x in vertex_labels]
    with open(f"Labels/{mesh_name}.pkl", "wb") as label_file:
        pkl.dump(vertex_labels, label_file)


def generate_dataset(dataset_name, mesh_name, labels_name, radius, rings, points):
    # Select some points and calculate the SpiderPatches for the ground truth for retrieval
    # 0 - face skin    1 - hairs   2 - eyebrow  3 - beard
    warnings.filterwarnings("ignore")
    SAMPLE_NUMBER = 200
    with open(f"U:\AssegnoDiRicerca\PythonProject\Executables\Mesh/{mesh_name}.pkl", "rb") as mesh_file:
        mesh = pkl.load(mesh_file)

    with open(f"U:\AssegnoDiRicerca\PythonProject\Executables\Labels/{labels_name}.pkl", "rb") as labels_file:
        labels = pkl.load(labels_file)

    random.seed(222)

    sample_seed_points = {}
    reference_spider_patches = {}
    reference_labels = {}
    reference_seed_points = {}
    for label in np.unique(labels):
        sample_seed_points[label] = []
        reference_spider_patches[label] = []
        reference_labels[label] = []
        reference_seed_points[label] = []
        tmp_label_indices = np.where(labels == label)[0]
        # Remove boundary vertices
        boundary_vertices = mesh.getBoundaryVertices(int(np.ceil(1.5 * radius)))
        label_indices = [el for el in tmp_label_indices if el not in boundary_vertices]
        seed_points = np.unique(random.sample(label_indices, 1000))
        while len(seed_points) < SAMPLE_NUMBER:
            seed_points = np.unique(random.sample(label_indices, 1000))
        seed_points = list(seed_points)
        random.shuffle(seed_points)
        for seed_point in seed_points:
            sample_seed_points[label].append({"seed_point": seed_point, "label": label})

    for label in tqdm(np.unique(labels)):
        for seed_point in sample_seed_points[label]:
            concentric_ring, _ = CSIRSv2Spiral(mesh, seed_point["seed_point"], radius, rings, points)
            if not concentric_ring.first_valid_rings(rings):
                continue
            try:
                spider_patch = SpiderPatch(concentric_ring, mesh, seed_point["seed_point"])
            except dgl.DGLError:
                continue
            reference_spider_patches[label].append(spider_patch)
            reference_labels[label].append(seed_point["label"])
            reference_seed_points[label].append(seed_point["seed_point"])
            if len(reference_spider_patches[label]) >= SAMPLE_NUMBER:
                break

    graphs = []
    labels = []
    seed_points = []
    for key in reference_seed_points.keys():
        graphs += reference_spider_patches[key]
        labels += reference_labels[key]
        seed_points += reference_seed_points[key]

    dataset = SpiderPatchDataset(dataset_name=dataset_name, graphs=graphs, labels=labels)
    dataset.seed_point_indices = seed_points
    dataset.save_to(f"Datasets")


def generate_test_dataset(name, mesh_name, labels_name, radius, rings, points):
    warnings.filterwarnings("ignore")
    train_dataset = SpiderPatchDataset(dataset_name=name)
    train_dataset.load_from(f"Datasets", name)

    with open(f"U:\AssegnoDiRicerca\PythonProject\Executables\Mesh/{mesh_name}.pkl", "rb") as mesh_file:
        mesh = pkl.load(mesh_file)

    with open(f"U:\AssegnoDiRicerca\PythonProject\Executables\Labels/{labels_name}.pkl", "rb") as labels_file:
        vertex_labels = pkl.load(labels_file)

    boundary_vertices = mesh.getBoundaryVertices(neighbors_level=int(np.ceil(1.5 * radius)))
    random.seed(222)
    seed_point_sequence = [i for i in range(len(mesh.vertices))]
    random.shuffle(seed_point_sequence)
    # seed_point_sequence = list(np.unique(random.sample(range(len(mesh.vertices) - 1), 25000)))
    # random.shuffle(seed_point_sequence)
    spider_patches = []
    seed_points = []
    labels = []
    for seed_point in tqdm(seed_point_sequence):
        if seed_point in boundary_vertices or seed_point in train_dataset.seed_point_indices:
            continue
        concentric_ring, _ = CSIRSv2Spiral(mesh, seed_point, radius, rings, points, radius)
        if not concentric_ring.first_valid_rings(1):
            continue
        try:
            spider_patches.append(SpiderPatch(concentric_ring, mesh, seed_point))
        except dgl.DGLError:
            continue
        labels.append(vertex_labels[seed_point])
        seed_points.append(seed_point)

    dataset = SpiderPatchDataset(dataset_name=name + "_test", graphs=spider_patches, labels=labels)
    dataset.seed_point_indices = seed_points
    dataset.save_to(f"Datasets")


def set_node_weights(dataset):
    for patch in dataset.graphs:
        weights = []
        for vertex in patch.ndata["vertices"]:
            weights.append(1 - np.linalg.norm(vertex - patch.seed_point))
        patch.ndata["weight"] = torch.tensor(np.array(weights).reshape((-1, 1)), dtype=torch.float32)
    dataset.save_to(f"Datasets")


def train(dataset_name, model, device, features, radius_to_keep):
    dataset = SpiderPatchDataset(dataset_name=dataset_name)
    dataset.load_from(f"Datasets", dataset_name)

    # mesh.draw_with_patches(dataset.graphs)
    # mesh.vertex_curvatures[5], mesh.face_curvatures[5] = getCurvatures(mesh, 1)
    # mesh.drawWithK2(5)
    # mesh.drawWithLD(5)
    # mesh.drawWithCurvedness(5)

    normalizers = dataset.normalize()
    with open("Normalizers/normalizers.pkl", "wb") as norm_file:
        pkl.dump(normalizers, norm_file)

    print(dataset.graphs[0].node_attr_schemes())
    dataset.keepCurvaturesResolution(radius_to_keep)
    dataset.aggregateNodeFeatures(features)
    dataset.aggregateEdgeFeatures()
    dataset.removeNonAggregatedFeatures(to_keep=["weight"])
    print(dataset.graphs[0].node_attr_schemes())
    dataset.to(device)

    model.train()
    epochs = 1000
    dataloader = GraphDataLoader(dataset, batch_size=5, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=10e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.85), int(epochs * 0.95)], gamma=0.1)
    start = time()
    losses = []
    best_epoch = -1
    for epoch in range(epochs):
        train_losses = 0
        for id, group in enumerate(optimizer.state_dict()['param_groups']):
            print(f"Learning rate of group {id}: {group['lr']}")
        for graph, label in tqdm(dataloader, position=0, leave=False, desc=f"Epoch {epoch + 1}: ", colour="white", ncols=80):
            optimizer.zero_grad()
            pred = model(graph, graph.edata["weights"])  # , graph.ndata["aggregated_feats"]
            loss_running = F.cross_entropy(pred, label)
            train_losses += loss_running.item()
            loss_running.backward()
            optimizer.step()
        losses.append(train_losses / len(dataset.graphs))
        print(f"Epoch {epoch + 1}/{epochs} ({int(time() - start)}s):"
              f" Epoch Loss={train_losses / len(dataset.graphs):.3f}")
        best_epoch = np.argmin(losses)
        if best_epoch == epoch:
            model.save("TrainedModelsv1/model.pt")
        scheduler.step()
    print("Best epoch:", best_epoch)


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

    label_colors = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1], 3: [0, 0, 0]}
    correct_prediction_number = np.empty(0)
    loss_predicted = np.empty((0, dataset.numClasses()))
    dataloader = GraphDataLoader(dataset, batch_size=100, drop_last=False)
    for graph, label in tqdm(dataloader):
        pred = model(graph, graph.edata["weights"])  # , graph.ndata["aggregated_feats"]
        pred = pred.cpu()
        correct_prediction_number = np.hstack((correct_prediction_number, pred.argmax(dim=1)))  # Take the highest value in the predicted classes vector
        loss_predicted = np.vstack((loss_predicted, pred.detach().numpy()))
    # point_cloud = np.empty((0, 3))
    # colors = np.empty((0, 3))
    # for id, graph in enumerate(dataset.graphs):
    #     point_cloud = np.vstack((point_cloud, graph.seed_point))
    #     colors = np.vstack((colors, label_colors[correct_prediction_number[id]]))

    return compute_scores(dataset, correct_prediction_number, loss_predicted)


def trainRetrieve(model, train_dataset, test_dataset, device, epochs=500, train_name=""):
    os.makedirs(f"TrainedModels/{train_name}", exist_ok=True)
    model.train()
    best_acc = 0
    best_loss = 1000
    dataloader = GraphDataLoader(train_dataset, batch_size=5, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=10e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.75), int(epochs * 0.90)], gamma=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, threshold=0.01)
    start = time()
    train_losses, val_epochs, val_accuracies, val_losses = [], [], [], []
    for epoch in range(epochs):
        train_losses.append(0)
        for graph, label in tqdm(dataloader, position=0, leave=False, desc=f"Epoch {epoch + 1}: ", colour="white", ncols=80):
            optimizer.zero_grad()
            pred = model(graph, graph.edata["weights"])
            loss_running = F.cross_entropy(pred, label)
            train_losses[-1] += loss_running.item()
            loss_running.backward()
            optimizer.step()
        train_losses[-1] /= len(train_dataset.graphs)
        print(f"Epoch {epoch + 1}/{epochs} ({int(time() - start)}s):"
              f" Epoch Loss={train_losses[-1]:.3f}")

        if train_losses[-1] < 0.001 or epoch == epochs - 1:
            acc, loss, cm = retrieve(test_dataset, model)
            val_accuracies.append(acc)
            val_losses.append(loss)
            val_epochs.append(epoch)
            print(f"Validation Test\n"
                  f"Acc. : {acc}\n"
                  f"Loss.: {loss}")

            if acc > best_acc:
                best_acc = acc
                os.makedirs(f"TrainedModels/{train_name}/MeshNetworkBestAcc", exist_ok=True)
                model.save(f"TrainedModels/{train_name}/MeshNetworkBestAcc/MLP.pt")
                save_confusion_matrix(cm, f"TrainedModels/{train_name}/MeshNetworkBestAcc/ConfusionMatrixEpoch.png")

            if loss < best_loss:
                best_loss = loss
                os.makedirs(f"TrainedModels/{train_name}/MeshNetworkBestLoss", exist_ok=True)
                model.save(f"TrainedModels/{train_name}/MeshNetworkBestLoss/MLP.pt")
                save_confusion_matrix(cm, f"TrainedModels/{train_name}/MeshNetworkBestLoss/ConfusionMatrix.png")

        # scheduler.step(loss)
        scheduler.step()
        plot_training_statistics(f"TrainedModels/{train_name}/{train_name}_statistics.png", title=f"Best Acc: {np.trunc(best_acc * 10000) / 100}, Loss: {best_loss}", epochs=range(epoch + 1), val_epochs=val_epochs, losses=train_losses, val_accuracies=val_accuracies,
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


if __name__ == "__main__":
    # generate_dataset("R05R4P5_BC_Spiral", "egyptFaceDenseBC", "egyptFaceDense", 0.5, 4, 5)
    # generate_test_dataset("R05R4P5_BC_Spiral", "egyptFaceDenseBC", "egyptFaceDense", 0.5, 4, 5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = "R05R4P5_BC_Spiral"

    features = np.array(["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"])
    features_to_keep = [0, 1, 2, 3, 4]
    radius_to_keep = [0, 1, 2, 3, 4]

    # TRAIN DATASET
    train_dataset = SpiderPatchDataset(dataset_name=dataset_name)
    train_dataset.load_from(f"Datasets", dataset_name)
    #
    print(train_dataset.graphs[0].node_attr_schemes())

    train_dataset.keepCurvaturesResolution(radius_to_keep)
    train_dataset.aggregateNodeFeatures(features[features_to_keep])
    train_dataset.aggregateEdgeFeatures()
    train_dataset.removeNonAggregatedFeatures(to_keep=["weight"])
    print(train_dataset.graphs[0].node_attr_schemes())

    normalizers = train_dataset.normalize()
    with open("Normalizers/normalizers.pkl", "wb") as norm_file:
        pkl.dump(normalizers, norm_file)

    train_dataset.to(device)

    # TEST DATASET
    test_dataset = SpiderPatchDataset(dataset_name=dataset_name + "_test")
    test_dataset.load_from(f"Datasets", dataset_name + "_test")

    print(test_dataset.graphs[0].node_attr_schemes())
    test_dataset.selectGraphsByNumNodes(train_dataset.graphs[0].num_nodes())
    test_dataset.keepCurvaturesResolution(radius_to_keep)
    test_dataset.aggregateNodeFeatures(features[features_to_keep])
    test_dataset.aggregateEdgeFeatures()
    test_dataset.removeNonAggregatedFeatures(to_keep=["weight"])
    print(test_dataset.graphs[0].node_attr_schemes())

    with open("Normalizers/normalizers.pkl", "rb") as norm_file:
        normalizers = pkl.load(norm_file)
    test_dataset.normalize(normalizers)

    test_dataset.to(device)

    # NETWORK MODEL
    # model = PatchReader1ConvLayer(len(radius_to_keep * len(features_to_keep)), hidden_dim=256, out_feats=4, dropout=0.2)
    # model = PatchReader2ConvLayerRetrieve(len(radius_to_keep) * len(features_to_keep), 256, 64, 64 * dataset.graphs[0].num_nodes(), 2**12, 4, 0.1)
    model = MLP(len(radius_to_keep) * len(features_to_keep) * train_dataset.graphs[0].num_nodes(), 512, 4, 0)
    model.to(device)

    trainRetrieve(model, train_dataset, test_dataset, device, 200, "Training_HIDDEN512_ALLCURV_" + dataset_name)

    model = MLP(len(radius_to_keep) * len(features_to_keep) * train_dataset.graphs[0].num_nodes(), 1024, 4, 0)
    model.to(device)

    trainRetrieve(model, train_dataset, test_dataset, device, 200, "Training_HIDDEN1024_ALLCURV_" + dataset_name)

    model = MLP(len(radius_to_keep) * len(features_to_keep) * train_dataset.graphs[0].num_nodes(), 2048, 4, 0)
    model.to(device)

    trainRetrieve(model, train_dataset, test_dataset, device, 200, "Training_HIDDEN2048_ALLCURV_" + dataset_name)
