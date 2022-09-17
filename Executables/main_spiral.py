import warnings
from time import time

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from Networks.SpiralNetworks import SpiralMeshReader
from PlotUtils import plot_confusion_matrix, save_confusion_matrix
from sklearn import metrics
import torch.nn.functional as F
from tqdm import tqdm, trange
from dgl.dataloading import GraphDataLoader

from SpiderDatasets.SpiralMeshGraphDatasetForNNTraining import SpiralMeshGraphDatasetForNNTraining


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###### TRAIN DI UN MODELLO SU 4 RISOLUZIONI DI UN DATASET  #########
    dataset = SpiralMeshGraphDatasetForNNTraining()
    dataset.load(f"../Datasets/SpiralMeshGraphsForTraining/SHREC17_R5_RI4_P12_SPIRAL20_SAMPLE20/SHREC17_R5_RI4_P12_SPIRAL20_SAMPLE20_CONN5_Normalized", f"SHREC17_R5_RI4_P12_SPIRAL20_SAMPLE20_CONN5")

    print(dataset.train_dataset.graphs[0].node_attr_schemes())
    dataset.aggregateNodeFeatures(["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"])
    dataset.removeNonAggregatedFeatures()
    dataset.train_dataset.graphs[0].draw()
    print(dataset.train_dataset.graphs[0].node_attr_schemes())
    dataset.to(device)

    model = SpiralMeshReader(in_dim=dataset.train_dataset.graphs[0].ndata["aggregated_feats"].shape[1], hidden_dim=10192, out_dim=15)
    model.to(device)

    trainMeshNetwork(model, dataset, 150, f"SHREC17_R5_RI4_P12_SPIRAL20_SAMPLE20_CONN5", dataset.validation_dataset)


def trainMeshNetwork(model, dataset, epochs=1000, train_name="", test_dataset=None):
    model.train()
    best_acc = 0

    dataloader = GraphDataLoader(dataset.train_dataset, batch_size=5, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=10e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.65), int(epochs * 0.90)], gamma=0.1)
    start = time()
    train_losses = []
    val_accuracies = []
    val_losses = []
    for epoch in range(epochs):
        train_losses.append(0)
        for id, group in enumerate(optimizer.state_dict()['param_groups']):
            print(f"Learning rate of group {id}: {group['lr']}")
        for graph, label in tqdm(dataloader, position=0, leave=False, desc=f"Epoch {epoch + 1}: ", colour="white", ncols=80):
            optimizer.zero_grad()
            pred = model(graph, None)
            loss_running = F.cross_entropy(pred, label)
            train_losses[-1] += loss_running.item()
            loss_running.backward()
            optimizer.step()
        train_losses[-1] /= len(dataset.train_dataset.graphs)
        print(f"Epoch {epoch + 1}/{epochs} ({int(time() - start)}s):"
              f" Epoch Loss={train_losses[-1]:.3f}")

        if test_dataset is None:
            acc, loss, cm = testMeshNetwork(model=model, dataset=dataset.validation_dataset)
        else:
            acc, loss, cm = testMeshNetwork(model=model, dataset=test_dataset)
        val_accuracies.append(acc)
        val_losses.append(loss)
        print(f"Validation Test\n"
              f"Acc. : {acc}\n"
              f"Loss.: {loss}")

        if acc > best_acc:
            best_acc = acc
            model.save(f"../TrainedModelsSpiral/Train_{train_name}")
            save_confusion_matrix(cm, f"../TrainedModelsSpiral/Train_{train_name}/ConfusionMatrix.png")
        scheduler.step()


def testMeshNetwork(model, dataset):
    model.eval()
    correct_prediction_number = np.empty(0)
    loss_predicted = np.empty((0, dataset.numClasses()))
    dataloader = GraphDataLoader(dataset, batch_size=20, drop_last=False)
    for graph, label in dataloader:
        pred = model(graph, None)
        pred = pred.cpu()
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


if __name__ == "__main__":
    main()
