
import warnings
from time import time


from PlotUtils import plot_confusion_matrix, save_confusion_matrix, plot_training_statistics, plot_embeddings, plot_embeddings_space, plot_model_parameters_comparison, plot_grad_flow, print_weights, print_weights_difference
from sklearn import metrics
from SpiderPatch.Networks import *

from tqdm import tqdm, trange
from dgl.dataloading import GraphDataLoader

from SpiderDatasets.MeshGraphDataset import MeshGraphDatasetForNNTraining, MeshGraphDataset


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##### GRAFICO COMPARAZIONE PESI DELLE RETI  #######
    # paths = ["TrainedModels/Train_SHREC17_R10_RI6_P6_PATCH10_level_0/MeshNetworkBestAcc/network.pt",
    #          "TrainedModels/Train_SHREC17_R10_RI6_P6_PATCH10_level_1/MeshNetworkBestAcc/network.pt",
    #          "TrainedModels/Train_SHREC17_R10_RI6_P6_PATCH10_level_2/MeshNetworkBestAcc/network.pt",
    #          "TrainedModels/Train_SHREC17_R10_RI6_P6_PATCH10_level_3/MeshNetworkBestAcc/network.pt"]
    #
    # plot_model_parameters_comparison(paths)

    ######## UN MODELLO 4 DATASET  ########
    # model = MeshNetwork()
    # model.load("TrainedModels/Train_SHREC17_R10_RI6_P6_PATCH30_level_2/MeshNetworkBestAcc/network.pt")
    # model.to(device)
    #
    # for level in ["level_0", "level_1", "level_2", "level_3"]:  # "all",
    #     dataset = MeshGraphDatasetForNNTraining()
    #     dataset.load(f"Datasets/MeshGraphsForTraining/SHREC17_R10_RI6_P6_PATCH30/SHREC17_R10_RI6_P6_PATCH30_{level}_Normalized", f"SHREC17_R10_RI6_P6_PATCH30_{level}")
    #     dataset.to(device)
    #     acc, loss, cm = testMeshNetwork(model, dataset.validation_dataset, device)
    #     print(f"Acc: {acc}, Loss: {loss}")
    #     plot_confusion_matrix(cm)



    ###### TRAIN DI UN MODELLO SU 4 RISOLUZIONI DI UN DATASET  #########
    for level in ["level_0"]:  #"all",,"level_1", "level_2", "level_3"
        dataset = MeshGraphDatasetForNNTraining()
        dataset.load(f"Datasets/Prova/SHREC17_R7_RI4_P8_PATCH10_SAMPLE5/SHREC17_R7_RI4_P8_PATCH10_SAMPLE5_{level}_Normalized", f"SHREC17_R7_RI4_P8_PATCH10_SAMPLE5_{level}")
        dataset.train_dataset.graphs[0].draw()
        dataset.to(device)
        model = MeshNetwork().to(device)
        trainMeshNetwork(model, dataset, device, 50, f"SHREC17_R7_RI4_P8_PATCH10_SAMPLE5_{level}")

    # for level in ["level_0", "level_1", "level_2", "level_3"]:  #"all",
    #     dataset = MeshGraphDatasetForNNTraining()
    #     dataset.load(f"Datasets/MeshGraphsForTraining/SHREC17_R10_RI6_P6_PATCH10/SHREC17_R10_RI6_P6_PATCH10_{level}_Normalized", f"SHREC17_R10_RI6_P6_PATCH10_{level}")
    #     dataset.to(device)
    #     model = MeshNetwork().to(device)
    #     trainMeshNetwork(model, dataset, device, 50, f"SHREC17_R10_RI6_P6_PATCH10_{level}")
    #
    # for level in [ "level_0", "level_1", "level_2", "level_3"]:  #"all",
    #     dataset = MeshGraphDatasetForNNTraining()
    #     dataset.load(f"Datasets/MeshGraphsForTraining/SHREC17_R5_RI4_P6_PATCH10/SHREC17_R5_RI4_P6_PATCH10_{level}_Normalized", f"SHREC17_R5_RI4_P6_PATCH10_{level}")
    #     dataset.to(device)
    #     model = MeshNetwork().to(device)
    #     trainMeshNetwork(model, dataset, device, 50, f"SHREC17_R5_RI4_P6_PATCH10_{level}")


def trainMeshNetwork(model, dataset, device, epochs=1000, train_name=""):
    model.train()
    best_acc = 0
    best_loss = 1000
    dataloader = GraphDataLoader(dataset.train_dataset, batch_size=1, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    start = time()
    train_losses = []
    val_accuracies = []
    val_losses = []
    for epoch in range(epochs):
        sampler = 0
        train_losses.append(0)
        for graph, label in tqdm(dataloader, position=0, leave=False, desc=f"Epoch {epoch + 1}: ", colour="white", ncols=80):
            opt.zero_grad()
            pred, embedding = model(graph, dataset.train_dataset.graphs[sampler].patches, device)
            loss_running = F.cross_entropy(pred, label)
            train_losses[-1] += loss_running.item()
            loss_running.backward()
            opt.step()
            sampler += 1

        train_losses[-1] /= len(dataset.train_dataset.graphs)
        print(f"Epoch {epoch + 1}/{epochs} ({int(time() - start)}s):"
              f" Epoch Loss={train_losses[-1]:.3f}")

        acc, loss, cm = testMeshNetwork(model=model, dataset=dataset.validation_dataset, device=device)
        val_accuracies.append(acc)
        val_losses.append(loss)
        print(f"Validation Test\n"
              f"Acc. : {acc}\n"
              f"Loss.: {loss}")

        if acc > best_acc:
            best_acc = acc
            model.save(f"TrainedModels/Train_{train_name}/MeshNetworkBestAcc")
            save_confusion_matrix(cm, f"TrainedModels/Train_{train_name}/MeshNetworkBestAcc/ConfusionMatrixEpoch{epoch+1}.png")

        if loss < best_loss:
            best_loss = loss
            model.save(f"TrainedModels/Train_{train_name}/MeshNetworkBestLoss")
            save_confusion_matrix(cm, f"TrainedModels/Train_{train_name}/MeshNetworkBestLoss/ConfusionMatrixEpoch{epoch+1}.png")
        plot_training_statistics(f"TrainedModels/Train_{train_name}/Train_{train_name}_statistics.png", title=f"Best Acc: {best_acc}, Loss: {best_loss}", epochs=range(epoch + 1), losses=train_losses, val_accuracies=val_accuracies, val_losses=val_losses)


def testMeshNetwork(model, dataset, device):
    model.eval()
    correct_prediction_number = torch.empty(0, device=device)
    loss_predicted = torch.empty(size=(0, dataset.numClasses()), device=device)
    dataloader = GraphDataLoader(dataset, batch_size=1, drop_last=False)
    sampler = 0
    for graph, label in dataloader:
        pred, _ = model(graph, dataset.graphs[sampler].patches, device)
        sampler += 1
        correct_prediction_number = torch.hstack((correct_prediction_number, pred.argmax(dim=1)))  # Take the highest value in the predicted classes vector
        loss_predicted = torch.vstack((loss_predicted, pred))
    model.train()
    return compute_scores(dataset, correct_prediction_number, loss_predicted)

def compute_scores(dataset, pred_labels, loss_predicted):
    # Computation of accuracy metrics
    accuracy = pred_labels.eq(dataset.labels).sum().item() / len(dataset.graphs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        confusion_matrix = metrics.confusion_matrix(dataset.labels.tolist(), pred_labels.tolist(), normalize="true")
    final_labels = torch.tensor(dataset.labels.tolist(), dtype=torch.int64, device=loss_predicted.device)
    loss = F.cross_entropy(loss_predicted, final_labels)
    return accuracy, loss.item(), confusion_matrix




if __name__ == "__main__":
    main()
