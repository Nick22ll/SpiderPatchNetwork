import torch

from PlotUtils import plot_model_parameters_comparison, plot_embeddings_space, plot_embeddings
from SpiderDatasets.MeshGraphForTrainingDataset import MeshGraphDatasetForNNTraining
from SpiderPatch.Networks import MeshNetwork


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    ##### GRAFICO COMPARAZIONE PESI DELLE RETI  #######
    # paths = ["TrainedModels/Train_SHREC17_R10_RI6_P6_PATCH30_level_0/MeshNetworkBestAcc/network.pt",
    #          "TrainedModels/Train_SHREC17_R10_RI6_P6_PATCH30_level_1/MeshNetworkBestAcc/network.pt",
    #          "TrainedModels/Train_SHREC17_R10_RI6_P6_PATCH30_level_2/MeshNetworkBestAcc/network.pt",
    #          "TrainedModels/Train_SHREC17_R10_RI6_P6_PATCH30_level_3/MeshNetworkBestAcc/network.pt"]
    #
    # plot_model_parameters_comparison(paths)

    ######  PLOT CROSS-EMBEDDINGS SPACE  ######
    model = MeshNetwork()
    model.load("TrainedModels/Train_SHREC17_R10_RI6_P6_PATCH10_level_0/MeshNetworkBestAcc/network.pt")
    model.to(device)
    dataset_paths = ["Datasets/MeshGraphsForTraining/SHREC17_R10_RI6_P6_PATCH30/SHREC17_R10_RI6_P6_PATCH30_level_0_Normalized",
                     "Datasets/MeshGraphsForTraining/SHREC17_R10_RI6_P6_PATCH30/SHREC17_R10_RI6_P6_PATCH30_level_2_Normalized",
                     ]
    #                  "Datasets/MeshGraphsForTraining/SHREC17_R10_RI6_P6_PATCH30/SHREC17_R10_RI6_P6_PATCH30_level_2_Normalized",
    #                  "Datasets/MeshGraphsForTraining/SHREC17_R10_RI6_P6_PATCH30/SHREC17_R10_RI6_P6_PATCH30_level_3_Normalized"
    plot_embeddings_space(model, dataset_paths, device, "CROSS_EMBEDDINGS_PATCH30.png")

    ##### PLOT EMBEDDINGS SPACE #######
    dataset = MeshGraphDatasetForNNTraining()
    dataset.load(f"Datasets/MeshGraphsForTraining/SHREC17_R10_RI6_P6_PATCH30/SHREC17_R10_RI6_P6_PATCH30_level_0_Normalized", f"SHREC17_R10_RI6_P6_PATCH30_level_0")
    dataset.to(device)
    plot_embeddings(model, dataset.train_dataset, device, "EMBEDDINGS_PATCH30_level_0.png")

if __name__ == "__main__":
    main()
