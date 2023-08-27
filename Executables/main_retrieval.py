import os
import pickle as pkl
import sys

import numpy as np
import torch
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm

from Networks.GATMeshGraphModules import GATMGEmbedder
from Networks.GATNetworks import TestNetwork
from Networks.GATSpiderPatchModules import GATSPEmbedder
from SpiderDatasets.MeshGraphDataset import MeshGraphDataset
from SpiderDatasets.RetrievalDataset import generateMesh, generateLabels


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_name = "SHREC17_R15_R4_P6_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5_RES3"
    # dataset_name = "SHREC20_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN5"
    dataset = MeshGraphDataset(dataset_name=dataset_name)
    dataset.load_from(f"../Datasets/MeshGraphs", dataset_name)

    features = np.array(["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"])
    features_to_keep = [0, 1, 2, 3, 4]
    radius_to_keep = [0, 1, 2, 3, 4]

    dataset.keepCurvaturesResolution(radius_to_keep)
    dataset.aggregateSpiderPatchesNodeFeatures(features[features_to_keep], "aggregated_feats")
    dataset.aggregateSpiderPatchesNodeFeatures(["weights", "rings", "points"], "aggregated_weights")
    dataset.aggregateSpiderPatchEdgeFeatures()
    dataset.removeNonAggregatedFeatures()

    mode = "normalization"  # ["standardization", "normalization", "robust", "quantile"]
    elim_mode = None  # ["standard", "quantile", None]

    node_normalizers = dataset.normalizeV2(np.arange(0, len(dataset)), mode, elim_mode)

    with open("../Retrieval/Normalizers/normalizers.pkl", "wb") as norm_file:
        pkl.dump(node_normalizers, norm_file)

    # with open("../Retrieval/Normalizers/normalizers.pkl", "rb") as norm_file:
    #     node_normalizers = pkl.load(norm_file)

    dataset_name = "SHREC20_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS1_SPIDER25_CONN5"
    dataset = MeshGraphDataset(dataset_name=dataset_name)
    dataset.load_from(f"../Datasets/MeshGraphs", dataset_name)

    features = np.array(["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"])
    features_to_keep = [0, 1, 2, 3, 4]
    radius_to_keep = [0, 1, 2, 3, 4]

    dataset.keepCurvaturesResolution(radius_to_keep)
    dataset.aggregateSpiderPatchesNodeFeatures(features[features_to_keep], "aggregated_feats")
    dataset.aggregateSpiderPatchesNodeFeatures(["weights", "rings", "points"], "aggregated_weights")
    dataset.aggregateSpiderPatchEdgeFeatures()
    dataset.removeNonAggregatedFeatures()

    for mesh_graph in tqdm(dataset.graphs, position=0, leave=True, desc=f"Normalizing: ", colour="white", ncols=80):
        for spider_patch in mesh_graph.patches:
            for feature in dataset.getSpiderPatchNodeFeatsNames():
                if spider_patch.node_attr_schemes()[feature].shape == ():
                    spider_patch.ndata[feature] = torch.tensor(node_normalizers[feature].transform(spider_patch.ndata[feature].reshape((-1, 1))), dtype=torch.float32)
                else:
                    spider_patch.ndata[feature] = torch.tensor(node_normalizers[feature].transform(spider_patch.ndata[feature]), dtype=torch.float32)

    network_parameters = {}

    ####  SPIDER PATCH PARAMETERS  ####
    network_parameters["SP"] = {}
    network_parameters["SP"]["module"] = GATSPEmbedder  # [ CONVSPEmbedder, GATSPEmbedder , GATWeightedSP, SPReader]
    network_parameters["SP"]["readout_function"] = "AR"  # [ "AR" , "UR", "SR"]
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

    model = TestNetwork(25, 3, 11, network_parameters=network_parameters, use_SP_triplet=False)
    # model.load("U:\AssegnoDiRicerca\PythonProject\TrainingResults\Experiments/13052023-151149\MeshNetworkBestAcc/network.pt")
    model.load("U:\AssegnoDiRicerca\PythonProject\TrainingResults\Experiments/11042023-073700\MeshNetworkBestAcc/network.pt")
    model.to(device)

    euclidean_distance_matrix, cosine_distance_matrix = generateDistanceMatrix(model, dataset.graphs, device)
    euclidean_distance_matrix[euclidean_distance_matrix == 0] = 100000000000000
    NN = torch.argmin(euclidean_distance_matrix, dim=0)
    ok = 0
    for i, nn in enumerate(NN):
        if dataset.labels[i] == dataset.labels[nn]:
            ok += 1
    print(f"Euclidean distance: {ok}")

    euclidean_distance_matrix[euclidean_distance_matrix == 1] = -1000000
    NN = torch.argmax(euclidean_distance_matrix, dim=0)
    ok = 0
    for i, nn in enumerate(NN):
        if dataset.labels[i] == dataset.labels[nn]:
            ok += 1
    print(f"Cosine distance: {ok}")


def generateDistanceMatrix(model, mesh_graphs, device):
    BATCH_SIZE = 128

    model.eval()
    MG_embeddings = torch.zeros((len(mesh_graphs), model.mesh_embedder.embed_dim), device=device)
    dataloader = GraphDataLoader(mesh_graphs, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)

    with torch.no_grad():
        for sampler, batched_MG in enumerate(dataloader):
            # To GPU memory
            batched_MG = batched_MG.to(device)

            # Prepare the batched SpiderPatches
            SP_sampler = np.arange((sampler * BATCH_SIZE), (sampler * BATCH_SIZE) + batched_MG.batch_size)
            spider_patches = [sp.to(device) for idx in SP_sampler for sp in mesh_graphs[idx].patches]

            _, MG_embeddings[(sampler * BATCH_SIZE): (sampler * BATCH_SIZE) + batched_MG.batch_size] = model(batched_MG, spider_patches, device)

    euclidean_dist_matrix = torch.cdist(MG_embeddings, MG_embeddings)

    norm = MG_embeddings / MG_embeddings.norm(dim=1)[:, None]
    cosine_dist_matrix = torch.mm(norm, norm.transpose(0, 1))

    return euclidean_dist_matrix, cosine_dist_matrix


def generateRetrievalMesh(mesh_name):
    generateMesh(f"../../MeshDataset/SHREC18/shrec_retrieval_tortorici/{mesh_name}.off", f"{mesh_name}")
    generateLabels(f"{mesh_name}", f"../../MeshDataset/SHREC18/shrec_retrieval_tortorici/Labels/{mesh_name}.mat")


if __name__ == "__main__":
    # Change the scripts working directory to the script's own directory
    os.chdir(os.path.dirname(sys.argv[0]))
    main()
