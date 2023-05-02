import os
import re
import sys

import numpy as np
import torch
from dgl.dataloading import GraphDataLoader
from torch import nn, cdist, cosine_similarity
from tqdm import tqdm

from CSIRS.CSIRS import CSIRSv2Spiral
from Networks.CONVSpiderPatchModules import SPReader
from Networks.GATMeshGraphModules import GATMGEmbedder
from Networks.GATNetworks import TestNetwork
from Networks.GATSpiderPatchModules import GATSPEmbedder
from Networks.MeshGraphReader import MeshGraphReader
from SpiderDatasets.MeshGraphDataset import MeshGraphDataset
from SpiderDatasets.RetrievalDataset import generateMesh, generateLabels, RetrievalDataset


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_name = "SHREC17_R10_R6_P8_CSIRSv2Spiral_MGRAPHS1_SPIDER25_CONN10_RES3"

    features = np.array(["gauss_curvature", "mean_curvature", "curvedness", "K2", "local_depth"])
    features_to_keep = [0, 1, 2, 3, 4]
    radius_to_keep = [0, 1, 2, 3, 4]

    dataset = MeshGraphDataset(dataset_name=dataset_name)
    dataset.load_from(f"../Datasets/MeshGraphs", dataset_name)

    print(dataset.graphs[0].patches[0].node_attr_schemes())
    dataset.keepCurvaturesResolution(radius_to_keep)
    dataset.aggregateSpiderPatchesNodeFeatures(features[features_to_keep], "aggregated_feats")
    dataset.aggregateSpiderPatchesNodeFeatures(["weights", "rings", "points"], "aggregated_weights")
    dataset.aggregateSpiderPatchEdgeFeatures()
    dataset.removeNonAggregatedFeatures()

    feat_in_channels = dataset.graphs[0].patches[0].ndata["aggregated_feats"].shape[1]
    weights_in_channels = dataset.graphs[0].patches[0].ndata["aggregated_weights"].shape[1]

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

    feat_in_channels = dataset.graphs[0].patches[0].ndata["aggregated_feats"].shape[1]
    weights_in_channels = dataset.graphs[0].patches[0].ndata["aggregated_weights"].shape[1]

    network_parameters = {}

    ####  SPIDER PATCH PARAMETERS  ####
    network_parameters["SP"] = {}
    network_parameters["SP"]["module"] = GATSPEmbedder  # [ CONVSPEmbedder, GATSPEmbedder , GATWeightedSP, SPReader]
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

    model = TestNetwork(feat_in_channels, weights_in_channels, 15, network_parameters=network_parameters, use_SP_triplet=False)
    model.load("../TrainingResults/Experiments/14032023-103155/MeshNetworkBestAcc/network.pt")

    spider_patch_reader = model.patch_embedder
    # spider_patch_reader =  SPReader(feat_in_channels)

    mesh_graph_reader = MeshGraphReader(spider_patch_reader)
    mesh_graph_reader.to(device)

    BATCH_SIZE = 5
    dataloader = GraphDataLoader(dataset[[i for i in range(len(dataset))]], batch_size=BATCH_SIZE, drop_last=False, shuffle=False)

    mesh_graphs_embeddings = torch.empty((0, mesh_graph_reader.spider_patch_embedder.embed_dim), device=device)

    for sampler, (batched_MG, labels) in enumerate(tqdm(dataloader, position=0, leave=False, desc=f"", colour="white", ncols=80)):
        # To GPU memory
        batched_MG = batched_MG.to(device)
        labels = labels.to(device)

        # Prepare the batched SpiderPatches
        SP_sampler = np.arange((sampler * BATCH_SIZE), (sampler * BATCH_SIZE) + len(labels))
        spider_patches = [sp.to(device) for idx in SP_sampler for sp in dataset.graphs[idx].patches]

        mesh_graphs_embeddings = torch.vstack((mesh_graphs_embeddings, mesh_graph_reader(batched_MG, spider_patches, device)))

    cosine = sim_matrix(mesh_graphs_embeddings, mesh_graphs_embeddings)
    distances = cdist(mesh_graphs_embeddings, mesh_graphs_embeddings)
    print(distances)


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


if __name__ == "__main__":
    # Change the scripts working directory to the script's own directory
    os.chdir(os.path.dirname(sys.argv[0]))
    main()
