import inspect
import os
import pickle
import re
import sys
from copy import copy, deepcopy
from time import sleep, time, perf_counter
import pickle as pkl
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import torch
from numpy import percentile
from tqdm import tqdm
import pygeodesic.geodesic as geodesic
import CSIRS.CSIRS
from Executables.main_trainMeshGCN import testNetwork
from MeshGraph.MeshGraph import MeshGraph
from Networks.GATMeshGraphModules import GATMGEmbedder
from Networks.GATNetworks import TestNetwork
from Networks.GATSpiderPatchModules import GATSPEmbedder
from Networks.Losses import CETripletMG
from PlotUtils import plotCVConfusionMatrix
from SHREC_Utils import readPermSHREC17
from SpiderDatasets.MeshGraphDataset import MeshGraphDataset
from SpiderPatch.SpiderPatch import SP_distanceV1, SP_distanceV2, SpiderPatch
import gdist
import contextlib
import sys


class DummyFile(object):
    def write(self, x): pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def main():
    os.chdir(os.path.dirname(sys.argv[0]))

    rng = np.random.default_rng(233)

    path = {
        0: "../Datasets/Meshes/SHREC17/class_0/id_0/resolution_level_0/mesh609.pkl",
        1: "../Datasets/Meshes/SHREC17/class_0/id_0/resolution_level_1/mesh452.pkl",
        2: "../Datasets/Meshes/SHREC17/class_0/id_0/resolution_level_2/mesh123.pkl",
        3: "../Datasets/Meshes/SHREC17/class_0/id_0/resolution_level_3/mesh448.pkl"
    }

    for RES in [0, 1, 2, 3]:
        with open(f"{path[RES]}", "rb") as mesh_file:
            mesh = pkl.load(mesh_file)

        for RADIUS in [6, 12]:
            for RINGS in [4, 8]:
                for POINTS in [6, 12]:
                    spiderPatches = []
                    spider_path_times = []
                    candidates = rng.choice(range(len(mesh.vertices)), 2000, replace=False)
                    seed_points = []
                    i = 0
                    boundary_vertices = mesh.getBoundaryVertices(neighbors_level=int(np.ceil(RADIUS / mesh.edge_length)))

                    while len(seed_points) <= 30:
                        if candidates[i] not in boundary_vertices:
                            seed_points.append(candidates[i])
                        i += 1

                    for i, seed_point in enumerate(seed_points):

                        with nostdout():
                            spider_path_times.append(perf_counter())
                            conc = CSIRS.CSIRS.CSIRSv2Spiral(mesh, seed_point, RADIUS, RINGS, POINTS)
                            if conc.firstValidRings(2):
                                spiderPatch = SpiderPatch(conc, mesh, conc.seed_point, seed_point_idx=False)
                                spiderPatches.append(spiderPatch)

                            spider_path_times[-1] = perf_counter() - spider_path_times[-1]

                            if not conc.firstValidRings(2):
                                del spider_path_times[-1]

                    mesh_graph_times = []
                    to_delete = np.empty(0, dtype=int)
                    for i, spiderPatch in enumerate(spiderPatches):
                        if spiderPatch.num_nodes() < (RINGS * POINTS) + 1:
                            to_delete = np.append(to_delete, i)
                    spiderPatches = np.delete(spiderPatches, to_delete)
                    for _ in range(100):
                        mesh_graph_times.append(perf_counter())
                        mesh_graph = MeshGraph(rng.choice(spiderPatches, 25), 5)
                        mesh_graph_times[-1] = perf_counter() - mesh_graph_times[-1]

                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    for name in [
                        "SHREC17_R10_R6_P8_CSIRSv2Spiral_MGRAPHS5_SPIDER20_CONN5_RES0"
                    ]:

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

                        dataset = MeshGraphDataset(dataset_name=dataset_name, graphs=np.array([deepcopy(mesh_graph) for _ in range(100)]), labels=torch.tensor(np.random.randint(0, 15, 100), dtype=torch.float32))

                        sp_radius = re.search('_R(.*?)_', dataset_name).group(1)
                        rings = int(re.search('_R(\d)_P', dataset_name).group(1))
                        points = int(re.search('_P(\d)_C', dataset_name).group(1))

                        experiment_dict["SP_radius"] = [sp_radius]
                        experiment_dict["SP_rings"] = [rings]
                        experiment_dict["SP_points"] = [points]
                        experiment_dict["dataset_MG_num"] = [int(re.search('MGRAPHS(\d*)_', dataset_name).group(1))]
                        experiment_dict["SP_per_MG"] = [int(re.search('_SPIDER(\d*)_', dataset_name).group(1))]
                        conn = re.search('_CONN(\d*)_', dataset_name)
                        if not conn:
                            conn = re.search('_CONN(\d*)', dataset_name)
                        experiment_dict["MG_connectivity"] = [int(conn.group(1) if conn else -1)]

                        res = re.search('_RES(\d*)', dataset_name)
                        experiment_dict["mesh_resolution"] = [int(res.group(1) if res else -1)]
                        experiment_dict["dataset_name"] = [dataset_name]
                        with nostdout():
                            dataset.aggregateSpiderPatchesNodeFeatures(features[features_to_keep], "aggregated_feats")
                            dataset.aggregateSpiderPatchesNodeFeatures(["weights", "rings", "points"], "aggregated_weights")
                            dataset.aggregateSpiderPatchEdgeFeatures()
                            dataset.removeNonAggregatedFeatures()
                            dataset.removeSpiderPatchByNumNodes((RINGS * POINTS) + 1)

                    class_num = 15

                    mode = "normalization"  # ["standardization", "normalization", "robust", "quantile"]
                    elim_mode = None  # ["standard", "quantile", None]

                    experiment_dict["normalization_mode"] = mode
                    experiment_dict["normalization_elim_mode"] = elim_mode if elim_mode is not None else "None"

                    print(dataset.graphs[0].patches[0].node_attr_schemes())

                    feat_in_channels = dataset.graphs[0].patches[0].ndata["aggregated_feats"].shape[1]
                    weights_in_channels = dataset.graphs[0].patches[0].ndata["aggregated_weights"].shape[1]
                    epochs = 55

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

                    for structure in ["MG", "SP"]:
                        for key, value in network_parameters[structure].items():
                            experiment_dict[f"{structure}_{key}"] = [value if not inspect.isclass(value) else value.__name__]

                    model = TestNetwork(feat_in_channels, weights_in_channels, class_num, network_parameters=network_parameters, use_SP_triplet=False)

                    model.load(f"U:\AssegnoDiRicerca\PythonProject\TrainingResults\CrossValExperiments\SHREC17_R10_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN10_RES3_CVFOLD1\MeshNetworkBestAcc/network.pt")
                    model.to(device)

                    #### TRAINING PARAMETERS  ####
                    experiment_dict["MG_batch_size"] = 128
                    experiment_dict["criterion"] = CETripletMG  # [nn.CrossEntropyLoss, TripletMG, CETripletMG]

                    CRITERION = experiment_dict["criterion"]()

                    inference_time = testNetwork(model=model, dataset=dataset, test_mask=np.arange(100), criterion=CRITERION, batch_size=128, device=device)

                    print(f"RES: {RES}    RADIUS: {RADIUS}  RINGS:{RINGS}  POINTS: {POINTS}")
                    print(f"SpiderPatch", sum(spider_path_times) / len(spider_path_times))
                    print(f"MeshGraph", sum(mesh_graph_times) / len(mesh_graph_times))
                    print(f"Inference", inference_time)
                    print("\n\n")


if __name__ == "__main__":
    main()
