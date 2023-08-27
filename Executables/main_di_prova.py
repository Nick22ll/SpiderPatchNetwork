import inspect
import os
import pickle
import re
import sys
from time import sleep, time
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
from PlotUtils import plotCVConfusionMatrix, plot_confusion_matrix, save_confusion_matrix
from SHREC_Utils import readPermSHREC17
from SpiderDatasets.MeshGraphDataset import MeshGraphDataset
from SpiderPatch.SpiderPatch import SP_distanceV1, SP_distanceV2, SpiderPatch
import gdist


def IQROutliers(data, iqr_multiplier=1.5):
    q25, q75 = percentile(data, 25), percentile(data, 75)
    iqr = q75 - q25
    cut_off = iqr * iqr_multiplier
    lower, upper = q25 - cut_off, q75 + cut_off
    return lower, upper


def SDOutliers(data, deviations=3):
    data_mean, data_std = np.mean(data), np.std(data)
    cut_off = data_std * deviations
    lower, upper = data_mean - cut_off, data_mean + cut_off
    return lower, upper


def set_node_weights(dataset):
    for mesh_graph in tqdm(dataset.graphs):
        for patch in mesh_graph.patches:
            weights = []
            for vertex_id, vertex in enumerate(patch.ndata["vertices"]):
                edge_indices = np.where(vertex_id == patch.edges()[1])[0]
                weight = torch.mean(patch.edata["node_distance"][edge_indices])
                weights.append(weight)
            patch.ndata["weight"] = torch.tensor(np.array(weights).reshape((-1, 1)), dtype=torch.float32)
    dataset.save_to(f"Datasets")


def main():
    # os.chdir(os.path.dirname(sys.argv[0]))
    #
    # rng = np.random.default_rng(233)
    #
    # with open("U:\AssegnoDiRicerca\PythonProject\Datasets\Meshes\SHREC17\class_0\id_1/resolution_level_0\mesh99.pkl", "rb") as mesh_file:
    #     mesh = pkl.load(mesh_file)
    #
    # # seed_points = [ 24419, 23779 ]
    # seed_points = rng.choice(range(len(mesh.vertices)), 10)
    # concs = [CSIRS.CSIRS.CSIRSv2(mesh, seed_point, 10, 6, 8) for seed_point in seed_points]
    # for conc in concs:
    #     mesh.drawWithConcRings(conc, lrf=False)
    # spiderPatches = [SpiderPatch(concs[i], mesh, seed_points[i]) for i in range(len(concs))]
    # # mesh.drawWithSpiderPatches([spiderPatches[0]])
    # mesh_graph = MeshGraph(spiderPatches, 2)
    # mesh.drawWithMeshGraph(mesh_graph)
    #
    # plotCVConfusionMatrix("../TrainingResults/tmp", cmap_name="Greens")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.chdir(os.path.dirname(sys.argv[0]))

    dataset_name = "SHREC17_R10_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN10_RES3"
    # dataset_name = "SHREC17_R10_R6_P8_CSIRSv2Spiral_MGRAPHS5_SPIDER20_CONN5_RES3"

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
    conn = re.search('_CONN(\d*)_', dataset_name)
    if not conn:
        conn = re.search('_CONN(\d*)', dataset_name)
    experiment_dict["MG_connectivity"] = [int(conn.group(1) if conn else -1)]

    res = re.search('_RES(\d*)', dataset_name)
    experiment_dict["mesh_resolution"] = [int(res.group(1) if res else -1)]
    experiment_dict["dataset_name"] = [dataset_name]

    dataset.keepCurvaturesResolution(radius_to_keep)
    dataset.aggregateSpiderPatchesNodeFeatures(features[features_to_keep], "aggregated_feats")
    dataset.aggregateSpiderPatchesNodeFeatures(["weights", "rings", "points"], "aggregated_weights")
    dataset.aggregateSpiderPatchEdgeFeatures()
    dataset.removeNonAggregatedFeatures()

    train_mask, test_mask = dataset.getTrainTestMask(10, percentage=False)

    mode = "normalization"  # ["standardization", "normalization", "robust", "quantile"]
    elim_mode = None  # ["standard", "quantile", None]

    experiment_dict["normalization_mode"] = mode
    experiment_dict["normalization_elim_mode"] = elim_mode if elim_mode is not None else "None"

    node_normalizers = dataset.normalizeV2(train_mask, mode, elim_mode)

    for name in [
        "SHREC17PERTURBED_R10_R6_P8_CSIRSv2Spiral_MGRAPHS25_SPIDER50_CONN10"
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
        conn = re.search('_CONN(\d*)_', dataset_name)
        if not conn:
            conn = re.search('_CONN(\d*)', dataset_name)
        experiment_dict["MG_connectivity"] = [int(conn.group(1) if conn else -1)]

        res = re.search('_RES(\d*)', dataset_name)
        experiment_dict["mesh_resolution"] = [int(res.group(1) if res else -1)]
        experiment_dict["dataset_name"] = [dataset_name]

        # print(dataset.graphs[0].patches[0].node_attr_schemes())
        # dataset.removeClasses([5, 10, 11, 12, 13])
        # dataset.removeSpiderPatchByNumNodes((rings * points) + 1)
        dataset.keepCurvaturesResolution(radius_to_keep)
        dataset.aggregateSpiderPatchesNodeFeatures(features[features_to_keep], "aggregated_feats")
        dataset.aggregateSpiderPatchesNodeFeatures(["weights", "rings", "points"], "aggregated_weights")
        dataset.aggregateSpiderPatchEdgeFeatures()
        dataset.removeNonAggregatedFeatures()

        # train_mask, test_mask = dataset.getTrainTestMask(10, percentage=False)
        # cross_validation_mask = dataset.getSHREC20CrossValidationMask()

        # for fold, masks in cross_validation_mask.items():
        #
        #     if fold in [2,3,4]:
        #         continue

        # train_mask = masks["train_indices"]
        # test_mask = masks["test_indices"]

        # class_num = len(np.unique(dataset.labels[train_mask]))
        class_num = len(np.unique(dataset.labels))

        mode = "normalization"  # ["standardization", "normalization", "robust", "quantile"]
        elim_mode = None  # ["standard", "quantile", None]

        experiment_dict["normalization_mode"] = mode
        experiment_dict["normalization_elim_mode"] = elim_mode if elim_mode is not None else "None"

        dataset.normalize_nodes(node_normalizers)
        #
        # dataset.normalize_edge(train_mask)
        # plot_data_distributions(dataset, train_mask)

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

        model = TestNetwork(feat_in_channels, weights_in_channels, class_num, network_parameters=network_parameters, use_SP_triplet=False, sp_nodes=(rings * points) + 1)

        model.load(f"U:\AssegnoDiRicerca\PythonProject\TrainingResults\Experiments/20032023-165234\MeshNetworkBestAcc/network.pt")

        model.to(device)

        #### TRAINING PARAMETERS  ####
        experiment_dict["MG_batch_size"] = 128
        experiment_dict["criterion"] = CETripletMG  # [nn.CrossEntropyLoss, TripletMG, CETripletMG]

        CRITERION = experiment_dict["criterion"]()

        mesh_graph_classification_statistics = testNetwork(model=model, dataset=dataset, test_mask=np.arange(len(dataset.labels)), criterion=CRITERION, batch_size=128, device=device)
        test_meters, cm = mesh_graph_classification_statistics
        with open(f"perturbed.pkl", "wb") as matrices_file:
            pkl.dump(cm, matrices_file)

        save_confusion_matrix(cm[0], "Res", f"PerturbedMeshGraphConfusionMatrix.png")
        save_confusion_matrix(cm[1], "Res", f"PerturbedMeshGraphConfusionMatrixAbs.png")
        to_write = f"Validation Test -->"
        losses_summary = {}
        for i, (name, test_meter) in enumerate(test_meters.items()):
            to_write += f"{test_meter.name}: {test_meter.avg:.3f}    "
            if "Loss" in test_meter.name:
                losses_summary[test_meter.name] = test_meter.avg
        to_write += "\n"
        print(to_write)

    # with open(f"U:\AssegnoDiRicerca\PythonProject\Datasets/Meshes\SHREC20\class_0\mesh12.pkl", "rb") as mesh_file:
    #     mesh = pickle.load(mesh_file)

    # with open(f"U:\AssegnoDiRicerca\PythonProject\Datasets\MeshGraphs\SHREC20_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS5_SPIDER20_CONN5.pkl", "rb") as mesh_file:
    #     dataset = pickle.load(mesh_file)
    # dataset.getSHREC20CrossValidationMask()
    #
    # mesh.drawWithMeshGraph(dataset.graphs[5])
    # mesh.drawWithGeodesicMeshGraph(dataset.graphs[6])
    #
    # geoalg = geodesic.PyGeodesicAlgorithmExact(mesh.vertices, mesh.faces)
    # distance, path = geoalg.geodesicDistance(13861, 33904)
    #
    # mesh.drawWithGeodesicPath(path)
    #
    # source_indices = np.array([25, 100, 50, 897, 567, 53, 345, 3456, 675, 987, 453, 5647, 456, 243, 6785, 987,635,763,286,2323,2346,4542])
    #
    # matrice_distanze = np.zeros((len(source_indices), len(source_indices)))
    # start = time()
    # for i in range(len(source_indices)):
    #     distanza = gdist.compute_gdist(mesh.vertices, mesh.faces, np.array([source_indices[i]]), source_indices[i + 1:])
    #     matrice_distanze[i, i + 1:] = distanza
    #     matrice_distanze[i + 1:, i] = distanza
    # print(time() - start)
    #
    # print(f"\n{matrice_distanze}\n\n")
    #
    # start = time()
    # matrice_distanze = gdist.distance_matrix_of_selected_points(mesh.vertices, mesh.faces, source_indices)
    # print(time() - start)
    # print(f"\n{matrice_distanze}\n\n")
    #
    # network_parameters = {}
    #
    # ####  SPIDER PATCH PARAMETERS  ####
    # network_parameters["SP"] = {}
    # network_parameters["SP"]["module"] = GATSPEmbedder  # [ CONVSPEmbedder, GATSPEmbedder , GATWeightedSP, SPReader]
    # network_parameters["SP"]["readout_function"] = "AR"  # [ "AR" , "UR", "SR"]
    # network_parameters["SP"]["jumping_mode"] = "cat"  # [ None, "lstm", "max", "cat"]
    # network_parameters["SP"]["layers_num"] = 4
    # network_parameters["SP"]["dropout"] = 0
    #
    # # GAT params
    # network_parameters["SP"]["residual"] = True  # bool
    # network_parameters["SP"]["exp_heads"] = False  # bool
    #
    # # Node Weigher params
    # network_parameters["SP"]["weigher_mode"] = "attn_weights+feats"  # [ "sp_weights", "attn_weights",  "attn_weights+feats" , None ]
    #
    # ####  MESH GRAPH PARAMETERS  ####
    # network_parameters["MG"] = {}
    # network_parameters["MG"]["module"] = GATMGEmbedder  # [ CONVMGEmbedder, GATMGEmbedder]
    # network_parameters["MG"]["readout_function"] = "AR"  # [ "AR" , "UR" ]
    # network_parameters["MG"]["jumping_mode"] = "cat"  # [ None, "lstm", "max", "cat"]
    # network_parameters["MG"]["layers_num"] = 3
    # network_parameters["MG"]["dropout"] = 0
    # network_parameters["MG"]["SP_batch_size"] = 512
    #
    # # GAT params
    # network_parameters["MG"]["residual"] = True  # bool
    # network_parameters["MG"]["exp_heads"] = False  # bool
    #
    # model = TestNetwork(25, 3, 15, network_parameters=network_parameters, use_SP_triplet=False)
    # model.load("U:\AssegnoDiRicerca\PythonProject\TrainingResults\Experiments/08042023-235725\MeshNetworkBestAcc/network.pt")
    #
    # a = 0

    # visited_label = []
    # samples = readPermSHREC17()
    # mesh_ids = []
    # mesh_classes = []
    # for dictionary in samples:
    #     for level, mesh_tuple in dictionary.items():
    #         mesh_ids.append(mesh_tuple[0])
    #         mesh_classes.append(mesh_tuple[1])
    # meshes = []
    # for id, sample in enumerate(samples):
    #     mesh_id = sample["level_3"][0]
    #     mesh_label = sample["level_3"][1]
    #     # if mesh_label != 12:
    #     #     continue
    #
    #     if mesh_id != 448:
    #         continue
    #     visited_label.append(mesh_label)
    #     print(mesh_id)
    #
    #     with open(f"U:\AssegnoDiRicerca\PythonProject\Datasets/Meshes\SHREC17\class_{mesh_label}\id_{id}/resolution_level_3\mesh{mesh_id}.pkl", "rb") as mesh_file:
    #         mesh = pickle.load(mesh_file)

    # mesh.drawWithGaussCurv(4)
    # mesh.drawWithMeanCurv(4)
    # mesh.drawWithCurvedness(4)
    # mesh.drawWithK2(4)
    # mesh.drawWithLD(4)
    # mesh.draw()

    # with open("U:\AssegnoDiRicerca\PythonProject\Datasets\Meshes\SHREC17\class_0\id_0/resolution_level_3\mesh448.pkl", "rb") as mesh_file:
    # with open("U:\AssegnoDiRicerca\PythonProject\Datasets/NormalizedMeshes\SHREC17\class_0\id_0/resolution_level_3\mesh448.pkl", "rb") as mesh_file:
    # mesh = pickle.load(mesh_file)
    # with open("U:\AssegnoDiRicerca\PythonProject\Datasets\SpiderPatches\SHREC17_R10_R4_P6_CSIRSv2Spiral\class_0\id_0/resolution_level_3\spiderPatches448.pkl", "rb") as conc_file:
    # # with open("U:\AssegnoDiRicerca\PythonProject\Datasets\SpiderPatches\SHREC17_R0.1_R6_P8_CSIRSv2Spiral\class_0\id_0/resolution_level_3\spiderPatches448.pkl", "rb") as conc_file:
    #     spider_patches = pickle.load(conc_file)
    # seed_spider = spider_patches[0]
    # rng = np.random.default_rng(233)

    # cls = 6
    # mesh_id = 184
    #
    # with open(f"../Datasets/Meshes/SHREC20/class_{cls}/mesh{mesh_id}.pkl", "rb") as mesh_file:
    #     mesh = pickle.load(mesh_file)
    #
    # with open(f"../Datasets/SpiderPatches/SHREC20_R0.1_R6_P8_CSIRSv2Spiral/class_{cls}/concRing{mesh_id}.pkl", "rb") as mesh_file:
    #     spider_patches = pickle.load(mesh_file)
    #
    # for i in range(0, len(spider_patches), 20):
    #     mesh.drawWithSpiderPatches(spider_patches[i:i+20])

    # dataset_name = "SHREC20_R0.1_R6_P8_CSIRSv2Spiral_MGRAPHS10_SPIDER50_CONN5"
    #
    # dataset = MeshGraphDataset(dataset_name=dataset_name)
    # dataset.load_from(f"../Datasets/MeshGraphs", dataset_name)
    #
    # for _ in range(100):
    #     i = rng.choice(range(len(dataset.graphs)))
    #     mesh_graph = dataset.graphs[i]
    #     print(dataset.labels[i])
    #     with open(f"../Datasets/Meshes/SHREC20/class_{dataset.labels[i]}/mesh{mesh_graph.mesh_id}.pkl", "rb") as mesh_file:
    #         mesh = pickle.load(mesh_file)
    #     mesh.drawWithMeshGraph(mesh_graph)

    # # seed_points = [  8, 168,4241, 4460, 3487,1072,2709,2108,2562,6033,6282,4541 ]
    # seed_points = rng.choice(range(len(mesh.vertices)), 10)
    # concs = [CSIRS.CSIRS.CSIRSv2Spiral(mesh, seed_point, 0.1, 6, 8) for seed_point in seed_points]
    # for conc in concs:
    #     mesh.drawWithConcRings(conc, lrf=False)
    # spiderPatches = [SpiderPatch(concs[i], mesh, seed_points[i]) for i in range(len(concs))]
    # # mesh.drawWithSpiderPatches([spiderPatches[0]])
    # mesh_graph = MeshGraph(spiderPatches, 2)
    # mesh.drawWithMeshGraph(mesh_graph)

    # spider_patches_to_draw = rng.choice(spider_patches[1:], 11, replace=False)
    # distances = [SP_distanceV1(seed_spider, SP, "local_depth") for SP in spider_patches_to_draw]
    # norm = colors.Normalize(vmin=0, vmax=max(distances))
    # color_map = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('hot'))
    # rgbs = color_map.to_rgba(distances)[:, :3]

    # ord_distances_idx = np.hstack((np.argsort(distances)[:100], np.argsort(distances)[-1]))
    # mesh.drawWithSpiderPatches(np.hstack((np.array(seed_spider), spider_patches_to_draw[ord_distances_idx])), np.vstack(((0, 1, 0), rgbs[ord_distances_idx])))
    # mesh.drawWithSpiderPatches(spider_patches_to_draw, )


if __name__ == "__main__":
    main()
