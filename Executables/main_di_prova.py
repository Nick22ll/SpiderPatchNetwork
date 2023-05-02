import pickle

import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import torch
from numpy import percentile
from tqdm import tqdm

import CSIRS.CSIRS
from MeshGraph.MeshGraph import MeshGraph
from PlotUtils import plotCVConfusionMatrix
from SHREC_Utils import subdivide_for_mesh
from SpiderPatch.SpiderPatch import SP_distanceV1, SP_distanceV2, SpiderPatch


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
    # plotCVConfusionMatrix("TrainingResults/CrossValExperiments")

    visited_label = []
    samples = subdivide_for_mesh()
    mesh_ids = []
    mesh_classes = []
    for dictionary in samples:
        for level, mesh_tuple in dictionary.items():
            mesh_ids.append(mesh_tuple[0])
            mesh_classes.append(mesh_tuple[1])
    meshes = []
    for id, sample in enumerate(samples):
        mesh_id = sample["level_3"][0]
        mesh_label = sample["level_3"][1]
        # if mesh_label != 12:
        #     continue

        if mesh_id != 448:
            continue
        visited_label.append(mesh_label)
        print(mesh_id)

        with open(f"U:\AssegnoDiRicerca\PythonProject\Datasets/Meshes\SHREC17\class_{mesh_label}\id_{id}/resolution_level_3\mesh{mesh_id}.pkl", "rb") as mesh_file:
            mesh = pickle.load(mesh_file)

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
    rng = np.random.default_rng(233)

    # seed_points = [  8, 168,4241, 4460, 3487,1072,2709,2108,2562,6033,6282,4541 ]
    seed_points = [3674]
    concs = [CSIRS.CSIRS.CSIRSv2Spiral(mesh, seed_point, 10, 10, 6) for seed_point in seed_points]
    for conc in concs:
        mesh.drawWithConcRings(conc, lrf=False)
    spiderPatches = [SpiderPatch(concs[i], mesh, seed_points[i]) for i in range(len(concs))]
    # mesh.drawWithSpiderPatches([spiderPatches[0]])
    mesh_graph = MeshGraph(spiderPatches, 2)
    mesh.drawWithMeshGraph(mesh_graph)

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
