import pickle

import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import torch
from numpy import percentile
from tqdm import tqdm

from SpiderPatch.SpiderPatch import SP_distanceV1, SP_distanceV2


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
    with open("U:\AssegnoDiRicerca\PythonProject\Datasets/NormalizedMeshes\SHREC17\class_0\id_0/resolution_level_3\mesh448.pkl", "rb") as mesh_file:
        mesh = pickle.load(mesh_file)
    with open("U:\AssegnoDiRicerca\PythonProject\Datasets\SpiderPatches\SHREC17_R0.1_R6_P8_CSIRSv2Spiral\class_0\id_0/resolution_level_3\spiderPatches448.pkl", "rb") as conc_file:
        spider_patches = pickle.load(conc_file)
    seed_spider = spider_patches[0]
    rng = np.random.default_rng(17)
    spider_patches_to_draw = rng.choice(spider_patches[1:], 300, replace=False)
    distances = [SP_distanceV1(seed_spider, SP, "local_depth") for SP in spider_patches_to_draw]
    norm = colors.Normalize(vmin=0, vmax=max(distances))
    color_map = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('hot'))
    rgbs = color_map.to_rgba(distances)[:, :3]

    ord_distances_idx = np.hstack((np.argsort(distances)[:100], np.argsort(distances)[-1]))
    mesh.drawWithSpiderPatches(np.hstack((np.array(seed_spider), spider_patches_to_draw[ord_distances_idx])), np.vstack(((0, 1, 0), rgbs[ord_distances_idx])))

if __name__ == "__main__":
    main()
