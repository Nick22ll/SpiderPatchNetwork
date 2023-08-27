import multiprocessing
import os
import pathlib
import pickle
import warnings

import dgl
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from SHREC_Utils import readPermSHREC17
from SpiderPatch.SpiderPatch import SpiderPatch
from SpiderPatch.SuperPatch import SuperPatch_preloaded


def generateSPDatasetFromConcRings(concRings_path, valid_rings=2):
    warnings.filterwarnings("ignore")

    if "NORM" in concRings_path:
        mesh_path = concRings_path.replace("ConcentricRings", "NormalizedMeshes")
    else:
        mesh_path = concRings_path.replace("ConcentricRings", "Meshes")

    if "SHREC17" in concRings_path:
        mesh_path = mesh_path.replace(mesh_path[mesh_path.find("SHREC17"):], "SHREC17")
    elif "SHREC20" in concRings_path:
        mesh_path = mesh_path.replace(mesh_path[mesh_path.find("SHREC20"):], "SHREC20")
    else:
        raise Exception("No mesh dataset found!")

    for label in tqdm(os.listdir(concRings_path), position=0, desc=f"Label: ", colour="white", ncols=80):
        for sample_id in tqdm(os.listdir(f"{concRings_path}/{label}"), position=1, desc=f"ID: ", colour="white", ncols=80):
            if os.path.isdir(f"{concRings_path}/{label}/{sample_id}"):
                for resolution_level in os.listdir(f"{concRings_path}/{label}/{sample_id}"):
                    conc_filename = os.listdir(f"{concRings_path}/{label}/{sample_id}/{resolution_level}")[0]
                    generateSPFromCR(mesh_path=f"{mesh_path}/{label}/{sample_id}/{resolution_level}/{conc_filename.replace('concRing', 'mesh')}", concRings_path=f"{concRings_path}/{label}/{sample_id}/{resolution_level}/{conc_filename}", valid_rings=valid_rings)
            else:
                conc_filename = f"{concRings_path}/{label}/{sample_id}"
                generateSPFromCR(mesh_path=f"{mesh_path}/{label}/{sample_id.replace('concRing', 'mesh')}", concRings_path=f"{conc_filename}", valid_rings=valid_rings)


def generateSPFromCR(mesh_path, concRings_path, valid_rings):
    with open(f"{mesh_path}", "rb") as mesh_file:
        mesh = pickle.load(mesh_file)

    with open(f"{concRings_path}", "rb") as file:
        conc_rings = pickle.load(file)

    spider_patches = []
    for conc_ring in tqdm(conc_rings, position=2, desc=f"Concentric Rings: ", colour="white", ncols=80):
        if not conc_ring.firstValidRings(valid_rings):
            continue
        try:
            spider_patches.append(SpiderPatch(conc_ring, mesh, conc_ring.seed_point, seed_point_idx=False))
        except dgl.DGLError:
            continue
    save_path = f"{concRings_path.replace('ConcentricRings', 'SpiderPatches').replace('concRings', 'spiderPatches')}"
    pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as file:
        pickle.dump(spider_patches, file, protocol=-1)


def parallelGenerateSuperPatchDataset(to_extract=None):
    if to_extract is None:
        to_extract = [i for i in range(180)]
    thread_num = 6
    pool = multiprocessing.Pool(processes=thread_num)
    mesh_for_thread = int(len(to_extract) / thread_num)
    pool.map(generateSuperPatchDataset, [l for l in [to_extract[i * mesh_for_thread: (i * mesh_for_thread) + mesh_for_thread] for i in range(thread_num)]])


def generateSuperPatchDataset(to_extract="all"):
    configurations = np.array([[3, 4, 6], [5, 6, 6]])
    patch_per_mesh = 100

    meshes = readPermSHREC17()
    load_path = f"U:/AssegnoDiRicerca/PythonProject/Datasets/Meshes/SHREC17"
    save_path = f"Datasets/SuperPatches/SHREC17"

    # Iterate over meshes of the mesh dataset
    if to_extract == "all":
        to_extract = list(range(180))
        mesh_to_extract = meshes
    else:
        mesh_to_extract = [meshes[i] for i in to_extract]

    for sample_id, sample in tqdm(enumerate(mesh_to_extract)):
        print(f"Generation of patches on sample {to_extract[sample_id]} STARTED!")
        sample_meshes = []
        for resolution_level in ["level_0", "level_1", "level_2", "level_3"]:
            mesh_id = sample[resolution_level][0]
            label = sample[resolution_level][1]
            with open(f"{load_path}/class_{label}/id_{to_extract[sample_id]}/resolution_{resolution_level}/mesh{mesh_id}.pkl", "rb") as mesh_file:
                mesh = pickle.load(mesh_file)
            sample_meshes.append(mesh)

        vertices_number = len(sample_meshes[0].vertices)
        boundary_vertices = []
        for config in configurations:
            max_radius, _, _ = config
            max_radius *= 1.25
            distances = pairwise_distances(sample_meshes[0].vertices[sample_meshes[0].getBoundaryVertices()], sample_meshes[0].vertices, metric="sqeuclidean")
            boundary_vertices.append(np.where(np.any(distances < pow(max_radius, 2), axis=0))[0])

        # Under development uses a fixed seed points sequence
        rng = np.random.default_rng(22)
        seed_point_sequence = list(rng.choice(range(vertices_number - 1), 4000, replacement=False))
        rng.shuffle(seed_point_sequence)
        for config_id, config in enumerate(configurations):
            radius, rings, points = config
            # Generate N super patches for a single sample
            processed_patch = 1
            super_patches_list = []
            for seed_point in seed_point_sequence:
                if processed_patch % (patch_per_mesh + 1) == 0:
                    break

                if seed_point in boundary_vertices[config_id]:
                    continue
                try:
                    patch = SuperPatch_preloaded(to_extract[sample_id], sample_meshes, sample_meshes[0].vertices[seed_point] + [0.5, 0.5, 0.5], radius, rings, points)
                except RuntimeError as err:
                    # print(err, f"In sample{to_extract[sample_id]} at seed {seed_point}")
                    continue

                super_patches_list.append(patch)

                processed_patch += 1

            os.makedirs(f"{save_path}_R{radius}_RI{rings}_P{points}", exist_ok=True)
            os.makedirs(f"{save_path}_R{radius}_RI{rings}_P{points}/class_{str(label)}", exist_ok=True)
            os.makedirs(f"{save_path}_R{radius}_RI{rings}_P{points}/class_{str(label)}/id_{to_extract[sample_id]}", exist_ok=True)
            with open(f"{save_path}_R{radius}_RI{rings}_P{points}/class_{str(label)}/id_{to_extract[sample_id]}/patches{mesh_id}.pkl", 'wb') as file:
                pickle.dump(super_patches_list, file, protocol=-1)
