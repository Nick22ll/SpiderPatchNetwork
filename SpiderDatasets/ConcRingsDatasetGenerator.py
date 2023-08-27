import multiprocessing
import os
import pathlib
import pickle
import pickle as pkl
import random
import re
import warnings

import numpy as np
from tqdm import tqdm

from CSIRS.CSIRS import CSIRSv2Spiral
from SHREC_Utils import readPermSHREC17, readPermSHREC20

# CONC_RING_PER_MESH = 1000
CONC_RING_PER_MESH = 400


def generateConcRingDataset(dataset_name, to_extract, configurations, use_normalized_meshes=False, progress_bar_position=0):
    """

    @param dataset_name: (string)
    @param configurations: a list of dicts [{"radius": int, "rings": int, "points": int, "CSIRS_type": string, "relative_radius":bool}, ...]
    @param to_extract: "all" or a list of ints
    @param use_normalized_meshes:
    @return:
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if "SHREC17" in dataset_name:
        meshes = readPermSHREC17(return_type="list")
    elif "SHREC20" in dataset_name:
        meshes = readPermSHREC20(return_type="list")
    else:
        raise Exception("Dataset Name not found!")

    if use_normalized_meshes:
        load_path = f"Datasets/NormalizedMeshes/{dataset_name}"
        save_path = f"Datasets/ConcentricRings/{dataset_name}_NORM_"
    else:
        load_path = f"Datasets/Meshes/{dataset_name}"
        save_path = f"Datasets/ConcentricRings/{dataset_name}"

    # Iterate over meshes of the mesh dataset

    if to_extract is None:
        mesh_to_extract = meshes
    else:
        mesh_to_extract = to_extract

    mesh_remaining = len(mesh_to_extract)
    pbar = tqdm(total=mesh_remaining, position=progress_bar_position, desc="Completed meshes: ")

    if "SHREC17" in dataset_name:
        for label in os.listdir(load_path):
            for sample_id in os.listdir(f"{load_path}/{label}"):
                for resolution_level in os.listdir(f"{load_path}/{label}/{sample_id}"):
                    mesh_id = os.listdir(f"{load_path}/{label}/{sample_id}/{resolution_level}")[0]
                    if int(re.sub(r"\D", "", mesh_id)) not in mesh_to_extract:
                        continue
                    with open(f"{load_path}/{label}/{sample_id}/{resolution_level}/{mesh_id}", "rb") as mesh_file:
                        mesh = pkl.load(mesh_file)
                    final_part_save_path = f"/{label}/{sample_id}/{resolution_level}"
                    mesh_id = re.sub(r"\D", "", mesh_id)
                    generateConcRings(mesh, mesh_id, configurations, save_path, final_part_save_path)
                    pbar.update(1)

    elif "SHREC20" in dataset_name:
        for label in os.listdir(load_path):
            for mesh_filename in os.listdir(f"{load_path}/{label}"):
                mesh_id = int(re.sub(r"\D", "", mesh_filename))
                if mesh_id not in mesh_to_extract:
                    continue
                with open(f"{load_path}/{label}/{mesh_filename}", "rb") as mesh_file:
                    mesh = pkl.load(mesh_file)
                final_part_save_path = f"/{label}"
                generateConcRings(mesh, mesh_id, configurations, save_path, final_part_save_path)
                pbar.update(1)

    pbar.close()


def generateConcRings(mesh, mesh_id, configurations, save_path, final_part_save_path=""):
    vertices_number = len(mesh.vertices)
    # Under development uses a fixed seed points sequence
    rng = np.random.default_rng(717)
    seed_point_sequence = list(rng.choice(range(vertices_number - 1), min(CONC_RING_PER_MESH * 2, int(vertices_number * 0.80)), replace=False))
    rng.shuffle(seed_point_sequence)
    for config in configurations:
        radius, rings, points = config["radius"], config["rings"], config["points"]
        rings = int(rings)
        points = int(points)
        if config["relative_radius"]:
            radius = radius * mesh.edge_length
        boundary_vertices = mesh.getBoundaryVertices(neighbors_level=int(np.ceil(radius / mesh.edge_length)))
        # Generate N number of patches for a single mesh
        processed_conc_rings = 1
        concentric_rings_list = []
        for seed_point in seed_point_sequence:
            if processed_conc_rings % (CONC_RING_PER_MESH + 1) == 0:
                break
            if seed_point in boundary_vertices:
                continue
            concentric_rings = config["CSIRS_type"](mesh, seed_point, radius, rings, points)
            if not concentric_rings.firstValidRings(2):
                continue
            concentric_rings_list.append(concentric_rings)
            processed_conc_rings += 1

        if config["relative_radius"]:
            radius, rings, points = config
            new_parent_file_path = f'{save_path}_RR{radius}_R{rings}_P{points}_{str(config["CSIRS_type"].__name__)}' + final_part_save_path
            pathlib.Path(new_parent_file_path).mkdir(parents=True, exist_ok=True)
        else:
            new_parent_file_path = f'{save_path}_R{radius}_R{rings}_P{points}_{str(config["CSIRS_type"].__name__)}' + final_part_save_path
            pathlib.Path(new_parent_file_path).mkdir(parents=True, exist_ok=True)

        with open(f'{new_parent_file_path}/concRing{mesh_id}.pkl', 'wb') as file:
            pickle.dump(concentric_rings_list, file, protocol=-1)


def parallelGenerateConcRingDataset(dataset_name, to_extract=None, configurations=None, use_normalized_meshs=False, num_thread=8):
    if to_extract is None:
        if "SHREC17" in dataset_name:
            to_extract = readPermSHREC17(return_type="list")
        elif "SHREC20" in dataset_name:
            to_extract = readPermSHREC20(return_type="list")
        else:
            raise Exception("Dataset Name not found!")
    mesh_for_thread = int(len(to_extract) / num_thread) + 1
    pool = multiprocessing.Pool(processes=num_thread)
    pool.starmap(generateConcRingDataset, [(dataset_name, to_extract[i * mesh_for_thread: (i * mesh_for_thread) + mesh_for_thread], configurations, use_normalized_meshs, i) for i in range(num_thread)])
