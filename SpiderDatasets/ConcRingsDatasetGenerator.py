import multiprocessing
import os
import re
import pickle
import random
import warnings

import numpy as np
import pickle as pkl
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from Mesh import *
from math import floor

from Mesh.Mesh import Mesh
from SHREC_Utils import subdivide_for_mesh
from CSIRS.CSIRS import CSIRSv2, CSIRSv2Spiral
from SpiderPatch.SpiderPatch import SpiderPatchLRF
from SpiderPatch.SuperPatch import SuperPatch_preloaded

DATASETS = {
    "SHREC17": ("../MeshDataset/SHREC17", ".off")
}


def generateConcRingDataset(to_extract="all", configurations=None, CSIRS_type=CSIRSv2Spiral):
    """

    @param configurations:
    @param to_extract:
    @return:
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if configurations is None:
        configurations = np.array([[5, 4, 12], [5, 4, 6], [7, 4, 6], [10, 4, 6], [10, 7, 12]])

    conc_ring_per_mesh = 1000

    meshes = subdivide_for_mesh(return_type="list")
    load_path = f"Datasets/Meshes/SHREC17"
    save_path = f"Datasets/ConcentricRings/SHREC17"

    # Iterate over meshes of the mesh dataset
    print(f"Generation of concentric rings STARTED!")

    if to_extract == "all":
        mesh_to_extract = meshes
    else:
        mesh_to_extract = [meshes[i] for i in to_extract]

    mesh_remaining = len(mesh_to_extract)

    for label in os.listdir(load_path):
        for id in os.listdir(f"{load_path}/{label}"):
            for resolution_level in os.listdir(f"{load_path}/{label}/{id}"):
                mesh_id = os.listdir(f"{load_path}/{label}/{id}/{resolution_level}")[0]
                if int(re.sub(r"\D", "", mesh_id)) not in mesh_to_extract:
                    continue
                print(f"Generation of patches on sample {mesh_id} STARTED!")
                print(f"Remaining {mesh_remaining} to the end...")
                mesh_remaining -= 1
                with open(f"{load_path}/{label}/{id}/{resolution_level}/{mesh_id}", "rb") as mesh_file:
                    mesh = pkl.load(mesh_file)

                vertices_number = len(mesh.vertices)
                # Under development uses a fixed seed points sequence
                random.seed(666)
                seed_point_sequence = list(np.unique(random.sample(range(vertices_number - 1), 4000)))
                random.shuffle(seed_point_sequence)
                for config in configurations:
                    radius, rings, points = config
                    boundary_vertices = mesh.getBoundaryVertices(neighbors_level=int(np.ceil(1.2 * radius)))
                    # Generate N number of patches for a single mesh
                    processed_conc_rings = 1
                    concentric_rings_list = []

                    for seed_point in seed_point_sequence:
                        if processed_conc_rings % (conc_ring_per_mesh + 1) == 0:
                            break

                        if seed_point in boundary_vertices:
                            continue

                        concentric_rings = CSIRS_type(mesh, seed_point, radius, rings, points)
                        if not concentric_rings.first_valid_rings(2):
                            continue

                        concentric_rings_list.append(concentric_rings)
                        processed_conc_rings += 1

                    os.makedirs(f"{save_path}_R{radius}_RI{rings}_P{points}", exist_ok=True)
                    os.makedirs(f'{save_path}_R{radius}_RI{rings}_P{points}/{label}', exist_ok=True)
                    os.makedirs(f'{save_path}_R{radius}_RI{rings}_P{points}/{label}/{id}', exist_ok=True)
                    os.makedirs(f'{save_path}_R{radius}_RI{rings}_P{points}/{label}/{id}/{resolution_level}', exist_ok=True)
                    mesh_id = re.sub(r"\D", "", mesh_id)
                    with open(f'{save_path}_R{radius}_RI{rings}_P{points}/{label}/{id}/{resolution_level}/concRing{mesh_id}.pkl', 'wb') as file:
                        pickle.dump(concentric_rings_list, file, protocol=-1)


def parallelGenerateConcRingDataset(to_extract=None):
    if to_extract is None:
        to_extract = subdivide_for_mesh(return_type="list")
    thread_num = 6
    pool = multiprocessing.Pool(processes=thread_num)
    mesh_for_thread = int(len(to_extract) / thread_num)
    pool.map(generateConcRingDataset, [l for l in [to_extract[i * mesh_for_thread: (i * mesh_for_thread) + mesh_for_thread] for i in range(thread_num)]])
