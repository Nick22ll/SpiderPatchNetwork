import multiprocessing
import os
import pickle
import pickle as pkl
import random
import re
import warnings

import numpy as np

from CSIRS.CSIRS import CSIRSv2Spiral
from SHREC_Utils import subdivide_for_mesh

DATASETS = {
    "SHREC17": ("../MeshDataset/SHREC17", ".off")
}


def generateConcRingDataset(to_extract="all", configurations=None, CSIRS_type=CSIRSv2Spiral, relative_radius=False):
    """

    @param CSIRS_type:
    @param configurations:
    @param to_extract:
    @return:
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if configurations is None:
        configurations = np.array([[10, 6, 8], [5, 4, 6]])  # [5, 4, 12], [10, 7, 12], [7, 4, 6],

    conc_ring_per_mesh = 500

    meshes = subdivide_for_mesh(return_type="list")
    load_path = f"Datasets/Meshes/SHREC17"
    save_path = f"Datasets/ConcentricRings/SHREC17"

    # Iterate over meshes of the mesh dataset
    print(f"Generation of concentric rings STARTED!")

    if to_extract == "all":
        mesh_to_extract = meshes
    else:
        mesh_to_extract = to_extract

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
                seed_point_sequence = list(np.unique(random.sample(range(vertices_number - 1), min(conc_ring_per_mesh * 2, int(vertices_number * 0.80)))))
                random.shuffle(seed_point_sequence)
                for config in configurations:
                    radius, rings, points = config
                    if relative_radius:
                        radius = radius * mesh.edge_length
                    boundary_vertices = mesh.getBoundaryVertices(neighbors_level=int(np.ceil((1.2 * radius) / mesh.edge_length)))
                    # Generate N number of patches for a single mesh
                    processed_conc_rings = 1
                    concentric_rings_list = []
                    for seed_point in seed_point_sequence:
                        if processed_conc_rings % (conc_ring_per_mesh + 1) == 0:
                            break

                        if seed_point in boundary_vertices:
                            continue
                        concentric_rings = CSIRS_type(mesh, seed_point, radius, rings, points)
                        if not concentric_rings.firstValidRings(2):
                            continue
                        concentric_rings_list.append(concentric_rings)
                        processed_conc_rings += 1
                    if relative_radius:
                        radius, rings, points = config
                        os.makedirs(f"{save_path}_RR{radius}_R{rings}_P{points}_{str(CSIRS_type.__name__)}", exist_ok=True)
                        os.makedirs(f'{save_path}_RR{radius}_R{rings}_P{points}_{str(CSIRS_type.__name__)}/{label}', exist_ok=True)
                        os.makedirs(f'{save_path}_RR{radius}_R{rings}_P{points}_{str(CSIRS_type.__name__)}/{label}/{id}', exist_ok=True)
                        os.makedirs(f'{save_path}_RR{radius}_R{rings}_P{points}_{str(CSIRS_type.__name__)}/{label}/{id}/{resolution_level}', exist_ok=True)
                        mesh_id = re.sub(r"\D", "", mesh_id)
                        with open(f'{save_path}_RR{radius}_R{rings}_P{points}_{str(CSIRS_type.__name__)}/{label}/{id}/{resolution_level}/concRing{mesh_id}.pkl', 'wb') as file:
                            pickle.dump(concentric_rings_list, file, protocol=-1)
                    else:
                        os.makedirs(f"{save_path}_R{radius}_R{rings}_P{points}_{str(CSIRS_type.__name__)}", exist_ok=True)
                        os.makedirs(f'{save_path}_R{radius}_R{rings}_P{points}_{str(CSIRS_type.__name__)}/{label}', exist_ok=True)
                        os.makedirs(f'{save_path}_R{radius}_R{rings}_P{points}_{str(CSIRS_type.__name__)}/{label}/{id}', exist_ok=True)
                        os.makedirs(f'{save_path}_R{radius}_R{rings}_P{points}_{str(CSIRS_type.__name__)}/{label}/{id}/{resolution_level}', exist_ok=True)
                        mesh_id = re.sub(r"\D", "", mesh_id)
                        with open(f'{save_path}_R{radius}_R{rings}_P{points}_{str(CSIRS_type.__name__)}/{label}/{id}/{resolution_level}/concRing{mesh_id}.pkl', 'wb') as file:
                            pickle.dump(concentric_rings_list, file, protocol=-1)


def parallelGenerateConcRingDataset(to_extract=None, configurations=None, relative_radius=False):
    if to_extract is None:
        to_extract = subdivide_for_mesh(return_type="list")
    thread_num = 13
    mesh_for_thread = int(len(to_extract) / thread_num)
    pool = multiprocessing.Pool(processes=thread_num)
    pool.starmap(generateConcRingDataset, [(to_extract[i * mesh_for_thread: (i * mesh_for_thread) + mesh_for_thread], configurations, CSIRSv2Spiral, relative_radius) for i in range(thread_num)])

    # pool.map(generateConcRingDataset, [l for l in [to_extract[i * mesh_for_thread: (i * mesh_for_thread) + mesh_for_thread] for i in range(thread_num)]])
