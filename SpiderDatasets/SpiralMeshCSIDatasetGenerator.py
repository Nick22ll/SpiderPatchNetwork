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

from MeshGraph.SpiralMeshCSI import SpiralMeshCSI
from SHREC_Utils import subdivide_for_mesh
from CSIRS.CSIRS import CSIRSv2Spiral


def generateSpiralMeshCSIDataset(to_extract="all"):
    """

    @param configurations:
    @param to_extract:
    @return:
    """
    mesh_dataset_path = "Datasets/Meshes/SHREC17"
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print(f"Loading SpiralMeshCSI Dataset from: {mesh_dataset_path}")
    import re
    rng = np.random.default_rng(22)
    radius, rings, points = 20, 4, 6
    node_radius, node_rings, node_points = 10, 4, 6
    resolution_level = "level_0"
    graph_for_mesh = 30
    connection_number = 0
    meshes = subdivide_for_mesh(return_type="list")
    save_path = f"Datasets/SpiralMeshCSI/SHREC17_R{radius}_RI{rings}_P{points}_NR{node_radius}_NRI{node_rings}_NP{node_points}"

    # Iterate over meshes of the mesh dataset
    print(f"Generation of concentric rings STARTED!")

    if to_extract == "all":
        mesh_to_extract = meshes
    else:
        mesh_to_extract = to_extract

    mesh_remaining = len(mesh_to_extract)

    if resolution_level != "all":
        for label in os.listdir(mesh_dataset_path):
            for mesh_sample_id in os.listdir(f"{mesh_dataset_path}/{label}"):
                mesh_filename = os.listdir(f"{mesh_dataset_path}/{label}/{mesh_sample_id}/resolution_{resolution_level}")[0]
                mesh_id = int(re.sub(r"\D", "", mesh_filename))
                if mesh_id not in mesh_to_extract:
                    continue
                print(f"Generation of patches on sample {mesh_id} STARTED!")
                print(f"Remaining {mesh_remaining} to the end...")
                mesh_remaining -= 1
                with open(f"{mesh_dataset_path}/{label}/{mesh_sample_id}/resolution_{resolution_level}/{mesh_filename}", "rb") as pkl_file:
                    mesh = pkl.load(pkl_file)
                    rng = np.random.default_rng(666)
                    boundary_vertices = mesh.getBoundaryVertices(neighbors_level=int(np.ceil(1.4 * radius) / mesh.edge_length))
                    seed_point_sequence = [x for x in range(len(mesh.vertices) - 1) if x not in boundary_vertices]
                    seed_point_sequence = list(rng.choice(seed_point_sequence, int(len(seed_point_sequence) * 0.30), replace=False))
                    rng.shuffle(seed_point_sequence)
                    samples = []
                    for iteration, seed_point in enumerate(seed_point_sequence):
                        if len(samples) >= graph_for_mesh or iteration > 300:
                            break
                        if seed_point in boundary_vertices:
                            continue
                        try:
                            spiral_mesh = SpiralMeshCSI(mesh, seed_point, radius, rings, points, node_radius, node_rings, node_points, neighbours_number=connection_number, resolution_level=resolution_level)
                            samples.append(spiral_mesh)
                        except (RuntimeError, np.linalg.LinAlgError) as err:
                            # print(err)
                            continue
                    print(f"Generated {len(samples)} graph on mesh {mesh_id}!")
                    os.makedirs(save_path, exist_ok=True)
                    os.makedirs(f'{save_path}/{label}', exist_ok=True)
                    os.makedirs(f'{save_path}/{label}/{mesh_sample_id}', exist_ok=True)
                    os.makedirs(f'{save_path}/{label}/{mesh_sample_id}/resolution_{resolution_level}', exist_ok=True)
                    with open(f'{save_path}/{label}/{mesh_sample_id}/resolution_{resolution_level}/spiralMeshCSI{mesh_id}.pkl', 'wb') as file:
                        pickle.dump(samples, file, protocol=-1)


    else:
        for label in os.listdir(f"{mesh_dataset_path}"):
            for mesh_sample_id in tqdm(os.listdir(f"{mesh_dataset_path}/{label}"), position=1, desc=f"Sample loading: ", colour="white", ncols=80, leave=False):
                for resolution_level in os.listdir(f"{mesh_dataset_path}/{label}/{mesh_sample_id}"):
                    mesh_filename = os.listdir(f"{mesh_dataset_path}/{label}/{mesh_sample_id}/resolution_{resolution_level}")[0]
                    mesh_id = int(re.sub(r"\D", "", mesh_filename))
                    if mesh_id not in mesh_to_extract:
                        continue
                    print(f"Generation of patches on sample {mesh_id} STARTED!")
                    print(f"Remaining {mesh_remaining} to the end...")
                    mesh_remaining -= 1
                    with open(f"{mesh_dataset_path}/{label}/{mesh_sample_id}/{resolution_level}/{mesh_filename}", "rb") as pkl_file:
                        mesh = pkl.load(pkl_file)
                        rng = np.random.default_rng(22)
                        boundary_vertices = mesh.getBoundaryVertices(neighbors_level=int(np.ceil(1.4 * radius) / mesh.edge_length))
                        seed_point_sequence = [x for x in range(len(mesh.vertices) - 1) if x not in boundary_vertices]
                        seed_point_sequence = list(rng.choice(seed_point_sequence, int(len(seed_point_sequence) * 0.30), replace=False))
                        rng.shuffle(seed_point_sequence)
                        samples = []
                        for iteration, seed_point in enumerate(seed_point_sequence):
                            if len(samples) >= graph_for_mesh or iteration > 100:
                                break
                            if seed_point in boundary_vertices:
                                continue
                            try:
                                spiral_mesh = SpiralMeshCSI(mesh, seed_point, radius, rings, points, node_radius, node_rings, node_points, neighbours_number=connection_number, resolution_level=resolution_level)
                                samples.append(spiral_mesh)
                            except RuntimeError:
                                continue
                        print(f"Generated {len(samples)} graph on mesh {mesh_id}!")
                        os.makedirs(save_path, exist_ok=True)
                        os.makedirs(f'{save_path}/{label}', exist_ok=True)
                        os.makedirs(f'{save_path}/{label}/{mesh_sample_id}', exist_ok=True)
                        os.makedirs(f'{save_path}/{label}/{mesh_sample_id}/resolution_{resolution_level}', exist_ok=True)
                        with open(f'{save_path}/{label}/{mesh_sample_id}/resolution_{resolution_level}/spiralMeshCSI{mesh_id}.pkl', 'wb') as file:
                            pickle.dump(samples, file, protocol=-1)


def parallelGenerateSpiralMeshCSIDataset(to_extract=None):
    if to_extract is None:
        to_extract = subdivide_for_mesh(return_type="list")
    thread_num = 8
    pool = multiprocessing.Pool(processes=thread_num)
    mesh_for_thread = int(len(to_extract) / thread_num)
    pool.map(generateSpiralMeshCSIDataset, [l for l in [to_extract[i * mesh_for_thread: (i * mesh_for_thread) + mesh_for_thread] for i in range(thread_num)]])
