import multiprocessing
import os
import pickle
import warnings
from math import floor

import dgl
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from CSIRS.CSIRS import CSIRSv2
from Mesh.Mesh import Mesh
from SHREC_Utils import subdivide_for_mesh
from SpiderPatch.SpiderPatch import SpiderPatchLRF, SpiderPatch
from SpiderPatch.SuperPatch import SuperPatch_preloaded

DATASETS = {
    "SHREC17": ("../MeshDataset/SHREC17", ".off")
}


def generateSPDatasetFromConcRings(concRings_path, valid_rings=2, normalized=False):
    warnings.filterwarnings("ignore")
    if normalized:
        mesh_path = concRings_path.replace("ConcentricRings", "NormalizedMeshes")
    else:
        mesh_path = concRings_path.replace("ConcentricRings", "Meshes")

    mesh_path = mesh_path.replace(mesh_path[mesh_path.find("SHREC17"):], "SHREC17")

    for label in tqdm(os.listdir(concRings_path), position=0, leave=False, desc=f"Label: ", colour="white", ncols=80):
        for mesh_id in tqdm(os.listdir(f"{concRings_path}/{label}"), position=0, leave=False, desc=f"ID: ", colour="white", ncols=80):
            for resolution_level in tqdm(os.listdir(f"{concRings_path}/{label}/{mesh_id}"), position=0, leave=False, desc=f"Resolution Level: ", colour="white", ncols=80):
                conc_filename = os.listdir(f"{concRings_path}/{label}/{mesh_id}/{resolution_level}")[0]
                mesh_filename = conc_filename.replace("concRing", "mesh")

                with open(f"{mesh_path}/{label}/{mesh_id}/{resolution_level}/{mesh_filename}", "rb") as mesh_file:
                    mesh = pickle.load(mesh_file)

                with open(f"{concRings_path}/{label}/{mesh_id}/{resolution_level}/{conc_filename}", "rb") as file:
                    conc_rings = pickle.load(file)

                spider_patches = []
                for conc_ring in tqdm(conc_rings, position=0, leave=False, desc=f"Concentric Rings: ", colour="white", ncols=80):
                    if not conc_ring.firstValidRings(valid_rings):
                        continue
                    try:
                        spider_patches.append(SpiderPatch(conc_ring, mesh, conc_ring.seed_point, seed_point_idx=False))
                    except dgl.DGLError:
                        continue
                save_path = concRings_path.replace('ConcentricRings', 'SpiderPatches')
                os.makedirs(f"{save_path}", exist_ok=True)
                os.makedirs(f"{save_path}/{label}", exist_ok=True)
                os.makedirs(f"{save_path}/{label}/{mesh_id}", exist_ok=True)
                os.makedirs(f"{save_path}/{label}/{mesh_id}/{resolution_level}", exist_ok=True)
                with open(f"{save_path}/{label}/{mesh_id}/{resolution_level}/{conc_filename.replace('concRing', 'spiderPatches')}", 'wb') as file:
                    pickle.dump(spider_patches, file, protocol=-1)


def parallelGenerateLRFPatchDataset(to_extract="all"):
    """

    @param mesh_dataset:
    @param save_path:
    @param configurations:
    @param to_extract:
    @param patch_per_mesh:
    @return:
    """

    configurations = np.array([[10, 10, 10], [10, 6, 6]])
    patch_per_mesh = 1000

    meshes = subdivide_for_mesh()
    extension = DATASETS["SHREC17"][1]
    load_path = f"{DATASETS['SHREC17'][0]}/PatternDB/"
    save_path = f"Datasets/LRFPatches/SHREC17"

    # Iterate over meshes of the mesh dataset
    print(f"Generation of patches STARTED!")

    if to_extract == "all":
        mesh_to_extract = meshes
    else:
        mesh_to_extract = [meshes[i] for i in to_extract]

    for sample_id, sample in enumerate(mesh_to_extract):
        print(f"Generation of patches on sample {to_extract[sample_id]} STARTED!")
        for level, tup in sample.items():
            mesh_id = tup[0]
            label = tup[1]
            mesh = Mesh()
            mesh.load(f"{load_path}{mesh_id}{extension}")
            if not mesh.has_curvatures():
                mesh.computeCurvatures(radius=7)
            vertices_number = len(mesh.vertices)
            max_radius = np.max(configurations[:, 0]) / np.min(configurations[:, 1])
            boundary_vertices = mesh.getBoundaryVertices(neighbors_level=int(np.ceil(1.5 * max_radius)))
            # Under development uses a fixed seed points sequence
            rng = np.random.default_rng(22)
            seed_point_sequence = list(np.unique(rng.choice(range(vertices_number - 1), 4000, replace=False)))
            rng.shuffle(seed_point_sequence)
            for config in configurations:
                radius, rings, points = config
                # Generate N number of patches for a single mesh
                processed_patch = 1
                patches = []
                concentric_rings_list = []
                for seed_point in seed_point_sequence:
                    if processed_patch % (patch_per_mesh + 1) == 0:
                        break

                    if seed_point in boundary_vertices:
                        continue

                    concentric_rings = CSIRSv2(mesh, seed_point, radius, rings, points)
                    if not concentric_rings.firstValidRings(2):
                        continue

                    patch = SpiderPatchLRF(concentric_rings, mesh, seed_point, radius)
                    patches.append(patch)

                    concentric_rings_list.append(concentric_rings)
                    processed_patch += 1

                os.makedirs(f"{save_path}_R{radius}_RI{rings}_P{points}", exist_ok=True)
                os.makedirs(f"{save_path}_R{radius}_RI{rings}_P{points}/class_{str(label)}", exist_ok=True)
                os.makedirs(f"{save_path}_R{radius}_RI{rings}_P{points}/class_{str(label)}/id_{to_extract[sample_id]}", exist_ok=True)
                os.makedirs(f"{save_path}_R{radius}_RI{rings}_P{points}/class_{str(label)}/id_{to_extract[sample_id]}/resolution_{level}", exist_ok=True)
                with open(f"{save_path}_R{radius}_RI{rings}_P{points}/class_{str(label)}/id_{to_extract[sample_id]}/resolution_{level}/patches{mesh_id}.pkl", 'wb') as file:
                    pickle.dump(patches, file, protocol=-1)

                os.makedirs(f"{save_path.replace('SpiderPatches', 'ConcentricRings')}_R{radius}_RI{rings}_P{points}", exist_ok=True)
                os.makedirs(f"{save_path.replace('SpiderPatches', 'ConcentricRings')}_R{radius}_RI{rings}_P{points}/class_{str(label)}", exist_ok=True)
                os.makedirs(f"{save_path.replace('SpiderPatches', 'ConcentricRings')}_R{radius}_RI{rings}_P{points}/class_{str(label)}/id_{to_extract[sample_id]}", exist_ok=True)
                os.makedirs(f"{save_path.replace('SpiderPatches', 'ConcentricRings')}_R{radius}_RI{rings}_P{points}/class_{str(label)}/id_{to_extract[sample_id]}/resolution_{level}", exist_ok=True)
                with open(f"{save_path.replace('SpiderPatches', 'ConcentricRings')}_R{radius}_RI{rings}_P{points}/class_{str(label)}/id_{to_extract[sample_id]}/resolution_{level}/patches{mesh_id}.pkl", 'wb') as file:
                    pickle.dump(patches, file, protocol=-1)


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

    meshes = subdivide_for_mesh()
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


def getSHREC17MeshClasses():
    mesh_to_labels = {}
    iter_counter = 0
    with open(DATASETS["SHREC17"][0] + "/perm.txt", "r") as perm:
        for line in perm:
            mesh_to_labels[int(line.replace("\n", ""))] = floor(iter_counter / 48)
            iter_counter += 1
    return mesh_to_labels
