import multiprocessing
import os
import pickle

import numpy as np
from tqdm import tqdm

from Mesh.Mesh import Mesh
from SHREC_Utils import readPermSHREC17, DATASETS, readPermSHREC20


def generateMeshDataset(dataset_name, from_mesh_dataset="SHREC17", samples_to_extract=None, normalize=False, perturbe=(0, 1), resolution_level=None, progress_bar_position=0):
    if from_mesh_dataset == "SHREC17":
        meshes = readPermSHREC17()
    elif from_mesh_dataset == "SHREC20":
        meshes = readPermSHREC20()
    else:
        raise "Name not found!"

    extension = DATASETS[from_mesh_dataset]["extension"]
    load_path = f"{DATASETS[from_mesh_dataset]['mesh_path']}"
    if normalize:
        save_path = f"Datasets/NormalizedMeshes/{dataset_name}"
    else:
        save_path = f"Datasets/Meshes/{dataset_name}"

    # Iterate over meshes of the mesh dataset

    if samples_to_extract is None:
        samples_to_extract = [i for i in range(len(meshes))]
        mesh_to_extract = meshes
    else:
        mesh_to_extract = [meshes[i] for i in samples_to_extract]

    for sample_id, sample in enumerate(tqdm(mesh_to_extract, desc="Generating Meshes:", position=progress_bar_position)):
        for level, tup in sample.items():
            if resolution_level is not None and level != resolution_level:
                continue
            mesh_id = tup[0]
            label = tup[1]
            mesh = Mesh()
            mesh.loadFromMeshFile(f"{load_path}{mesh_id}{extension}", normalize, perturbe)

            if not mesh.has_curvatures():
                mesh.computeCurvatures(radius=0)
                mesh.computeCurvatures(radius=1)
                mesh.computeCurvatures(radius=2)
                mesh.computeCurvatures(radius=3)
                mesh.computeCurvatures(radius=4)

            os.makedirs(f"{save_path}", exist_ok=True)
            os.makedirs(f"{save_path}/class_{str(label)}", exist_ok=True)
            prev = f"{save_path}/class_{str(label)}"
            if from_mesh_dataset == "SHREC17":
                os.makedirs(f"{prev}/id_{samples_to_extract[sample_id]}", exist_ok=True)
                os.makedirs(f"{prev}/id_{samples_to_extract[sample_id]}/resolution_{level}", exist_ok=True)
                prev = f"{prev}/id_{samples_to_extract[sample_id]}/resolution_{level}"
            with open(f"{prev}/mesh{mesh_id}.pkl", 'wb') as file:
                pickle.dump(mesh, file, protocol=-1)


def parallelGenerateMeshDataset(dataset_name, from_mesh_dataset="SHREC17", samples_to_extract=None, normalize=False, perturbe=(0, 1), resolution_level=None, num_threads=10):
    if samples_to_extract is None:
        if from_mesh_dataset == "SHREC17":
            samples_to_extract = [i for i in range(len(readPermSHREC17()))]
        elif from_mesh_dataset == "SHREC20":
            samples_to_extract = [i for i in range(len(readPermSHREC20()))]
        else:
            raise
    mesh_for_thread = int(len(samples_to_extract) / num_threads) + 1
    pool = multiprocessing.Pool(processes=num_threads)
    pool.starmap(generateMeshDataset, [(dataset_name, from_mesh_dataset, samples_to_extract[i * mesh_for_thread: (i * mesh_for_thread) + mesh_for_thread], normalize, perturbe, resolution_level, i) for i in range(num_threads)])
