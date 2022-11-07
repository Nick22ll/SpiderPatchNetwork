import multiprocessing
import os
import pickle

from Mesh.Mesh import Mesh
from SHREC_Utils import subdivide_for_mesh

DATASETS = {
    "SHREC17": ("../MeshDataset/SHREC17", ".off")
}


def generateMeshDataset(to_extract="all"):
    meshes = subdivide_for_mesh()
    extension = DATASETS["SHREC17"][1]
    load_path = f"{DATASETS['SHREC17'][0]}/PatternDB/"
    save_path = f"Datasets/Meshes/SHREC17"

    # Iterate over meshes of the mesh dataset
    print(f"Generation of meshes STARTED!")

    if to_extract == "all":
        mesh_to_extract = meshes
    else:
        mesh_to_extract = [meshes[i] for i in to_extract]

    for sample_id, sample in enumerate(mesh_to_extract):
        for level, tup in sample.items():
            mesh_id = tup[0]
            label = tup[1]
            mesh = Mesh()
            mesh.loadFromMeshFile(f"{load_path}{mesh_id}{extension}")
            if not mesh.has_curvatures():
                mesh.computeCurvatures(radius=0)
                mesh.computeCurvatures(radius=1)
                mesh.computeCurvatures(radius=2)
                mesh.computeCurvatures(radius=3)
                mesh.computeCurvatures(radius=4)

            os.makedirs(f"{save_path}", exist_ok=True)
            os.makedirs(f"{save_path}/class_{str(label)}", exist_ok=True)
            os.makedirs(f"{save_path}/class_{str(label)}/id_{to_extract[sample_id]}", exist_ok=True)
            os.makedirs(f"{save_path}/class_{str(label)}/id_{to_extract[sample_id]}/resolution_{level}", exist_ok=True)
            with open(f"{save_path}/class_{str(label)}/id_{to_extract[sample_id]}/resolution_{level}/mesh{mesh_id}.pkl", 'wb') as file:
                pickle.dump(mesh, file, protocol=-1)


def parallelGenerateMeshDataset(to_extract=None):
    if to_extract is None:
        to_extract = [i for i in range(180)]
    thread_num = 6
    pool = multiprocessing.Pool(processes=thread_num)
    mesh_for_thread = int(len(to_extract) / thread_num)
    pool.map(generateMeshDataset, [l for l in [to_extract[i * mesh_for_thread: (i * mesh_for_thread) + mesh_for_thread] for i in range(thread_num)]])
