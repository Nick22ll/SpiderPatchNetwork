import os
import pickle

from tqdm import tqdm

from Mesh import *
from math import floor

from SHREC_Utils import subdivide_for_mesh
from .Patch import *
from CSIRS.CSIRS import CSIRSv2

DATASETS = {
    "SHREC17": ("../MeshDataset/SHREC17", ".off")
}


def generatePatchDataset(mesh_dataset="SHREC17", save_path="", configurations=None, to_extract="all", patch_per_mesh=200, start_idx=None, stop_idx=None):
    """

    :param mesh_dataset:
    :param save_path:
    :param configurations:
    :param to_extract:
    :param patch_per_mesh:
    :return:
    """
    if configurations is None:
        configurations = np.array([[5, 4, 6], [7, 4, 8], [10, 6, 6]])

    if start_idx is None:
        start_idx = 0

    if mesh_dataset == "SHREC17":
        meshes = subdivide_for_mesh()
        extension = DATASETS["SHREC17"][1]
        load_path = f"{DATASETS['SHREC17'][0]}/PatternDB/"
        save_path = save_path + f"Datasets/Patches/SHREC17"
    else:
        raise Exception("Dataset NOT Found!")

    # Iterate over meshes of the mesh dataset
    print(f"Generation of {mesh_dataset} patches STARTED!")
    if to_extract == "all":
        mesh_to_extract = meshes
    else:
        mesh_to_extract = [meshes[i] for i in to_extract]

    if stop_idx is None:
        stop_idx = len(mesh_to_extract)

    for mesh_type in tqdm(mesh_to_extract[start_idx:stop_idx], position=0, leave=True, desc=f"Calculating Mesh: ", colour="green", ncols=120):
        for level, tup in mesh_type.items():
            mesh_id = tup[0]
            label = tup[1]
            mesh = Mesh()
            mesh.load(f"{load_path}{mesh_id}{extension}")
            if not mesh.has_curvatures():
                mesh.computeCurvatures()
            vertices_number = len(mesh.vertices)
            max_radius = np.max(configurations[:, 0]) / np.min(configurations[:, 1])
            boundary_vertices = mesh.getBoundaryVertices(neighbors_level=int(np.ceil(1.5 * max_radius)))
            # Under development uses a fixed seed points sequence
            np.random.seed(22)
            seed_point_sequence = np.unique(np.random.randint(0, vertices_number - 1, size=2000))
            for config in tqdm(configurations, position=0, leave=True, desc=f"Mesh ID {mesh_id}: ", colour="white", ncols=80):
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

                    concentric_rings = CSIRSv2(mesh, seed_point, radius, points, rings)
                    if not concentric_rings.first_valid_rings(2):
                        continue

                    patch = Patch(concentric_rings, mesh, seed_point)
                    patches.append(patch)
                    concentric_rings_list.append(concentric_rings)
                    processed_patch += 1
                os.makedirs(f"{save_path}_R{radius}_RI{rings}_P{points}", exist_ok=True)
                os.makedirs(f"{save_path}_R{radius}_RI{rings}_P{points}/{level}", exist_ok=True)
                os.makedirs(f"{save_path}_R{radius}_RI{rings}_P{points}/{level}/{str(label)}", exist_ok=True)
                with open(f"{save_path}_R{radius}_RI{rings}_P{points}/{level}/{str(label)}/patches{mesh_id}.pkl", 'wb') as file:
                    pickle.dump(patches, file)
                os.makedirs(f"{save_path.replace('Patches', 'ConcentricRings')}_R{radius}_RI{rings}_P{points}", exist_ok=True)
                os.makedirs(f"{save_path.replace('Patches', 'ConcentricRings')}_R{radius}_RI{rings}_P{points}/{level}/", exist_ok=True)
                os.makedirs(f"{save_path.replace('Patches', 'ConcentricRings')}_R{radius}_RI{rings}_P{points}/{level}/{str(label)}", exist_ok=True)
                with open(f"{save_path.replace('Patches', 'ConcentricRings')}_R{radius}_RI{rings}_P{points}/{level}/{str(label)}/concentric_rings{mesh_id}.pkl", 'wb') as file:
                    pickle.dump(concentric_rings_list, file)


def getSHREC17MeshClasses():
    mesh_to_labels = {}
    iter_counter = 0
    with open(DATASETS["SHREC17"][0] + "/perm.txt", "r") as perm:
        for line in perm:
            mesh_to_labels[int(line.replace("\n", ""))] = floor(iter_counter / 48)
            iter_counter += 1
    return mesh_to_labels
