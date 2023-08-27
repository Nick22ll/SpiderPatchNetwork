from math import floor

DATASETS = {
    "SHREC17":
        {
            "general_path": "../MeshDataset/SHREC17/",
            "mesh_path": "../MeshDataset/SHREC17/meshes/",
            "perm_path": "../MeshDataset/SHREC17/perm.txt",
            "extension": ".off"
        },
    "SHREC20":
        {
            "general_path": "../MeshDataset/SHREC20/",
            "mesh_path": "../MeshDataset/SHREC20/models/",
            "perm_path": "../MeshDataset/SHREC20/sh2020_model_permutation.txt",
            "extension": ".ply"
        },
}


def readPermSHREC17(return_type="tuple"):
    meshes = []
    iter_counter = 0
    with open(DATASETS["SHREC17"]["perm_path"], "r") as perm_file:
        if return_type == "tuple":
            for line in perm_file:
                if iter_counter % 4 == 0:
                    meshes.append({})
                meshes[-1][f"level_{iter_counter % 4}"] = (int(line.replace("\n", "")), floor(iter_counter / 48))
                iter_counter += 1
        elif return_type == "list":
            for line in perm_file:
                meshes.append(int(line.replace("\n", "")))
                iter_counter += 1
    return meshes


def readPermSHREC20(return_type="tuple"):
    meshes = []
    with open(DATASETS["SHREC20"]["perm_path"], "r") as perm_file:
        if return_type == "tuple":
            for i, line in enumerate(perm_file):
                meshes.append({"level_0": (int(line.replace("\n", "")), floor(i / 20))})
        elif return_type == "list":
            for line in perm_file:
                meshes.append(int(line.replace("\n", "")))
        else:
            raise
    return meshes

# LEVEL_0 --> MASSIMA RISOLUZIONE
# LEVEL_2 --> MINIMA RISOLUZIONE
#
# LEVEL_2 < LEVEL_1 < LEVEL_3 < LEVEL_0
