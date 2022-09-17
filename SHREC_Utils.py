from math import floor


def read_perm(path="../MeshDataset/SHREC17/perm.txt"):
    return open(path, "r")


def subdivide_for_mesh(return_type="tuple"):
    meshes = []
    iter_counter = 0
    if return_type == "tuple":
        for line in read_perm():
            if iter_counter % 4 == 0:
                meshes.append({})
            meshes[-1][f"level_{iter_counter % 4}"] = (int(line.replace("\n", "")), floor(iter_counter / 48))
            iter_counter += 1
    elif return_type == "list":
        for line in read_perm():
            meshes.append(int(line.replace("\n", "")))
            iter_counter += 1
    return meshes


# LEVEL_0 --> MASSIMA RISOLUZIONE
# LEVEL_2 --> MINIMA RISOLUZIONE
#
# LEVEL_2 < LEVEL_1 < LEVEL_3 < LEVEL_0