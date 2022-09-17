import pickle
import random

from CSIRS.CSIRS import CSIRS, CSIRSv2Arbitrary, CSIRSv2, CSIRSv2Spiral
from CSIRS.Ring import ConcentricRings
from GeometricUtils import LRF_normals, LRF
from Mesh import *
from SHREC_Utils import subdivide_for_mesh
from mayavi.mlab import *

from SpiderPatch.SpiderPatch import SpiderPatch
from SpiderPatch.SuperPatch import SuperPatch


def main():
    with open("U:\AssegnoDiRicerca\PythonProject\Executables\Mesh\egyptFaceDenseLC.pkl", "rb") as mesh_file:
        mesh = pickle.load(mesh_file)
    conc, lrf = CSIRSv2Spiral(mesh, 49989, 1, 4, 6, 2)
    mesh.drawWithConcRings(conc)
    spider = SpiderPatch(conc, mesh, 49989)
    mesh.drawWithSpiderPatches([spider])

    # mesh_id = list(sample.items())[0][1][0]
    # level_0 = Mesh()
    # level_0.load(f"U:\AssegnoDiRicerca\MeshDataset\SHREC17\PatternDB\{mesh_id}.off")
    # level_0.draw()
    #
    #
    # vertices = np.array(level_0.vertices)

    # mesh_id = sample["level_3"][0]
    # level_3 = Mesh()
    # level_3.load(f"U:\AssegnoDiRicerca\MeshDataset\SHREC17\PatternDB\{mesh_id}.off")
    # level_0.drawWithMesh(level_3)
    # vertices = np.vstack((vertices, np.array(level_3.vertices).round(1)))
    # vertices = np.delete(vertices, np.unique(vertices, return_index=True, axis=0)[1], axis=0)
    #
    # mesh_id = sample["level_1"][0]
    # level_1 = Mesh()
    # level_1.load(f"U:\AssegnoDiRicerca\MeshDataset\SHREC17\PatternDB\{mesh_id}.off")
    # level_0.drawWithMesh(level_1)
    # vertices = np.vstack((vertices, np.array(level_1.vertices).round(1)))
    # vertices = np.delete(vertices, np.unique(vertices, return_index=True, axis=0)[1], axis=0)

    # mesh_id = sample["level_2"][0]
    # level_2 = Mesh()
    # level_2.load(f"U:\AssegnoDiRicerca\MeshDataset\SHREC17\PatternDB\{mesh_id}.off")
    #
    # triangular_mesh(level_2.vertices[:, 0], level_2.vertices[:, 1], level_2.vertices[:, 2], level_2.faces)
    #
    # old_normals2 = np.empty((0, 3))
    #
    # random.seed(22)
    # uniques = np.unique(random.sample(range(len(level_2.vertices)), 10))
    # seed_point = level_2.vertices[uniques[0]]
    # lrf = LRF(level_2, seed_point, 5)
    # V0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # V1 = np.array([lrf[0], lrf[1], lrf[2]])
    # for idx in uniques:
    #     old_normals2 = np.vstack((old_normals2, level_2.vertex_normals[idx]))
    # # new_normals2 = np.linalg.solve(V1, V0).dot(old_normals2)
    # new_normals2 = old_normals2.dot(np.linalg.solve(V1, V0).T)
    #
    # N = np.empty((0, 3))
    # P = np.empty((0, 3))
    # new_normals = np.empty((0, 3))
    # old_normals = np.empty((0, 3))
    # scalars = np.empty(0)
    # random.seed(22)
    # seed_point = level_2.vertices[uniques[0]]
    # lrf = LRF(level_2, seed_point, 5)
    # V0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # V1 = np.array([lrf[0], lrf[1], lrf[2]])
    # for idx in uniques:
    #     new_normals = np.vstack((new_normals, np.linalg.solve(V1, V0).dot(level_2.vertex_normals[idx])))
    #     old_normals = np.vstack((old_normals, level_2.vertex_normals[idx]))
    #     N = np.vstack((N, [lrf[0], lrf[1], lrf[2]]))
    #     P = np.vstack((P, np.tile(seed_point, (3, 1))))
    #     scalars = np.hstack((scalars, [3, 2, 1]))
    #
    # x = P[:, 0]
    # y = P[:, 1]
    # z = P[:, 2]
    # u = N[:, 0]
    # v = N[:, 1]
    # w = N[:, 2]
    # quiver3d(x, y, z, u, v, w, line_width=3, scalars=scalars, scale_mode="vector")
    #
    # x = P[[i for i in range(0, len(P), 3)], 0]
    # y = P[[i for i in range(0, len(P), 3)], 1]
    # z = P[[i for i in range(0, len(P), 3)], 2]
    # u = old_normals[:, 0]
    # v = old_normals[:, 1]
    # w = old_normals[:, 2]
    # quiver3d(x, y, z, u, v, w, line_width=3, color=(0, 1, 0))
    #
    # u = new_normals[:, 0]
    # v = new_normals[:, 1]
    # w = new_normals[:, 2]
    # quiver3d(x, y, z, u, v, w, line_width=3, color=(0, 0, 1))
    # u = new_normals2[:, 0]
    # v = new_normals2[:, 1]
    # w = new_normals2[:, 2]
    # quiver3d(x, y, z, u, v, w, line_width=3, color=(0, 1, 1))
    # show()

    # level_0.drawWithMesh(level_2)
    # level_0.drawWithPointCloud(level_2.vertices)
    # vertices = np.vstack((vertices, np.array(level_2.vertices)))
    # vertices = np.delete(vertices, np.unique(vertices, return_index=True, axis=0)[1], axis=0)
    # level_0.drawWithPointCloud(vertices)
    # a = CSIRSv2Arbitrary(level_0, level_2.vertices[150], 10, 6, 6)
    # level_0.draw_with_rings(a)

    return


if __name__ == "__main__":
    main()
