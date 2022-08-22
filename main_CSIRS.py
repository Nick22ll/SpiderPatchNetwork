import Mesh
from CSIRS.CSIRS import *
from MeshCurvatures import getCurvatures
from SpiderPatch.Patch import Patch


def main():
    large_mesh = Mesh()
    large_mesh.load("U:\AssegnoDiRicerca\MeshDataset\SHREC17\PatternDB/22.off")
    little_mesh = Mesh()
    little_mesh.load("U:\AssegnoDiRicerca\MeshDataset\SHREC17\PatternDB/556.off")
    # little_mesh.computeCurvatures(7)
    meshes = [little_mesh, large_mesh]

    for radius in range(7, 22, 5):
        for mesh in meshes:
            print(radius)
            # prova = CSIRSv2(mesh, 4600, radius, 6, 4)
            prova = CSIRSv2Arbitrary(mesh, np.array([22, 10, 67]), radius, 6, 4)
            a = 0
            # prova = Patch(CSIRSv2(mesh, 4600, radius, 6, 4), mesh, 4600)
            prova.draw()
            # mesh.draw_with_rings(CSIRSv2(mesh, 4600, radius, 20, 10))


if __name__ == "__main__":
    main()
