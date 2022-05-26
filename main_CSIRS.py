import Mesh
from CSIRS.CSIRS import *
from MeshCurvatures import getCurvatures
from SpiderPatch.Patch import Patch


def main():
    large_mesh = Mesh()
    large_mesh.load("C:/Users/Nicco/Desktop/CARTELLE/Uni/AssegnoDiRicerca/MeshGraphDataset/SHREC17/PatternDB/22.off")
    little_mesh = Mesh()
    little_mesh.load("C:/Users/Nicco/Desktop/CARTELLE/Uni/AssegnoDiRicerca/MeshGraphDataset/SHREC17/PatternDB/556.off")
    little_mesh.computeCurvatures()
    meshes = [little_mesh, large_mesh]

    for radius in range(7, 22, 5):
        for mesh in meshes:
            print(radius)
            prova = Patch(CSIRSv2(mesh, 4600, radius, 6, 4),mesh, 4600)
            prova.draw()
            mesh.draw_with_rings(CSIRSv2(mesh, 4600, radius, 20, 10))

if __name__ == "__main__":
    main()

