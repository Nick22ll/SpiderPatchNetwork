from CSIOR.CSIOR import *
from Mesh import *

def main():
    mesh = Mesh()
    mesh.load("towel1_2_a_s10000a.off")
    print(mesh.averageEdgeLength())
    resampled_mesh, rings = CSIOR(mesh, edge_length=2.5)
    resampled_mesh.draw()
    return


if __name__ == "__main__":
    main()
