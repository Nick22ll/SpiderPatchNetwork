from GeometricUtils import *
from sklearn.metrics import pairwise_distances


def fixFacetInAll(vertex, facetIn, new_facet, length):
    facet_add = np.empty((0, 3), dtype=int)
    if facetIn.shape[0] != 0:
        fixed, facet_fixed = fixFacetIn(vertex, facetIn[-1], new_facet[0], length)
        while fixed:
            if facetIn.shape[0] > 1:
                facetIn = facetIn[:-1]
                facet_add = np.vstack((facet_add, facet_fixed))
                fixed, facet_fixed = fixFacetIn(vertex, facetIn[-1], facet_fixed, length)
            else:
                facetIn = np.empty((0, 3), dtype=int)
                facet_add = np.vstack((facet_add, facet_fixed))
                fixed = False
    if facet_add.shape[0] == 0:
        facetIn = np.vstack((facetIn, new_facet[0]))
    else:
        facetIn = np.vstack((facetIn, facet_add[-1]))

    if new_facet.shape[0] > 1:
        facetIn, facet_add2 = fixFacetInAll(vertex, facetIn, new_facet[1].reshape(-1, 3), length)
        facet_add = np.vstack((facet_add, facet_add2))

    return facetIn, facet_add


def fixFacetIn(vertex, facet1, facet2, length):
    fixed = False
    new_facet = np.empty((0, 3), dtype=int)
    if facet1[2] == facet2[1] and facet1[0] != facet2[0]:
        edge1 = vertex[facet1[2]] - vertex[facet1[1]]
        edge2 = vertex[facet2[1]] - vertex[facet2[2]]
        edge1 = edge1 / norm(edge1)
        edge2 = edge2 / norm(edge2)

        angle = getAngleBetweenNormals(edge1, edge2)
        if angle <= 105:
            p11 = vertex[facet1[0]]
            p12 = vertex[facet1[1]]
            p13 = vertex[facet1[2]]
            p21 = vertex[facet2[0]]
            p22 = vertex[facet2[1]]
            p23 = vertex[facet2[2]]
            normal1 = np.cross(p12 - p11, p13 - p11)
            normal2 = np.cross(p22 - p21, p23 - p21)
            normal1 = normal1 / norm(normal1)
            normal2 = normal2 / norm(normal2)
            angle = getAngleBetweenNormals(normal1, normal2)
            if angle < 45 or pairwise_distances(p12, p23, metric="sqeuclidean") < pow(length * 1.2, 2):
                new_facet = np.array([facet1[2], facet1[1], facet2[2]], dtype=int)
                fixed = True
            else:
                print("facetIn NOT fixed!")
    return fixed, new_facet
