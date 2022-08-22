import itertools

from GeometricUtils import *
from Mesh import *


def CSI(mesh, radius, circle_center, circle_normal, circle_delta, center_face=-1, max_iter=5):
    """

    @param mesh: Mesh Object
    @param radius: the radius of the Circle
    @param circle_center: the center of the Circle
    @param circle_normal: the normal of the Circle
    @param circle_delta: the delta of the Circle
    @param center_face: can be an index of the face that contains the circle_center or -1 if circle_center is a vertex of a mesh_point
    @param max_iter: max iteration number
    @return: a  sequence (found, intersecting_points, intersecting_faces)
    """
    intersecting_points = np.empty((0, 3))
    intersecting_faces = np.empty(0, dtype=int)  # array of face indices
    prev_fac = np.empty(0, dtype=int)
    expand = False
    if center_face == -1:
        center_idx = int(np.where(np.all(mesh.vertices == circle_center, axis=1))[0][0])
        f_idx = mesh.vertex_faces[center_idx]
        neigh_facet = expandFacet(f_idx, mesh)
    else:
        neigh_facet = expandFacet(center_face, mesh)

    iteration_CSI = 0
    while intersecting_points.shape[0] < 2:
        if iteration_CSI >= max_iter:
            an_array = np.full((1, 3), nan)
            return False, an_array, an_array  # return arrays of NaN

        iteration_CSI += 1

        if expand:
            prev_fac = np.append(prev_fac, neigh_facet)
            neigh_facet = expandFacet(neigh_facet, mesh)
            neigh_facet = neigh_facet[np.logical_not(np.isin(neigh_facet, prev_fac))]

        if neigh_facet.shape[0] == 0:
            an_array = np.full((1, 3), nan)
            return False, an_array, an_array  # return arrays of NaN

        # Calcolo le intersezioni del cerchio
        for face_idx in neigh_facet:
            face = mesh.faces[face_idx]

            p1 = mesh.vertices[face[0]]
            p2 = mesh.vertices[face[1]]
            p3 = mesh.vertices[face[2]]
            face_normal = mesh.faces_normals()[face_idx]

            face_delta = np.dot(p1, face_normal)

            p = threePlaneIntersection(face_normal, face_delta, circle_normal, circle_delta)  # Mi serve a trovare uin punto qualsiasi appartenente ai due piani (x0,y0,z0) per generare la forma parametrica della retta L(t) (guarda articolo)
            # p = punto generico appartenente ai due piani (Piano della faccia della mesh e Piano definito dal cerchio)
            line = np.cross(face_normal, circle_normal)  # Definisco il vettore parallelo alla retta (prodotto vettoriale tra le normali ai piani)
            points = lineSphereIntersection(line, p, circle_center, radius)

            points = pointsInFace(points, [p1, p2, p3])

            if len(points) != 0:
                intersecting_points = np.vstack((intersecting_points, points))
                intersecting_faces = np.append(intersecting_faces, np.tile(face_idx, (points.shape[0], 1)))
        intersecting_points, intersecting_faces = cleanIntersectionCandidates(intersecting_points, intersecting_faces)
        expand = intersecting_points.shape[0] < 2
    return True, intersecting_points, intersecting_faces


def CSI_Arbitrary(mesh, radius, circle_center, circle_normal, circle_delta, center_face=-1, max_iter=10):
    """

    @param mesh: Mesh Object
    @param radius: the radius of the Circle
    @param circle_center: the center of the Circle
    @param circle_normal: the normal of the Circle
    @param circle_delta: the delta of the Circle
    @param center_face: can be an index of the face that contains the circle_center or -1 if circle_center is a vertex of a mesh_point
    @param max_iter: max iteration number
    @return: a  sequence (found, intersecting_points, intersecting_faces)
    """
    intersecting_points = np.empty((0, 3))
    intersecting_faces = np.empty(0, dtype=int)  # array of face indices
    prev_fac = np.empty(0, dtype=int)
    expand = False
    if center_face == -1:
        f_idx = faceOf3DPointv2(circle_center, mesh)
        neigh_facet = expandFacet(f_idx, mesh)
    else:
        neigh_facet = expandFacet(center_face, mesh)

    iteration_CSI = 0
    while intersecting_points.shape[0] < 2:
        if iteration_CSI >= max_iter:
            an_array = np.full((1, 3), nan)
            return False, an_array, an_array  # return arrays of NaN

        iteration_CSI += 1

        if expand:
            prev_fac = np.append(prev_fac, neigh_facet)
            neigh_facet = expandFacet(neigh_facet, mesh)
            neigh_facet = neigh_facet[np.logical_not(np.isin(neigh_facet, prev_fac))]

        if neigh_facet.shape[0] == 0:
            an_array = np.full((1, 3), nan)
            return False, an_array, an_array  # return arrays of NaN

        # Calcolo le intersezioni del cerchio
        for face_idx in neigh_facet:
            face = mesh.faces[face_idx]

            p1 = mesh.vertices[face[0]]
            p2 = mesh.vertices[face[1]]
            p3 = mesh.vertices[face[2]]
            face_normal = mesh.faces_normals()[face_idx]

            face_delta = np.dot(p1, face_normal)

            p = threePlaneIntersection(face_normal, face_delta, circle_normal, circle_delta)  # Mi serve a trovare un punto qualsiasi appartenente ai due piani (x0,y0,z0) per generare la forma parametrica della retta L(t) (guarda articolo)
            # p = punto generico appartenente ai due piani (Piano della faccia della mesh e Piano definito dal cerchio)
            line = np.cross(face_normal, circle_normal)  # Definisco il vettore parallelo alla retta (prodotto vettoriale tra le normali ai piani)
            points = lineSphereIntersection(line, p, circle_center, radius)

            points = pointsInFace(points, [p1, p2, p3])
            points = pointsInFacev2(points, [p1, p2, p3])

            if len(points) != 0:
                intersecting_points = np.vstack((intersecting_points, points))
                intersecting_faces = np.append(intersecting_faces, np.tile(face_idx, (points.shape[0], 1)))
        intersecting_points, intersecting_faces = cleanIntersectionCandidates(intersecting_points, intersecting_faces)
        expand = intersecting_points.shape[0] < 2
    return True, intersecting_points, intersecting_faces


def cleanIntersectionCandidates(candidates_v, candidates_faces):  # TODO È ottimizzabile? Per far si che CSI funzioni a dovere devo stare attento quando mi arrivano 2 punti molto vicini: a quel punto devo restituirne 1? e se fossero su facce diverse?
    """
    Removes unnecessary candidate points when there are more than two candidates.
    @param candidates_v: (Nx3) array of points
    @param candidates_faces: (N) array of face indices representing the faces containing the candidates
    @return:
    """

    if candidates_v.shape[0] > 2:
        # Approssimo alla decima cifra decimale
        candidates_v = approximate(candidates_v, 10)
        candidates_v, idx = np.unique(candidates_v, return_index=True, axis=0)
        candidates_faces = candidates_faces[idx]
        if candidates_v.shape[0] > 2:
            combinations = np.array(list(itertools.combinations(np.arange(candidates_v.shape[0]), 2)))
            distances = np.linalg.norm(candidates_v[combinations[:, 0]] - candidates_v[combinations[:, 1]], axis=1)
            idx = np.argmax(distances)
            idx = combinations[idx]
            candidates_v = candidates_v[idx]
            candidates_faces = candidates_faces[idx]

    if candidates_v.shape[0] == 2 and np.linalg.norm(candidates_v[0] - candidates_v[1]) < pow(10, -3):  # Se i punti trovati od ottenuti nell'if precedente sono due e la loro distanza è piccola restituisco la media tra i due punti
        candidates_v = np.mean((candidates_v[0], candidates_v[1]), axis=0).reshape((1, 3))
        candidates_faces = candidates_faces[0]
        # TODO Ripensare alla scelta della faccia: e se i due punti si trovassero su facce diverse? Il punto medio dove andrebbe a finire?

    return candidates_v, candidates_faces


def CSIv2(mesh, radius, circle_center, circle_normal, circle_delta, center_face=-1, max_iter=5):
    """

    @param mesh: Mesh Object
    @param radius: the radius of the Circle
    @param circle_center: the center of the Circle
    @param circle_normal: the normal of the Circle
    @param circle_delta: the delta of the Circle
    @param center_face: can be an index of the face that contains the circle_center or -1 if circle_center is a vertex of a mesh_point
    @param max_iter: max iteration number
    @return: a  sequence (found, intersecting_points, intersecting_faces)
    """
    intersecting_points = np.empty((0, 3))
    intersecting_faces = np.empty(0, dtype=int)  # array of face indices
    prev_fac = np.empty(0, dtype=int)
    expand = False
    if center_face == -1:
        center_idx = int(np.where(all(mesh.vertices == circle_center, axis=1))[0][0])
        f_idx = mesh.vertex_faces[center_idx]
        neigh_facet = expandFacetv2(f_idx, mesh)
    else:
        neigh_facet = expandFacetv2(center_face, mesh)

    iteration_CSI = 0
    while intersecting_points.shape[0] < 2:
        if iteration_CSI >= max_iter:
            an_array = np.full((1, 3), nan)
            return False, an_array, an_array  # return arrays of NaN

        iteration_CSI += 1

        if expand:
            prev_fac = np.append(prev_fac, neigh_facet)
            neigh_facet = expandFacetv2(neigh_facet, mesh)
            neigh_facet = neigh_facet[np.logical_not(np.isin(neigh_facet, prev_fac))]

        if neigh_facet.shape[0] == 0:
            an_array = np.full((1, 3), nan)
            return False, an_array, an_array  # return arrays of NaN

        # Calcolo le intersezioni del cerchio
        for face_idx in neigh_facet:
            face = mesh.faces[face_idx]

            p1 = mesh.vertices[face[0]]
            p2 = mesh.vertices[face[1]]
            p3 = mesh.vertices[face[2]]
            face_normal = mesh.face_normals[face_idx]

            face_delta = np.dot(p1, face_normal)

            p = threePlaneIntersection(face_normal, face_delta, circle_normal, circle_delta)  # Mi serve a trovare uin punto qualsiasi appartenente ai due piani (x0,y0,z0) per generare la forma parametrica della retta L(t) (guarda articolo)
            # p = punto generico appartenente ai due piani (Piano della faccia della mesh e Piano definito dal cerchio)
            line = np.cross(face_normal, circle_normal)  # Definisco il vettore parallelo alla retta (prodotto vettoriale tra le normali ai piani)
            points = lineSphereIntersection(line, p, circle_center, radius)

            points = pointsInFace(points, [p1, p2, p3])

            if len(points) != 0:
                intersecting_points = np.vstack((intersecting_points, points))
                intersecting_faces = np.append(intersecting_faces, np.tile(face_idx, (points.shape[0], 1)))
        intersecting_points, intersecting_faces = cleanIntersectionCandidates(intersecting_points, intersecting_faces)
        expand = intersecting_points.shape[0] < 2
    return True, intersecting_points, intersecting_faces


def approximate(values, decs=0):
    return np.trunc(values * 10 ** decs) / (10 ** decs)
