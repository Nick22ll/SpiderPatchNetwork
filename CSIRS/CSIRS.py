from sklearn.decomposition import PCA

from GeometricUtils import LRF
from CSI.CSI import *
from .Ring import *

from sklearn.metrics import pairwise_distances


def CSIRS(mesh, seed_point, radius, n_rings, n_points):
    """
    CSIRS (Circle-Surface Intersection): Generates concentric rings of points on the surface (vertex, face)
    @param mesh: a Mesh object
    @param seed_point: index of a seed point
    @param radius: the max extension of the rings
    @param n_points: define the number of points per ring
    @param n_rings: number of concentric rings

    @return:
    """
    radius = radius / n_rings

    ###### INITIALIZATION PHASE  #########
    # Creates the first ring
    circle_centers = np.tile(mesh.vertices[seed_point], (int(n_points / 2), 1))  # Only pair n_points # Array of vertices indices
    circle_normals = generateCircleNormals(mesh.vertex_normals[seed_point], int(n_points / 2))  # Only pair n_points
    circle_deltas = np.sum(circle_centers * circle_normals, axis=1)  # Substitute of circle_deltas = np.dot(circle_centers, circle_normals)

    seed_face_idx = mesh.vertex_faces[seed_point][0]
    rings = ConcentricRings(mesh.vertices[seed_point], seed_face_idx)
    intersecting_points = np.zeros((n_points, 3))
    intersecting_faces = - np.ones(n_points, dtype=int)
    candidates_v, candidates_faces = [], []

    for r in range(n_rings):
        for center_idx in range(circle_centers.shape[0]):
            if np.isnan(circle_centers[center_idx]).any():  # controllo che il centro del punto dell'anello precedente esista (che sia diverso da [-1, -1, -1])
                found = False
            else:
                found, candidates_v, candidates_faces = CSI(mesh, radius, circle_centers[center_idx], circle_normals[center_idx % int(n_points / 2)], circle_deltas[center_idx], intersecting_faces[center_idx])

            if found:
                # Assegno a ogni anello i punti d'intersezione
                if r == 0:  # Primo anello
                    if center_idx == 0:  # Primo cerchio
                        intersecting_points[center_idx] = candidates_v[0]
                        intersecting_faces[center_idx] = candidates_faces[0]
                        intersecting_points[int(n_points / 2)] = candidates_v[1]
                        intersecting_faces[int(n_points / 2)] = candidates_faces[1]
                    else:
                        idx = np.argmin(pairwise_distances(candidates_v, intersecting_points[center_idx - 1].reshape((1, 3)), metric="sqeuclidean",
                                                           force_all_finite="allow-nan"))  # Prendo il punto intersecante più vicino al punto precedente(nella prima metà dell'array) in modo da costruire gli array in modo ordinato
                        idx = (np.arange(2) == idx)
                        intersecting_points[center_idx] = candidates_v[idx]
                        intersecting_faces[center_idx] = candidates_faces[idx]
                        intersecting_points[center_idx + int(n_points / 2)] = candidates_v[np.logical_not(idx)]
                        intersecting_faces[center_idx + int(n_points / 2)] = candidates_faces[np.logical_not(idx)]
                else:  # Per ogni anello che non sia il primo prendo il punto d'intersezione più lontano dal punto dell'iterazione precedente
                    if r <= 1:
                        idx = np.argmax(pairwise_distances(candidates_v, mesh.vertices[seed_point].reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))
                    else:
                        idx = np.argmax(pairwise_distances(candidates_v, rings.rings[r - 2].points[center_idx].reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))
                    intersecting_points[center_idx] = candidates_v[idx]
                    intersecting_faces[center_idx] = candidates_faces[idx]
            else:  # se non trovo dei punti d'intersezione non fermo l'algoritmo ma do un valore particolare al punto
                if r == 0:
                    intersecting_points[center_idx] = np.full(3, np.nan)
                    intersecting_faces[center_idx] = -1
                    intersecting_points[center_idx + int(n_points / 2)] = np.full(3, np.nan)
                    intersecting_faces[center_idx + int(n_points / 2)] = -1
                else:
                    intersecting_points[center_idx] = np.full(3, np.nan)
                    intersecting_faces[center_idx] = -1

        circle_centers = np.array(intersecting_points)
        circle_normals, circle_deltas = getCircles(intersecting_points)
        rings.addRing(intersecting_points, intersecting_faces)
    return rings


def CSIRSv2(mesh, seed_point, radius, n_rings, n_points):
    """
    CSIRS (Circle-Surface Intersection): Generates concentric rings of points on the surface (vertex, face)
    @param mesh: a Mesh object
    @param seed_point: index of a seed point
    @param radius: the max extension of the rings
    @param n_points: define the number of points per ring
    @param n_rings: number of concentric rings

    @return:
    """
    radius = radius / n_rings

    ###### INITIALIZATION PHASE  #########
    # Creates the first ring
    circle_centers = np.tile(mesh.vertices[seed_point], (n_points, 1))
    circle_normals = generateCircleNormals(mesh.vertex_normals[seed_point], n_points, on360=True)
    circle_deltas = np.sum(circle_centers * circle_normals, axis=1)  # Substitute of circle_deltas = np.dot(circle_centers, circle_normals)

    seed_face_idx = mesh.vertex_faces[seed_point][0]
    rings = ConcentricRings(mesh.vertices[seed_point], seed_face_idx)
    intersecting_points = np.zeros((n_points, 3))
    intersecting_faces = np.tile(seed_face_idx, (n_points, 1))
    candidates_v, candidates_faces = [], []  # per evitare i warnings

    for r in range(n_rings):
        for center_idx in range(circle_centers.shape[0]):
            if np.isnan(circle_centers[center_idx]).any():  # controllo che il centro del punto dell'anello precedente esista (che sia diverso da [nan, nan, nan])
                found = False
            else:
                found, candidates_v, candidates_faces = CSI(mesh, radius, circle_centers[center_idx], circle_normals[center_idx], circle_deltas[center_idx], intersecting_faces[center_idx])

            if found:
                # Assegno a ogni anello i punti d'intersezione
                if r == 0:  # Primo anello
                    if center_idx == 0:  # Primo cerchio
                        intersecting_points[center_idx] = candidates_v[0]
                        intersecting_faces[center_idx] = candidates_faces[0]
                    else:
                        idx = np.argmin(pairwise_distances(candidates_v, intersecting_points[center_idx - 1].reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))  # Prendo il punto intersecante più vicino al punto precedente in modo da costruire gli array in modo ordinato
                        intersecting_points[center_idx] = candidates_v[idx]
                        intersecting_faces[center_idx] = candidates_faces[idx]
                else:  # Per ogni anello che non sia il primo prendo il punto d'intersezione più lontano dal punto dell'iterazione precedente
                    if r <= 1:
                        idx = np.argmax(pairwise_distances(candidates_v, mesh.vertices[seed_point].reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))
                    else:
                        idx = np.argmax(pairwise_distances(candidates_v, rings.rings[r - 2].points[center_idx].reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))
                    intersecting_points[center_idx] = candidates_v[idx]
                    intersecting_faces[center_idx] = candidates_faces[idx]
            else:  # se non trovo dei punti d'intersezione non fermo l'algoritmo ma do un valore particolare al punto
                intersecting_points[center_idx] = np.full(3, np.nan)
                intersecting_faces[center_idx] = -1

        circle_centers = np.array(intersecting_points)
        circle_normals, circle_deltas = getCircles(circle_centers)  # circle_deltas = np.sum(circle_centers * circle_normals, axis=1)
        rings.addRing(intersecting_points, intersecting_faces)
    return rings


def CSIRSv2Spiral(mesh, seed_point, radius, n_rings, n_points):
    """
    CSIRS (Circle-Surface Intersection): Generates concentric rings of points on the surface (vertex, face)
    @param mesh: a Mesh object
    @param seed_point: index of a seed point
    @param radius: the max extension of the rings
    @param n_points: define the number of points per ring
    @param n_rings: number of concentric rings

    @return:
    """

    lrf = LRF(mesh, mesh.vertices[seed_point], radius)
    radius = radius / n_rings
    lrf = alignZAxis(lrf[0], lrf[2], lrf[1], mesh.vertex_normals[seed_point])
    lrf[1], lrf[2] = lrf[2], lrf[1]

    ###### INITIALIZATION PHASE  #########
    # Creates the first ring
    circle_centers = np.tile(mesh.vertices[seed_point], (n_points, 1))
    circle_normals = generateCircleNormalsv2(lrf, n_points, on360=True)
    circle_deltas = np.sum(circle_centers * circle_normals, axis=1)  # Substitute of circle_deltas = np.dot(circle_centers, circle_normals)

    seed_face_idx = mesh.vertex_faces[seed_point][0]
    rings = ConcentricRings(mesh.vertices[seed_point], seed_face_idx)
    intersecting_points = np.zeros((n_points, 3))
    intersecting_faces = np.tile(seed_face_idx, (n_points, 1))
    candidates_v, candidates_faces = [], []  # per evitare i warnings

    for r in range(n_rings):
        for center_idx in range(circle_centers.shape[0]):
            if np.isnan(circle_centers[center_idx]).any():  # controllo che il centro del punto dell'anello precedente esista (che sia diverso da [nan, nan, nan])
                found = False
            else:
                found, candidates_v, candidates_faces = CSI(mesh, radius, circle_centers[center_idx], circle_normals[center_idx], circle_deltas[center_idx], intersecting_faces[center_idx])

            if found:
                # Assegno a ogni anello i punti d'intersezione
                if r == 0:  # Primo anello
                    if center_idx == 0:  # Primo cerchio
                        edges_directions = np.tile(circle_centers[center_idx], (2, 1)) - candidates_v
                        edges_directions /= np.linalg.norm(edges_directions, axis=0)
                        dot_products = np.sum(np.tile(lrf[2], (2, 1)) * edges_directions, axis=1)  # Substitute of np.dot(lrf[2], edges_directions)
                        max_parallel_idx = np.argmax(dot_products)
                        intersecting_points[center_idx] = candidates_v[max_parallel_idx]
                        intersecting_faces[center_idx] = candidates_faces[max_parallel_idx]
                    else:
                        idx = np.argmin(pairwise_distances(candidates_v, intersecting_points[center_idx - 1].reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))  # Prendo il punto intersecante più vicino al punto precedente in modo da costruire gli array in modo ordinato
                        intersecting_points[center_idx] = candidates_v[idx]
                        intersecting_faces[center_idx] = candidates_faces[idx]
                else:  # Per ogni anello che non sia il primo prendo il punto d'intersezione più lontano dal punto dell'iterazione precedente
                    if r <= 1:
                        idx = np.argmax(pairwise_distances(candidates_v, mesh.vertices[seed_point].reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))
                    else:
                        idx = np.argmax(pairwise_distances(candidates_v, rings.rings[r - 2].points[center_idx].reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))
                    intersecting_points[center_idx] = candidates_v[idx]
                    intersecting_faces[center_idx] = candidates_faces[idx]
            else:  # se non trovo dei punti d'intersezione non fermo l'algoritmo ma do un valore particolare al punto
                intersecting_points[center_idx] = np.full(3, np.nan)
                intersecting_faces[center_idx] = -1

        circle_centers = np.array(intersecting_points)
        circle_normals, circle_deltas = getCircles(circle_centers)  # circle_deltas = np.sum(circle_centers * circle_normals, axis=1)
        rings.addRing(intersecting_points, intersecting_faces)
    rings.lrf = lrf
    return rings


def SpiralCSIRSArbitrary(mesh, seed_point, radius, n_rings, n_points, intersec_direction=None, inFace=None):
    """
    CSIRS (Circle-Surface Intersection): Generates concentric concentricRings of points on the surface (vertex, face)
    @param mesh: a Mesh object
    @param seed_point: (1x3) array
    @param radius: the max extension of the concentricRings
    @param n_points: define the number of points per ring
    @param n_rings: number of concentric concentricRings
    @param intersec_direction: (1x3) array, define the direction to generate the seed_point on the mesh surface

    @return:
    """

    radius = radius / n_rings
    if inFace is None:
        isInFace = False
        if intersec_direction is None:
            # Calculate the nearest mesh vertex
            closest_v_idx = np.argmin(pairwise_distances(mesh.vertices, seed_point.reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))  # Prendo il punto intersecante più vicino al punto precedente in modo da costruire gli array in modo ordinato
            # Select the adjacent faces
            faces_idx = mesh.vertex_faces[closest_v_idx]
            # Search for the intersection
            for f_idx in faces_idx:
                intersection_point = intersectLinePlane(seed_point, mesh.face_normals[f_idx], mesh.vertices[closest_v_idx], mesh.face_normals[f_idx])
                isInFace = pointInFace(intersection_point, mesh.vertices[mesh.faces[f_idx]])
                if isInFace:
                    seed_face_idx = f_idx
                    break

            if not isInFace:
                # Calculate the direction to intersect the various mesh resolutions surfaces
                pca = PCA(n_components=3)
                pca.fit(mesh.vertices)
                intersec_direction = pca.components_[2]

                for f_idx in range(len(mesh.faces)):
                    intersection_point = intersectLinePlane(seed_point, intersec_direction, mesh.vertices[mesh.faces[f_idx][0]], mesh.face_normals[f_idx])
                    isInFace = pointInFace(intersection_point, mesh.vertices[mesh.faces[f_idx]])
                    if isInFace:
                        seed_face_idx = f_idx
                        break
        else:
            for f_idx in range(len(mesh.faces)):
                intersection_point = intersectLinePlane(seed_point, intersec_direction, mesh.vertices[mesh.faces[f_idx][0]], mesh.face_normals[f_idx])
                isInFace = pointInFace(intersection_point, mesh.vertices[mesh.faces[f_idx]])
                if isInFace:
                    seed_face_idx = f_idx
                    break

        if not isInFace:
            return None
    else:
        seed_face_idx = inFace
        intersection_point = seed_point

    ###### INITIALIZATION PHASE  #########
    seed_point = intersection_point
    radius = radius / n_rings
    lrf = LRF(mesh, seed_point, radius)

    ###### INITIALIZATION PHASE  #########
    # Creates the first ring
    circle_centers = np.tile(seed_point, (n_points, 1))
    circle_normals = generateCircleNormalsv2(lrf, n_points, on360=True)
    circle_deltas = np.sum(circle_centers * circle_normals, axis=1)  # Substitute of circle_deltas = np.dot(circle_centers, circle_normals)

    rings = ConcentricRings(seed_point, seed_face_idx)
    intersecting_points = np.zeros((n_points, 3))
    intersecting_faces = np.tile(seed_face_idx, (n_points, 1))
    candidates_v, candidates_faces = [], []  # per evitare i warnings

    for r in range(n_rings):
        for center_idx in range(circle_centers.shape[0]):
            if np.isnan(circle_centers[center_idx]).any():  # controllo che il centro del punto dell'anello precedente esista (che sia diverso da [nan, nan, nan])
                found = False
            else:
                found, candidates_v, candidates_faces = CSI(mesh, radius, circle_centers[center_idx], circle_normals[center_idx], circle_deltas[center_idx], intersecting_faces[center_idx])

            if found:
                # Assegno a ogni anello i punti d'intersezione
                if r == 0:  # Primo anello
                    if center_idx == 0:  # Primo cerchio
                        edges_directions = np.tile(circle_centers[center_idx], (2, 1)) - candidates_v
                        edges_directions /= np.linalg.norm(edges_directions, axis=0)
                        dot_products = np.sum(np.tile(lrf[2], (2, 1)) * edges_directions, axis=1)  # Substitute of np.dot(lrf[2], edges_directions)
                        max_parallel_idx = np.argmax(dot_products)
                        intersecting_points[center_idx] = candidates_v[max_parallel_idx]
                        intersecting_faces[center_idx] = candidates_faces[max_parallel_idx]
                    else:
                        idx = np.argmin(pairwise_distances(candidates_v, intersecting_points[center_idx - 1].reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))  # Prendo il punto intersecante più vicino al punto precedente in modo da costruire gli array in modo ordinato
                        intersecting_points[center_idx] = candidates_v[idx]
                        intersecting_faces[center_idx] = candidates_faces[idx]
                else:  # Per ogni anello che non sia il primo prendo il punto d'intersezione più lontano dal punto dell'iterazione precedente
                    if r <= 1:
                        idx = np.argmax(pairwise_distances(candidates_v, seed_point.reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))
                    else:
                        idx = np.argmax(pairwise_distances(candidates_v, rings.rings[r - 2].points[center_idx].reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))
                    intersecting_points[center_idx] = candidates_v[idx]
                    intersecting_faces[center_idx] = candidates_faces[idx]
            else:  # se non trovo dei punti d'intersezione non fermo l'algoritmo ma do un valore particolare al punto
                intersecting_points[center_idx] = np.full(3, np.nan)
                intersecting_faces[center_idx] = -1

        circle_centers = np.array(intersecting_points)
        circle_normals, circle_deltas = getCircles(circle_centers)  # circle_deltas = np.sum(circle_centers * circle_normals, axis=1)
        rings.addRing(intersecting_points, intersecting_faces)
    rings.lrf = lrf
    return rings








def CSIRSv2Arbitrary(mesh, seed_point, radius, n_rings, n_points, intersec_direction=None):
    """
    CSIRS (Circle-Surface Intersection): Generates concentric concentricRings of points on the surface (vertex, face)
    @param mesh: a Mesh object
    @param seed_point: (1x3) array
    @param radius: the max extension of the concentricRings
    @param n_points: define the number of points per ring
    @param n_rings: number of concentric concentricRings
    @param intersec_direction: (1x3) array, define the direction to generate the seed_point on the mesh surface

    @return:
    """

    radius = radius / n_rings
    isInFace = False
    if intersec_direction is None:
        # Calculate the nearest mesh vertex
        closest_v_idx = np.argmin(pairwise_distances(mesh.vertices, seed_point.reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))  # Prendo il punto intersecante più vicino al punto precedente in modo da costruire gli array in modo ordinato
        # Select the adjacent faces
        faces_idx = mesh.vertex_faces[closest_v_idx]
        # Search for the intersection
        for f_idx in faces_idx:
            intersection_point = intersectLinePlane(seed_point, mesh.face_normals[f_idx], mesh.vertices[closest_v_idx], mesh.face_normals[f_idx])
            isInFace = pointInFace(intersection_point, mesh.vertices[mesh.faces[f_idx]])
            if isInFace:
                seed_face_idx = f_idx
                break

        if not isInFace:
            # Calculate the direction to intersect the various mesh resolutions surfaces
            pca = PCA(n_components=3)
            pca.fit(mesh.vertices)
            intersec_direction = pca.components_[2]

            for f_idx in range(len(mesh.faces)):
                intersection_point = intersectLinePlane(seed_point, intersec_direction, mesh.vertices[mesh.faces[f_idx][0]], mesh.face_normals[f_idx])
                isInFace = pointInFace(intersection_point, mesh.vertices[mesh.faces[f_idx]])
                if isInFace:
                    seed_face_idx = f_idx
                    break
    else:
        for f_idx in range(len(mesh.faces)):
            intersection_point = intersectLinePlane(seed_point, intersec_direction, mesh.vertices[mesh.faces[f_idx][0]], mesh.face_normals[f_idx])
            isInFace = pointInFace(intersection_point, mesh.vertices[mesh.faces[f_idx]])
            if isInFace:
                seed_face_idx = f_idx
                break

    if not isInFace:
        return None

    ###### INITIALIZATION PHASE  #########
    seed_point = intersection_point
    concentricRings = ConcentricRings(seed_point, seed_face_idx)

    # Creates the first ring
    circle_centers = np.tile(seed_point, (n_points, 1))
    circle_normals = generateCircleNormals(mesh.face_normals[seed_face_idx], n_points, on360=True)
    circle_deltas = np.sum(circle_centers * circle_normals, axis=1)  # Substitute of circle_deltas = np.dot(circle_centers, circle_normals)

    intersecting_points = np.zeros((n_points, 3))
    intersecting_faces = np.tile(seed_face_idx, (n_points, 1))
    candidates_v, candidates_faces = [], []  # per evitare i warnings
    for r in range(n_rings):
        for center_idx in range(circle_centers.shape[0]):
            if np.isnan(circle_centers[center_idx]).any():  # controllo che il centro del punto dell'anello precedente esista (che sia diverso da [nan, nan, nan])
                found = False
            else:
                found, candidates_v, candidates_faces = CSI_Arbitrary(mesh, radius, circle_centers[center_idx], circle_normals[center_idx], circle_deltas[center_idx], intersecting_faces[center_idx])

            if found:
                # Assegno a ogni anello i punti d'intersezione
                if r == 0:  # Primo anello
                    if center_idx == 0:  # Primo cerchio
                        intersecting_points[center_idx] = candidates_v[0]
                        intersecting_faces[center_idx] = candidates_faces[0]
                    else:
                        closest_v_idx = np.argmin(pairwise_distances(candidates_v, intersecting_points[center_idx - 1].reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))  # Prendo il punto intersecante più vicino al punto precedente in modo da costruire gli array in modo ordinato
                        intersecting_points[center_idx] = candidates_v[closest_v_idx]
                        intersecting_faces[center_idx] = candidates_faces[closest_v_idx]
                else:  # Per ogni anello che non sia il primo prendo il punto d'intersezione più lontano dal punto dell'iterazione precedente
                    if r <= 1:
                        closest_v_idx = np.argmax(pairwise_distances(candidates_v, seed_point.reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))
                    else:
                        closest_v_idx = np.argmax(pairwise_distances(candidates_v, concentricRings.rings[r - 2].points[center_idx].reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))
                    intersecting_points[center_idx] = candidates_v[closest_v_idx]
                    intersecting_faces[center_idx] = candidates_faces[closest_v_idx]
            else:  # se non trovo dei punti d'intersezione non fermo l'algoritmo ma do un valore particolare al punto
                intersecting_points[center_idx] = np.full(3, np.nan)
                intersecting_faces[center_idx] = -1

        circle_centers = np.array(intersecting_points)
        circle_deltas = np.sum(circle_centers * circle_normals, axis=1)
        concentricRings.addRing(intersecting_points, intersecting_faces)
    return concentricRings


def CSIRSv3(mesh, seed_point, radius, n_points, n_rings):
    """
    CSIRS (Circle-Surface Intersection): Generates concentric rings of points on the surface (vertex, face)
    @param mesh: a Mesh object
    @param seed_point: index of a seed point
    @param radius: the max extension of the rings
    @param n_points: define the number of points per ring
    @param n_rings: number of concentric rings

    @return:
    """
    radius = radius / n_rings

    ###### INITIALIZATION PHASE  #########
    # Creates the first ring
    circle_centers = np.tile(mesh.vertices[seed_point], (n_points, 1))
    circle_normals = generateCircleNormals(mesh.vertex_normals[seed_point], n_points, on360=True)
    circle_deltas = np.sum(circle_centers * circle_normals, axis=1)  # Substitute of circle_deltas = np.dot(circle_centers, circle_normals)

    rings = ConcentricRings()
    intersecting_points = np.zeros((n_points, 3))
    intersecting_faces = - np.ones(n_points, dtype=int)
    candidates_v, candidates_faces = [], []  # per evitare i warnings
    for r in range(n_rings):
        for center_idx in range(circle_centers.shape[0]):
            if np.isnan(circle_centers[center_idx]).any():  # controllo che il centro del punto dell'anello precedente esista (che sia diverso da [nan, nan, nan])
                found = False
            else:
                found, candidates_v, candidates_faces = CSI(mesh, radius, circle_centers[center_idx], circle_normals[center_idx], circle_deltas[center_idx], intersecting_faces[center_idx])

            if found:
                # Assegno a ogni anello i punti d'intersezione
                if r == 0:  # Primo anello
                    if center_idx == 0:  # Primo cerchio
                        intersecting_points[center_idx] = candidates_v[0]
                        intersecting_faces[center_idx] = candidates_faces[0]
                    else:
                        idx = np.argmin(pairwise_distances(candidates_v, intersecting_points[center_idx - 1].reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))  # Prendo il punto intersecante più vicino al punto precedente in modo da costruire gli array in modo ordinato
                        intersecting_points[center_idx] = candidates_v[idx]
                        intersecting_faces[center_idx] = candidates_faces[idx]
                else:  # Per ogni anello che non sia il primo prendo il punto d'intersezione più lontano dal punto dell'iterazione precedente
                    if r <= 1:
                        idx = np.argmax(pairwise_distances(candidates_v, mesh.vertices[seed_point].reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))
                    else:
                        idx = np.argmax(pairwise_distances(candidates_v, rings.rings[r - 2].points[center_idx].reshape((1, 3)), metric="sqeuclidean", force_all_finite="allow-nan"))
                    intersecting_points[center_idx] = candidates_v[idx]
                    intersecting_faces[center_idx] = candidates_faces[idx]
            else:  # se non trovo dei punti d'intersezione non fermo l'algoritmo ma do un valore particolare al punto
                intersecting_points[center_idx] = np.full(3, np.nan)
                intersecting_faces[center_idx] = -1

        circle_centers = np.array(intersecting_points)
        circle_deltas = np.sum(circle_centers * circle_normals, axis=1)

        rings.addRing(intersecting_points, intersecting_faces)
    return rings


def getCircles(points):
    """
    For each point in points generates a vector to be used as normal for circles generation.
    @param points: (Nx3) array of points
    @return: (Nx3) array of normals and (N) deltas

    """
    shift_dx = np.roll(points, 1, axis=0)
    shift_sx = np.roll(points, -1, axis=0)

    normals = np.empty((len(points), 3))
    deltas = np.empty(len(points))
    for p in range(len(points)):
        vector = shift_sx[p] - shift_dx[p]
        normals[p] = vector / np.linalg.norm(vector)
        deltas[p] = np.dot(points[p], normals[p])
    return normals, deltas
