from GeometricUtils import *

from sklearn.metrics import pairwise_distances
from numpy.linalg import norm
from scipy.spatial.distance import pdist


def intersectCircleMesh(vertices, facetIn, facetOut, faceIn_idx, original_mesh, geodesic_distance, edge_length, nbh_region_size):  # Ha senso portarsi dietro il facetOut? Non basterebbe ritornare una nuova faccia?
    """

    @param vertices: vertices of the generated by the resamplig of the original mesh
    @param facetIn: a nx3 array of vertex indices representing the facetIn where to do the calculus
    @param facetOut:
    @param faceIn_idx:  index representing the face of facetIn where to do the calculus
    @param original_mesh: the original mesh
    @param geodesic_distance:
    @param edge_length:
    @param nbh_region_size:
    @return:
    """
    found = True
    exception = False
    faceIn = facetIn[faceIn_idx]
    radius = sqrt(pow(edge_length, 2) - pow(edge_length / 2, 2))
    pIn = vertices[faceIn[0]]
    pOut1 = vertices[faceIn[1]]
    pOut2 = vertices[faceIn[2]]

    mid_point = (pOut1 + pOut2) / 2
    circle_normal = (pOut2 - pOut1) / norm(pOut2 - pOut1)
    d_circle = np.dot(circle_normal, mid_point)  # mi serve per definire il piano su cui giace la circonferenza
    # d_circle = delta_circle, deriva dall'equazione vettoriale di un piano per cui il prodotto scalare di un punto appartenete al piano e la normale al piano deve essere pari a delta dot(x, n) = delta

    ### SELECT REGION #####
    mp_distances = pairwise_distances(original_mesh.vertices, mid_point.reshape((1, 3)), metric="sqeuclidean")
    v_idx = np.where(mp_distances < pow(nbh_region_size, 2))[0]
    while len(v_idx) < 1:
        # increase region size if not enough points
        nbh_region_size *= 1.5
        v_idx = np.where(mp_distances < pow(nbh_region_size, 2))[0]

    region_facet = np.where(np.any(np.isin(original_mesh.faces, v_idx), 1) == True)[0]  # find the complete faces for all vertices near to the middle edge point (vertices in the region)
    region_facet = original_mesh.faces[region_facet]

    ### FIND THE INTERSECTION #####
    p_candidate = []
    p_candidate_facet = np.empty(shape=(0, 3), dtype=int)  # array of vertex indeces representing the face of the relative p_candidate
    # Cerco le intersezioni per ogni faccia della mesh originale presente in region_facet generando un vettore di punti candidati tra cui andrò a scegliere
    for face in region_facet:
        p1 = original_mesh.vertices[face[0]]
        p2 = original_mesh.vertices[face[1]]
        p3 = original_mesh.vertices[face[2]]

        face_normal = np.cross(p2 - p1, p3 - p1)
        face_normal = face_normal / norm(face_normal)
        d_face = np.dot(p1, face_normal)

        # The rect generated from as the intersection of the original_mesh face plane and the plane of the facetIn
        line_point = threePlaneIntersection(face_normal, d_face, circle_normal, d_circle)
        line_dir = np.cross(face_normal, circle_normal)  # line direction vector, it is the line generated from as the intersection of the original_mesh face plane and the plane of the facetIn

        points = lineSphereIntersection(line_dir, line_point, mid_point, radius)

        ###### counter-clockwise rule  #######
        # trovati i punti guardo quale dei due fa parte della faccia della mesh originale
        for point in points:
            ab = np.cross(p2 - p1, p3 - p1)
            ac = np.cross(p2 - p1, point - p1)
            cb = np.cross(point - p1, p3 - p1)

            if np.sign(ab[2]) == np.sign(ac[2]) and np.sign(ac[2]) == np.sign(cb[2]):
                ab = np.cross(p3 - p2, p1 - p2)
                ac = np.cross(p3 - p2, point - p2)
                cb = np.cross(point - p2, p1 - p2)

                if np.sign(ab[2]) == np.sign(ac[2]) and np.sign(ac[2]) == np.sign(cb[2]):
                    p_candidate.append(point)
                    p_candidate_facet = np.vstack((p_candidate_facet, [face]))

    if len(p_candidate) == 0:
        exception = True
        found = False
        return found, vertices, facetOut, exception

    #### CHOSE BEST INTERSECTION POINT ######
    p_candidate = np.array(p_candidate)
    p_candidate_facet = np.array(p_candidate_facet, dtype=int)
    pIn_distances = pairwise_distances(p_candidate, pIn.reshape((1, 3)), metric="sqeuclidean")
    if p_candidate.shape[0] <= 2:
        if p_candidate.shape[0] == 1 or (p_candidate.shape[0] > 1 and abs(pIn_distances[0] - pIn_distances[1]) > pow(radius, 2)):  # perchè si fa la differenza tra le distanze?
            idx = np.argmax(pIn_distances)
            if pIn_distances[idx] < pow(radius, 2):
                found = False
                exception = True  # TODO aggiunto per prova
                return found, vertices, facetOut, exception
            else:
                best_point = p_candidate[idx]
        else:
            if pIn_distances.shape[0] > 1 and (facetOut.shape[0] == 0 or (facetOut.shape[0] != 0 and facetOut[-1, 2] == faceIn[1])):  # se sto generando il primo faceOut o l'ultimo punto dell'ultimo faceOut corrisponde al secondo punto del faceIn
                best = np.argmax(np.mean(geodesic_distance[region_facet[p_candidate_facet]]))
                best_point = p_candidate[best]
            elif pIn_distances.shape[0] > 1 and (facetOut.shape[0] != 0 and facetOut[-1, 2] == faceIn[1]):  # Se non è il primo faceOut ma l'ultimo punto dell'ultimo faceOut corrispondente al secondo punto del faceIn
                best = np.argmin(pairwise_distances(p_candidate, original_mesh.vertices[facetOut[-1, 1]], metric="sqeuclidean"))
                best_point = p_candidate[best]
            else:
                # if next to the border
                found = False
                return found, vertices, facetOut, exception
    else:
        # More than 2 candidates (error while intersecting the circle)
        # Quando succede? Se ci sono  dei ripiegamenti sulla mesh?
        idx = np.where(pIn_distances > pow(radius, 2))[0]
        p_candidate = p_candidate[idx]
        p_candidate_facet = p_candidate_facet[idx]
        if p_candidate.shape[0] > 1:
            wrong = findWrongPCandidate(original_mesh, p_candidate_facet, p_candidate)
            p_candidate = p_candidate[np.logical_not(wrong)]
            # p_candidate_facet = p_candidate_facet[np.logical_not(wrong)]
        if p_candidate.shape[0] > 1:
            if facetOut.shape[0] == 0:
                best = findBestPCandidate(p_candidate, mid_point)
            else:
                best = findBestPCandidate(p_candidate, vertices[facetOut[-1, 1]])
            best_point = p_candidate[best]
        elif p_candidate.shape[0] == 0:
            # If next to the border
            found = False
            return found, vertices, facetOut, exception
        else:
            best_point = np.array(p_candidate)

    best_point = best_point.reshape((1, 3))
    if norm(best_point - vertices[-1]) <= pow(edge_length / 2, 2):
        vertices[-1] = np.mean(np.vstack((vertices[-1], best_point)), axis=0)
        facetOut = np.vstack((facetOut, np.array([faceIn[1], vertices.shape[0] - 1, faceIn[2]], ndmin=2)))
    elif facetOut.shape[0] != 0 and faceIn_idx > faceIn.shape[0] - 2 and norm(best_point - vertices[facetOut[0, 1]]) <= edge_length / 2:
        # Merge with the first FacetOut (exception)
        vertices[facetOut[0, 1]] = np.mean(np.vstack((vertices[facetOut[0, 1]], best_point)))
        facetOut = np.vstack((facetOut, np.array([faceIn[1], facetOut[0, 1], faceIn[2]])))
    else:
        # Add the new point
        vertices = np.vstack((vertices, best_point))
        facetOut = np.vstack((facetOut, np.array([faceIn[1], vertices.shape[0] - 1, faceIn[2]], ndmin=2)))
    return found, vertices, facetOut, exception


def intersectCircleMesh_slim(vertices, faceIn, original_mesh, edge_length, nbh_region_size):
    radius = sqrt(pow(edge_length, 2) - pow(edge_length / 2, 2))
    pIn = vertices[faceIn[0]]
    pOut1 = vertices[faceIn[1]]
    pOut2 = vertices[faceIn[2]]

    mid_point = (pOut1 + pOut2) / 2
    circle_normal = (pOut2 - pOut1) / norm(pOut2 - pOut1)
    d_circle = np.dot(circle_normal, mid_point)  # mi serve per definire il piano su cui giace la circonferenza
    # d_circle = delta_circle, deriva dall'equazione vettoriale di un piano per cui il prodotto scalare di un punto appartenete al piano e la normale al piano deve essere pari a delta dot(x, n) = delta

    ### SELECT REGION #####
    mp_distances = pairwise_distances(original_mesh.vertices, mid_point.reshape((1, 3)), metric="sqeuclidean")
    v_idx = np.where(mp_distances < pow(nbh_region_size, 2))[0]
    while len(v_idx) < 1:
        # increase region size if not enough points
        nbh_region_size *= 1.5
        v_idx = np.where(mp_distances < pow(nbh_region_size, 2))[0]

    region_facet = np.where(np.any(np.isin(original_mesh.faces, v_idx), 1) == True)[0]  # find the complete faces for all vertices near to the middle edge point (vertices in the region)
    region_facet = original_mesh.faces[region_facet]

    ### FIND THE INTERSECTION #####
    p_candidate = []
    p_candidate_facet = np.empty(shape=(0, 3), dtype=int)  # array of vertex indeces representing the face of the relative p_candidate
    # Cerco le intersezioni per ogni faccia della mesh originale presente in region_facet generando un vettore di punti candidati tra cui andrò a scegliere
    for face in region_facet:
        p1 = original_mesh.vertices[face[0]]
        p2 = original_mesh.vertices[face[1]]
        p3 = original_mesh.vertices[face[2]]

        face_normal = np.cross(p2 - p1, p3 - p1)
        face_normal = face_normal / norm(face_normal)
        d_face = np.dot(p1, face_normal)

        # The rect generated from as the intersection of the original_mesh face plane and the plane of the facetIn
        line_point = threePlaneIntersection(face_normal, d_face, circle_normal, d_circle)
        line_dir = np.cross(face_normal, circle_normal)  # line direction vector, it is the line generated from as the intersection of the original_mesh face plane and the plane of the facetIn

        points = lineSphereIntersection(line_dir, line_point, mid_point, radius)

        ###### counter-clockwise rule  #######
        # trovati i punti guardo quale dei due fa parte della faccia della mesh originale
        for point in points:
            ab = np.cross(p2 - p1, p3 - p1)
            ac = np.cross(p2 - p1, point - p1)
            cb = np.cross(point - p1, p3 - p1)

            if np.sign(ab[2]) == np.sign(ac[2]) and np.sign(ac[2]) == np.sign(cb[2]):
                ab = np.cross(p3 - p2, p1 - p2)
                ac = np.cross(p3 - p2, point - p2)
                cb = np.cross(point - p2, p1 - p2)

                if np.sign(ab[2]) == np.sign(ac[2]) and np.sign(ac[2]) == np.sign(cb[2]):
                    p_candidate.append(point)
                    p_candidate_facet = np.vstack((p_candidate_facet, [face]))

    if len(p_candidate) == 0:
        return np.array(p_candidate)
    p_candidate = np.array(p_candidate)
    idx = np.where(pairwise_distances(p_candidate, pIn.reshape(1, -1), metric="sqeuclidean") >= pow(radius, 2))[0]
    return p_candidate[idx]


def findWrongPCandidate(original_mesh, p_faces, p_candidates):
    """
    Returns the indices of p_candidates wrongly selected, by checking their distance to their original facet
    @param original_mesh: the original Mesh Object
    @param p_faces: (nx3) array of vertex indices, represents the faces of p_candidates
    @param p_candidates: (nx3) array representing the candidate points
    @return: a list of indices of p_candidates wrongly selected
    """
    n_candidate = p_candidates.shape[0]
    max_dist = []
    dist = []
    for idx in range(n_candidate):
        face_points = original_mesh.vertices[p_faces[idx]]
        max_dist.append(pow(np.amax(pdist(face_points)), 2))

        dist.append(np.amax(pairwise_distances(p_candidates, face_points, metric="sqeuclidean")))

    return np.array(dist) > np.array(max_dist)


def findBestPCandidate(p_candidates, p_ref):
    """
    Finds the best p_candidate
    @param p_candidates: (nx3) array of candidate points
    @param p_ref: (1x3) reference point
    @return: the index of the best candidate point
    """
    return np.argmin(pairwise_distances(p_candidates, p_ref.reshape((-1, 3)), metric="sqeuclidean"))
