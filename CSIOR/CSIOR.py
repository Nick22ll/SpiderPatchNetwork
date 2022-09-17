import time

from CSI.CSI import CSI
from Mesh.Mesh import Mesh
from .IntersectCircleMesh import *
from .FixFacet import *

from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import gdist


def CSIOR(original_mesh, edge_length=None, seed_point_idx=None):
    """
    Circle-Surface Intersection Ordered Resampling (CSIOR), a mesh tesselation algorithm that resample a mesh in a equilateral(quasi-equilateral) triangular mesh
    preserving the object shape and local geometric properties as corrugations or relief patterns.

    @param original_mesh: the mesh to resample
    @param edge_length: (optional) desired equilateral edge length, if None is calculated as the average of the original mesh edge lengths
    @param seed_point_idx: (optional) original mesh point from which start the resampling

    @return resampled_mesh: the resampled mesh
    @return rings:
    """
    CSIOR_start = time.time()
    if not edge_length:
        edge_length = original_mesh.edge_length

    if not seed_point_idx:
        seed_point_idx = generateSeedPoint(original_mesh)

    if original_mesh.edge_length < edge_length:
        nbh_region_size = edge_length + original_mesh.edge_length
    else:
        nbh_region_size = original_mesh.edge_length

    geodesic_distances = gdist.compute_gdist(original_mesh.vertices, original_mesh.faces, np.array(seed_point_idx, dtype=int).reshape(1))  # Computes the exact geodesic distance (fast marching is an approximate alghoritm). Interesting link:
    # Fast Marching wrapper: https://github.com/SILI1994/fast_matching_python        Geodesic Distance: https://github.com/the-virtual-brain/tvb-gdist
    iter_max = max(geodesic_distances) / edge_length  # Set to avoid infinite loop in case of resampling errors

    ##########  FIRST EXAGON GENERATION  ############
    radius = sqrt(pow(edge_length, 2) - pow((edge_length / 2), 2))

    circle_centers = np.tile(original_mesh.vertices[seed_point_idx], (3, 1))  # Array of vertices indices
    circle_normals = generateCircleNormals(original_mesh.vertex_normals[seed_point_idx], 3)
    circle_deltas = np.sum(circle_centers * circle_normals, axis=1)  # Substitute of circle_deltas = np.dot(circle_centers, circle_normals)
    first_hexagon = np.zeros((6, 3))

    for i in range(circle_centers.shape[0]):
        _, intersecting_points, _ = CSI(original_mesh, edge_length, circle_centers[i], circle_normals[i], circle_deltas[i])
        if i == 0:
            first_hexagon[i] = intersecting_points[0]
            first_hexagon[int(6 / 2) + i] = intersecting_points[1]
        else:
            idx = np.argmin(pairwise_distances(intersecting_points, first_hexagon[i - 1].reshape((1, 3)), metric="sqeuclidean"))  # Prendo il punto intersecante più vicino al punto precedente(nella prima metà dell'array) in modo da costruire gli array in modo ordinato
            idx = (np.arange(2) == idx)
            first_hexagon[i] = intersecting_points[idx]
            first_hexagon[i + int(6 / 2)] = intersecting_points[np.logical_not(idx)]

    ### New Mesh Generation ###
    new_mesh_vertices = np.concatenate((original_mesh.vertices[seed_point_idx].reshape((1, 3)), first_hexagon), axis=0)
    new_mesh_faces = np.concatenate((np.zeros(shape=(6, 1)), np.arange(1, 7).reshape((6, 1)), np.roll(np.arange(1, 7).reshape((6, 1)), -1)), axis=1).astype(int)

    ring_count = 1
    rings = {ring_count: np.arange(new_mesh_faces.shape[0])}

    vertices_valence = np.ones(7)
    vertices_valence[0] = 6
    vertices_valence[1:] = 2

    ############ ITERABLE RESAMPLING ################
    facetIn = np.array(new_mesh_faces)
    n_face = 0
    iteration = 0

    while n_face != new_mesh_faces.shape[0] and iteration <= iter_max:
        start = time.time()
        iteration += 1
        n_face = new_mesh_faces.shape[0]
        facetOut = np.empty(shape=(0, 3), dtype=int)

        ######## DISCOVER NEW FacetOut ##########
        facetIn_used = np.ones(len(facetIn), dtype=bool)
        for face_idx in range(len(facetIn)):
            face = facetIn[face_idx]
            pIn = new_mesh_vertices[face[0]]  # Non serve nel for...
            pOut1 = new_mesh_vertices[face[1]]  # Non serve nel for...
            pOut2 = new_mesh_vertices[face[2]]
            if facetOut.shape[0] == 0 or pairwise_distances(pOut2.reshape((1, 3)), new_mesh_vertices[facetOut[-1, 1]].reshape((1, 3)), metric="sqeuclidean") > pow(edge_length, 2):
                # Use Circle Intersection
                found, new_mesh_vertices, facetOut, exception = intersectCircleMesh(new_mesh_vertices, facetIn, facetOut, face_idx, original_mesh, geodesic_distances, edge_length, nbh_region_size)
                run = 1
                while not found and exception and run <= 3:
                    found, new_mesh_vertices, facetOut, exception = intersectCircleMesh(new_mesh_vertices, facetIn, facetOut, face_idx, original_mesh, geodesic_distances, edge_length * (pow(1.5, run)), nbh_region_size * (pow(1.5, run)))
                    run += 1
                if found:
                    vertices_valence = updateValence(vertices_valence, facetOut[-1])
                    if face_idx > 1 and face[1] == facetIn[face_idx - 1, 2] and facetIn[face_idx - 1, 1] == facetIn[face_idx - 2, 2] and not facetIn_used[face_idx - 1] and facetIn_used[face_idx - 2]:
                        new_mesh_vertices = np.concatenate((new_mesh_vertices, np.mean(np.concatenate((new_mesh_vertices[-1].reshape((-1, 3)), new_mesh_vertices[-2].reshape((-1, 3))), axis=0), axis=0).reshape((-1, 3))), axis=0)
                        new_face = np.array([facetIn[face_idx - 1, 1], new_mesh_vertices.shape[0] - 1, facetIn[face_idx - 1, 2]])
                        vertices_valence = updateValence(vertices_valence, new_face)
                        facetOut = np.array([facetOut[:-2], new_face, facetOut[-1]])
                        facetIn_used[face_idx - 1] = True
                else:
                    facetIn_used[face_idx - 1] = False
            else:
                # If previously discovered vertex is close to FacetIn then connect it
                facetOut = np.concatenate((facetOut, np.array([face[1], facetOut[-1, 1], face[2]]).reshape((-1, 3))))
                vertices_valence = updateValence(vertices_valence, facetOut[-1])

        ######   FIND FacetIn   #########
        n_fout = facetOut.shape[0]
        facetOut_shift = np.roll(facetOut, shift=-1, axis=0)
        facetIn_new = np.empty((0, 3), dtype=int)
        prev_p = np.empty((0, 3))
        p_idx = np.empty(0, dtype=int)
        for fout in range(n_fout):
            if facetOut[fout, 2] == facetOut_shift[fout, 0] and facetOut[fout, 1] != facetOut_shift[fout, 1]:  # If consecutive facetOut connect them
                if vertices_valence[facetOut[fout, 2]] >= 5 and \
                        pairwise_distances(new_mesh_vertices[facetOut[fout, 1]].reshape(1, -1), new_mesh_vertices[facetOut_shift[fout, 1]].reshape(1, -1), metric="sqeuclidean") < pow(edge_length * 1.5, 2) or \
                        pairwise_distances(new_mesh_vertices[facetOut[fout, 1]].reshape(1, -1), new_mesh_vertices[facetOut_shift[fout, 1]].reshape(1, -1), metric="sqeuclidean") <= pow(edge_length, 2):
                    # If valance 5 then direct connection or consecutive FacetOUT are close enough
                    if vertices_valence[facetOut[fout, 2]] < 5 and pairwise_distances(new_mesh_vertices[facetOut[fout, 1]].reshape((-1, 3)), new_mesh_vertices[facetOut_shift[fout, 1]].reshape((-1, 3)), metric="sqeuclidean") <= pow(edge_length, 2):
                        print("pause(.1);")  # TODO chiedere
                    new_face = np.array([facetOut[fout, 2], facetOut[fout, 1], facetOut_shift[fout, 1]], dtype=int)
                    vertices_valence = updateValence(vertices_valence, new_face)

                    facetIn_new, facet_add = fixFacetInAll(new_mesh_vertices, facetIn_new.reshape((-1, 3)), new_face.reshape((-1, 3)), edge_length)
                    new_mesh_faces = np.vstack((new_mesh_faces, facetOut[fout], facet_add, new_face))
                    for i in range(facet_add.shape[0]):
                        vertices_valence = updateValence(vertices_valence, facet_add[i])
                else:
                    # If valance not 5 then add a point in the middle or consecutive FacetOut are distant
                    p_candidate1 = intersectCircleMesh_slim(new_mesh_vertices, facetOut[fout], original_mesh, edge_length, nbh_region_size)
                    p_candidate2 = intersectCircleMesh_slim(new_mesh_vertices, np.roll(facetOut_shift[fout], shift=1, axis=0), original_mesh, edge_length, nbh_region_size)

                    if p_candidate1.shape[0] > 1:
                        idx = np.argmin(pairwise_distances(p_candidate1, np.mean(np.vstack((new_mesh_vertices[facetOut[fout, 1]], new_mesh_vertices[facetOut_shift[fout, 1]])), axis=0).reshape((-1, 3)), metric="sqeuclidean"))
                        p_candidate1 = p_candidate1[idx]
                    if p_candidate2.shape[0] > 1:
                        idx = np.argmin(pairwise_distances(p_candidate2, np.mean(np.vstack((new_mesh_vertices[facetOut[fout, 1]], new_mesh_vertices[facetOut_shift[fout, 1]])), axis=0).reshape((-1, 3)), metric="sqeuclidean"))
                        p_candidate2 = p_candidate2[idx]
                    if p_candidate1.shape[0] == 0:
                        p = p_candidate2
                    elif p_candidate2.shape[0] == 0:
                        p = p_candidate1
                    else:
                        p = (p_candidate1 + p_candidate2) / 2

                    if p.shape[0] == 0:
                        new_mesh_faces = np.vstack((new_mesh_faces, facetOut[fout]))
                        continue

                    if p_idx.shape[0] != 0:
                        p_dist = pairwise_distances(prev_p.reshape(-1, 3), p.reshape(-1, 3), metric="sqeuclidean")
                        min_dist = p_dist.min()
                        idx = np.argmin(p_dist)
                    if p_idx.shape[0] != 0 and min_dist < pow(radius, 2):  # New point is close to a previous one (exception)
                        new_mesh_vertices[p_idx[idx]] = np.mean(np.vstack((new_mesh_vertices[p_idx[idx]], p)), axis=0)
                        p_idx = np.append(p_idx, p_idx[idx])
                    else:
                        new_mesh_vertices = np.vstack((new_mesh_vertices, p))
                        prev_p = np.vstack((prev_p, p))
                        p_idx = np.append(p_idx, new_mesh_vertices.shape[0] - 1)

                    new_face = np.vstack((np.array([facetOut[fout, 2], facetOut[fout, 1], p_idx[-1]], dtype=int), np.array([facetOut[fout, 2], p_idx[-1], facetOut_shift[fout, 1]], dtype=int)))
                    vertices_valence = updateValence(vertices_valence, new_face[0])
                    vertices_valence = updateValence(vertices_valence, new_face[1])

                    if p_idx.shape[0] > 1 and min_dist < pow(radius, 2):  # New point is close to a previous one (exception)
                        facetIn_new = np.vstack((facetIn_new[0:-2], new_face[1]))
                        new_mesh_faces = np.vstack((new_mesh_faces, facetOut[fout], new_face))
                    else:
                        facetIn_new, facet_add = fixFacetInAll(new_mesh_vertices, facetIn_new, new_face.reshape((-1, 3)), edge_length)
                        new_mesh_faces = np.concatenate((new_mesh_faces, facetOut[fout].reshape(-1, 3), facet_add, new_face), axis=0)
                        for i in range(facet_add.shape[0]):
                            vertices_valence = updateValence(vertices_valence, facet_add[i])
            else:
                new_mesh_faces = np.concatenate((new_mesh_faces, facetOut[fout].reshape((-1, 3))), axis=0)
            if fout == n_fout and facetIn_new.shape[0] != 0:  # exception for the last FacetIn
                fixed, facet_fixed = fixFacetIn(new_mesh_vertices, facetIn_new[-1], facetIn_new[0], edge_length)
                if fixed:
                    facetIn_new = facetIn_new[1:-2]
                    vertices_valence = updateValence(vertices_valence, facet_fixed)
                    facetIn_new, facet_add = fixFacetInAll(new_mesh_vertices, facetIn_new, facet_fixed.reshape((-1, 3)), edge_length)
                    new_mesh_faces = np.concatenate((new_mesh_faces, facet_add, facet_fixed))
                    for i in range(facet_add.shape[0]):
                        vertices_valence = updateValence(vertices_valence, facet_add[i])
        facetIn = facetIn_new.copy()
        print(f"Ring {ring_count} completed in {time.time() - start} s!")
        ring_count += 1
        rings[ring_count] = np.arange(n_face, new_mesh_faces.shape[0])
    resampled_mesh = Mesh(new_mesh_vertices, new_mesh_faces)
    print(f"Remeshing completato in {time.time() - CSIOR_start}s!")
    return resampled_mesh, rings


def updateValence(vertices_valence, face):
    """
    @param vertices_valence: array of vertices valences
    @param face: a 3x1 array of vertex indices representing a face
    @return:
    """
    if np.any(face >= vertices_valence.shape[0]):
        vertices_valence = np.append(vertices_valence, 0)

    vertices_valence[face] += 1
    return vertices_valence


def generateSeedPoint(mesh):
    pca = PCA(n_components=3)
    pca.fit(mesh.vertices)
    less_var_vector = pca.components_[2]
    return intersectLineDiscreteSurface(mesh.vertices, less_var_vector, np.mean(mesh.vertices, axis=0))


def generate_hex_vertices(radius):
    """
    Generates hexagon vertices tha are laying on a circle.
    @param radius: scalar, the radius of the circle.
    @return: (6x3) array, the hexagon vertices
    """
    angles = np.arange(0, pi * 2, pi / 3).reshape((6, 1))
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = 0 * angles
    return np.hstack((x, y, z))


def project_hexagon(center_point, mesh_vertices, nbh_region_size, hexagon):
    """
    Project an hexagon on a mesh surface
    @param center_point: the point of the mesh in which the hexagon will be centered
    @param mesh_vertices:
    @param nbh_region_size: neighbourhood region size, it determines the considered region for the calculus of the region normal
    @param hexagon: (6x3) hexagon to project
    @return:
    """
    mesh_vertices = mesh_vertices - center_point  # Traslate to the center point
    v_distances = norm(mesh_vertices, axis=1)
    nbh_region_idx = np.where(v_distances < nbh_region_size)[0]

    # Increase region size if not enough points
    while len(nbh_region_idx) < 20:
        nbh_region_size *= 2
        nbh_region_idx = np.where(v_distances < nbh_region_size)[0]

    nbh_region = mesh_vertices[nbh_region_idx]

    # ROTO-TRASLATION
    pca = PCA(n_components=3)
    pca.fit(nbh_region)
    nbh_region = np.transpose(np.matmul(pca.components_, np.transpose(nbh_region)))

    # INTERPOLATION AND PROJECTION
    xy = nbh_region[:, :2]
    z = nbh_region[:, 2]
    hex_x = hexagon[:, 0]
    hex_y = hexagon[:, 1]
    hex_z = griddata(xy, z, (hex_x, hex_y), method='linear')  # linear interpolation using scipy, natural method doesn't exist
    # hex_z = naturalneighbor.griddata(xy, z, (hex_x, hex_y))
    hexagon = np.hstack((hex_x, hex_y, hex_z))

    hexagon = np.linalg.lstsq(pca.components_, np.transpose(hexagon), rcond=None)[0]
    hexagon = np.transpose(hexagon) + center_point
    return hexagon
