import numpy as np
from numpy.linalg import *
from math import *

from scipy.spatial.transform import Rotation
from sklearn.metrics import pairwise_distances


def threePlaneIntersection(n1, d1, n2, d2, n3=None, d3=0):
    """

    @param n1: normal to the first plane
    @param d1: delta of the first plane ( delta = dot(normal_to_plane, point_of_plane)
    @param n2:
    @param d2:
    @param n3:
    @param d3:
    @return:
    """

    if n3 is None:
        n3 = [1, 0, 0]

    A = np.vstack((n1, n2, n3))
    detA = det(A)
    d = np.array([d1, d2, d3])
    if detA == 0:
        #  Warning : No Three Plane Intersection
        np.random.seed(2)
        n3 = np.random.rand(3)
        p = threePlaneIntersection(n1, d1, n2, d2, n3, d3)
    else:
        Ax = np.array(A)
        Ax[:, 0] = np.array(d)
        Ay = np.array(A)
        Ay[:, 1] = np.array(d)
        Az = np.array(A)
        Az[:, 2] = np.array(d)
        p = np.hstack((det(Ax) / detA, det(Ay) / detA, det(Az) / detA))
    return p


def lineSphereIntersection(line_dir, p_line, sphere_center, radius):
    p = []
    a = np.sum(line_dir ** 2)

    b = 2 * line_dir[0] * (p_line[0] - sphere_center[0]) + \
        2 * line_dir[1] * (p_line[1] - sphere_center[1]) + \
        2 * line_dir[2] * (p_line[2] - sphere_center[2])
    c = pow(p_line[0] - sphere_center[0], 2) + \
        pow(p_line[1] - sphere_center[1], 2) + \
        pow(p_line[2] - sphere_center[2], 2) - pow(radius, 2)
    delta = pow(b, 2) - (4 * a * c)
    roots = np.empty(shape=(2, 1))
    if delta >= 0:
        roots[0] = (-b + sqrt(delta)) / (2 * a)
        roots[1] = (-b - sqrt(delta)) / (2 * a)
        x = p_line[0] + line_dir[0] * roots
        y = p_line[1] + line_dir[1] * roots
        z = p_line[2] + line_dir[2] * roots
        p = np.concatenate((x, y, z), axis=1)
    return p


def intersectLineDiscreteSurface(points, vector, Xo):
    """
    @param points: set of points (Nx3)
    @param vector: directional(collinear) vector for the line passing by Xo (1x3)
    @param Xo:
    @return closest_point: the closest point, in the set of points, to the line passing by Xo and collinear to the vector n
    """
    distances = np.abs(distPointToLine(points, vector, Xo))
    return np.argmin(distances)


def intersectLinePlane(l_point, l_dir, p_point, p_normal, epsilon=1e-8):
    """
    @param l_point: (1x3) array representing a point of the line
    @param l_dir: directional vector for the line passing by l_point (1x3)
    @param p_point: (1x3) array representing a point of the plane
    @param p_normal: (1x3) array representing the plane normal
    @return (1x3) array or None: the intersection point of line and plane

    """
    dot = np.dot(p_normal, l_dir)
    if np.abs(dot) > epsilon:
        w = l_point - p_point
        fac = -(np.dot(p_normal, w)) / dot
        return l_point + (l_dir * fac)
    return None


def pointsInFace(points, face):
    """

    @param points: (Nx3) array of points
    @param face: (3x3) array of points that represents a face
    @return: (Rx3) array of points
    """
    intersecting_points = []
    for point in points:
        a = face[1] - face[0]
        b = face[2] - face[0]
        c = point - face[0]
        ab = a[0] * b[1] - a[1] * b[0]
        ac = a[0] * c[1] - a[1] * c[0]
        cb = c[0] * b[1] - c[1] * b[0]

        if np.sign(ab) == np.sign(ac) and np.sign(ac) == np.sign(cb) or np.any(np.absolute(np.array([ab, ac, cb])) < 10e-18):
            a = face[2] - face[1]
            b = face[0] - face[1]
            c = point - face[1]
            ab = a[0] * b[1] - a[1] * b[0]
            ac = a[0] * c[1] - a[1] * c[0]
            cb = c[0] * b[1] - c[1] * b[0]
            if np.sign(ab) == np.sign(ac) and np.sign(ac) == np.sign(cb) or np.any(np.absolute(np.array([ab, ac, cb])) < 10e-18):
                intersecting_points.append(point)

    return np.array(intersecting_points)


def pointsInFacev2(points, face):
    inFace = [pointInFace(point, face) for point in points]
    return points[inFace]


def pointInFace(point, face):
    """
    implementation of triangle interior theorem
    @param point:
    @param face:
    @return:
    """
    v0 = face[2] - face[0]
    v1 = face[1] - face[0]
    v2 = point - face[0]

    # Compute dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    # Compute barycentric coordinates
    commonDen = (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) / commonDen
    v = (dot00 * dot12 - dot01 * dot02) / commonDen

    # Check if point is in triangle
    return (u >= 0) and (v >= 0) and (u + v < 1)


def faceOf3DPoint(point, mesh):
    """
    @return: (Rx3) array of points
    """

    for face_id, face in enumerate(mesh.faces):
        a = mesh.vertices[face[1]] - mesh.vertices[face[0]]
        b = mesh.vertices[face[2]] - mesh.vertices[face[0]]
        c = point - mesh.vertices[face[0]]
        ab = a[0] * b[1] - a[1] * b[0]
        ac = a[0] * c[1] - a[1] * c[0]
        cb = c[0] * b[1] - c[1] * b[0]

        if np.sign(ab) == np.sign(ac) and np.sign(ac) == np.sign(cb) or np.any(np.absolute(np.array([ab, ac, cb])) < .00000000001):
            a = mesh.vertices[face[2]] - mesh.vertices[face[1]]
            b = mesh.vertices[face[0]] - mesh.vertices[face[1]]
            c = point - mesh.vertices[face[1]]
            ab = a[0] * b[1] - a[1] * b[0]
            ac = a[0] * c[1] - a[1] * c[0]
            cb = c[0] * b[1] - c[1] * b[0]
            if np.sign(ab) == np.sign(ac) and np.sign(ac) == np.sign(cb) or np.any(np.absolute(np.array([ab, ac, cb])) < .00000000001):
                return face_id

    return -1


def faceOf3DPointv2(point, mesh):
    """
    @return: (Rx3) array of points
    """
    tmp = {}
    for face_id, face in enumerate(mesh.faces):
        if pointInFace(point, mesh.vertices[face]):
            tmp[face_id] = np.mean(pairwise_distances(point.reshape((1, 3)), mesh.vertices[face], metric="sqeuclidean"))
    if tmp:
        return min(tmp, key=tmp.get)
    return -1


def distPointToLine(points, line_orientation, Xo):
    """
    Calculates the distance between a point and a line
    @param points: the point or Nx3 points where N is the number of points
    @param line_orientation: the line orientation  1x3
    @param Xo: a point belonging to the line  1x3
    @return dist:
    """
    N = points.shape[0]
    line_orientations = np.tile(line_orientation, (N, 1))
    Xos = np.tile(Xo, (N, 1))
    cross_products_norm = norm(np.cross(points - Xos, line_orientations), axis=1)
    return cross_products_norm / norm(line_orientation)


def getAngleBetweenNormals(normals, v):
    normals = normals.reshape((-1, 3))
    v = v / norm(v)

    n_normal = normals.shape[0]
    angle = np.zeros((n_normal, 1))

    for i in range(n_normal):
        u = normals[i]
        u = u / norm(u)
        angle[i] = np.arctan2(norm(np.cross(u, v)), np.dot(u, v)) * 180 / pi

    return angle


def generateCircleNormals(vertex_normals, n_normals, on360=False):
    """
    Generates n_normals rotating a vertex_normal


    @param vertex_normals: Nx3 array of vertices normal used to calculate the circles normals
    @param n_normals:
    @param on360: set True if you want a 360-degree rotation subdivision
    @return: (n_normals x 3 ) array
    """

    circle_normals = np.empty((n_normals, 3))
    circle_normals[0] = np.cross(vertex_normals, [1, 0, 0])
    if np.sum(circle_normals[0]) == 0:
        circle_normals[0] = np.cross(vertex_normals, [0, 1, 0])
    if on360:
        rotation_radians = np.radians(360 / n_normals)
    else:
        rotation_radians = np.radians(180 / n_normals)
    for i in range(1, n_normals):
        rotation_vector = rotation_radians * i * vertex_normals
        rotation = Rotation.from_rotvec(rotation_vector)
        circle_normals[i] = rotation.apply(circle_normals[0])

    return circle_normals


def faceArea(mesh, face_idx):
    corner = mesh.vertices[mesh.faces[face_idx][:, 0]]
    a = mesh.vertices[mesh.faces[face_idx][:, 1]] - corner
    b = mesh.vertices[mesh.faces[face_idx][:, 2]] - corner
    return norm(np.cross(a, b), axis=1) / 2


def LRF_normals(mesh, seed_point_idx, radius):
    distances = pairwise_distances(mesh.vertices, mesh.vertices[seed_point_idx].reshape((1, 3)), metric="euclidean")
    indices = np.where(distances < radius)[0]
    feature_point = mesh.vertex_normals()[seed_point_idx]
    support_points = mesh.vertex_normals()[indices]
    distances = distances[indices]
    M = np.zeros((3, 3))
    factor = 0
    for idx, p in enumerate(support_points):
        factor += radius - distances[idx]
        M += (radius - distances[idx]) * (np.dot((p - feature_point).reshape((3, 1)), np.transpose((p - feature_point).reshape((3, 1)))))
    M /= factor
    eigenvalues, eigenvectors = np.linalg.eig(M)

    sort_eig_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_eig_indices]
    eigenvectors = eigenvectors[sort_eig_indices]
    x_positives, x_negatives = 0, 0
    z_positives, z_negatives = 0, 0
    for idx, p in enumerate(support_points):
        if np.dot((p - feature_point), eigenvectors[0]) >= 0:
            x_positives += 1
        else:
            x_negatives += 1

        if np.dot((p - feature_point), eigenvectors[2]) >= 0:
            z_positives += 1
        else:
            z_negatives += 1

    if x_positives >= x_negatives:
        x = eigenvectors[0]
    else:
        x = - eigenvectors[0]

    if z_positives >= z_negatives:
        z = eigenvectors[2]
    else:
        z = - eigenvectors[2]

    y = np.cross(x, z)

    return x, y, z


def LRF(mesh, seed_point, radius):
    distances = pairwise_distances(mesh.vertices, seed_point.reshape((1, 3)), metric="euclidean")
    indices = np.where(distances < radius)[0]
    feature_point = seed_point
    support_points = mesh.vertices[indices]
    distances = distances[indices]
    M = np.zeros((3, 3))
    factor = 0
    for idx, p in enumerate(support_points):
        factor += radius - distances[idx]
        M += (radius - distances[idx]) * (np.dot((p - feature_point).reshape((3, 1)), np.transpose((p - feature_point).reshape((3, 1)))))
    M /= factor
    eigenvalues, eigenvectors = np.linalg.eig(M)

    sort_eig_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_eig_indices]
    eigenvectors = eigenvectors[sort_eig_indices]
    x_positives, x_negatives = 0, 0
    z_positives, z_negatives = 0, 0
    for idx, p in enumerate(support_points):
        if np.dot((p - feature_point), eigenvectors[0]) >= 0:
            x_positives += 1
        else:
            x_negatives += 1

        if np.dot((p - feature_point), eigenvectors[2]) >= 0:
            z_positives += 1
        else:
            z_negatives += 1

    if x_positives >= x_negatives:
        x = eigenvectors[0]
    else:
        x = - eigenvectors[0]

    if z_positives >= z_negatives:
        z = eigenvectors[2]
    else:
        z = - eigenvectors[2]

    y = np.cross(x, z)

    return x, y, z
