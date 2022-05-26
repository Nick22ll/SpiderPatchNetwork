

import numpy as np
from numpy.linalg import *
from math import *


from scipy.spatial.transform import Rotation


def threePlaneIntersection(n1, d1, n2, d2, n3=None, d3=0):
    """

    :param n1: normal to the first plane
    :param d1: delta of the first plane ( delta = dot(normal_to_plane, point_of_plane)
    :param n2:
    :param d2:
    :param n3:
    :param d3:
    :return:
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


def intersectLineSurface(points, vector, Xo):
    """
    :param points: set of points (Nx3)
    :param vector: directional(collinear) vector for the line passing by Xo (1x3)
    :param Xo:
    :return closest_point: the closest point, in the set of points, to the line passing by Xo and collinear to the vector n
    """
    distances = np.abs(distPointToLine(points, vector, Xo))
    return np.argmin(distances)


def pointInFace(points, face):
    """

    :param points: (Nx3) array of points
    :param face: (3x3) array of points that represents a face
    :return: (Rx3) array of points
    """
    intersecting_points = []
    for point in points:
        a = face[1] - face[0]
        b = face[2] - face[0]
        c = point - face[0]
        ab = a[0] * b[1] - a[1] * b[0]
        ac = a[0] * c[1] - a[1] * c[0]
        cb = c[0] * b[1] - c[1] * b[0]

        if np.sign(ab) == np.sign(ac) and np.sign(ac) == np.sign(cb) or np.any(np.absolute(np.array([ab, ac, cb])) < .00000000001):
            a = face[2] - face[1]
            b = face[0] - face[1]
            c = point - face[1]
            ab = a[0] * b[1] - a[1] * b[0]
            ac = a[0] * c[1] - a[1] * c[0]
            cb = c[0] * b[1] - c[1] * b[0]
            if np.sign(ab) == np.sign(ac) and np.sign(ac) == np.sign(cb) or np.any(np.absolute(np.array([ab, ac, cb])) < .00000000001):
                intersecting_points.append(point)

    return np.array(intersecting_points)


def distPointToLine(points, line_orientation, Xo):
    """
    Calculates the distance between a point and a line
    :param points: the point or Nx3 points where N is the number of points
    :param line_orientation: the line orientation  1x3
    :param Xo: a point belonging to the line  1x3
    :return dist:
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


    :param vertex_normals: Nx3 array of vertices normal used to calculate the circles normals
    :param n_normals:
    :param on360: set True if you want a 360-degree rotation subdivision
    :return: (n_normals x 3 ) array
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

