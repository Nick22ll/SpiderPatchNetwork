import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from trimesh import Trimesh
from trimesh.curvature import discrete_mean_curvature_measure, discrete_gaussian_curvature_measure


def getCurvatures(mesh):
    ## VERTEX FEATURES
    Cmean_vertex, Cgauss_vertex, __, __, lambda1, lambda2 = principalCurvatures(mesh, True)
    Curvedness_vertex = np.sqrt((np.multiply(lambda1, lambda1) + np.multiply(lambda2, lambda2)) / 2)
    K2_vertex = np.maximum(lambda1, lambda2)
    localDepth_vertex = calculateLocalDepth(mesh)
    vertex_curvatures = {"mean_curvature": Cmean_vertex, "gauss_curvature": Cgauss_vertex, "curvedness": Curvedness_vertex, "K2": K2_vertex, "local_depth": localDepth_vertex}

    ## FACET FEATURES
    face_curvatures = {}
    face_curvatures["mean_curvature"] = (Cmean_vertex[mesh.faces[:, 0]] + Cmean_vertex[mesh.faces[:, 1]] + Cmean_vertex[mesh.faces[:, 2]]) / 3
    face_curvatures["gauss_curvature"] = (Cgauss_vertex[mesh.faces[:, 0]] + Cgauss_vertex[mesh.faces[:, 1]] + Cgauss_vertex[mesh.faces[:, 2]]) / 3
    face_curvatures["curvedness"] = (Curvedness_vertex[mesh.faces[:, 0]] + Curvedness_vertex[mesh.faces[:, 1]] + Curvedness_vertex[mesh.faces[:, 2]]) / 3
    face_curvatures["K2"] = (K2_vertex[mesh.faces[:, 0]] + K2_vertex[mesh.faces[:, 1]] + K2_vertex[mesh.faces[:, 2]]) / 3
    face_curvatures["local_depth"] = (localDepth_vertex[mesh.faces[:, 0]] + localDepth_vertex[mesh.faces[:, 1]] + localDepth_vertex[mesh.faces[:, 2]]) / 3
    return vertex_curvatures, face_curvatures


def principalCurvatures(mesh, usethird=False):
    # Number of vertices
    vertex_number = mesh.vertices.shape[0]

    # Calculate vertices normals
    vertex_normals = mesh.vertex_normals()

    Lambda1 = np.zeros((vertex_number, 1))
    Lambda2 = np.zeros((vertex_number, 1))
    Dir1 = np.zeros((vertex_number, 3))
    Dir2 = np.zeros((vertex_number, 3))

    for i in range(0, vertex_number):
        rot, invRot = VectorRotationMatrix(vertex_normals[i])
        if not usethird:
            vertex_neighbours = mesh.verticesNeighbours(mesh.adjacency_list[i])
        else:
            vertex_neighbours = mesh.verticesNeighbours(mesh.verticesNeighbours(mesh.adjacency_list[i]))

        Vertices = mesh.vertices[vertex_neighbours]

        We = np.dot(Vertices, invRot)
        f = We[:, 0]
        x = We[:, 1]
        y = We[:, 2]

        # f(x,y) = ax^2 + by^2 + cxy + dx + ey + f
        FM = np.transpose(np.vstack((np.ravel(x) ** 2, np.ravel(y) ** 2, (np.ravel(x) * np.ravel(y)), np.ravel(x), np.ravel(y), np.ones(np.size(x)))))
        sol = np.linalg.lstsq(FM, np.ravel(f), rcond=None)[0]

        # H =  [2*a c;c 2*b];
        Dxx = 2 * sol[0]
        Dxy = sol[2]
        Dyy = 2 * sol[1]

        Lambda1[i], Lambda2[i], I1, I2 = eig2(Dxx, Dxy, Dyy)
        dir1 = np.dot(np.hstack((0, I1[0], I1[1])), rot)
        dir2 = np.dot(np.hstack((0, I2[0], I2[1])), rot)
        Dir1[i, :] = dir1 / np.sqrt(dir1[0] ** 2 + dir1[1] ** 2 + dir1[2] ** 2)
        Dir2[i, :] = dir2 / np.sqrt(dir2[0] ** 2 + dir2[1] ** 2 + dir2[2] ** 2)

    Cmean = (Lambda1 + Lambda2) / 2
    Cgaussian = Lambda1 * Lambda2

    return Cmean.flatten(), Cgaussian.flatten(), Dir1.flatten(), Dir2.flatten(), Lambda1.flatten(), Lambda2.flatten()


def eig2(Dxx=None, Dxy=None, Dyy=None):
    # Compute the eigenvectors
    tmp = np.sqrt((Dxx - Dyy) ** 2 + 4 * Dxy ** 2)
    v2x = 2 * Dxy
    v2y = Dyy - Dxx + tmp

    # Normalize
    mag = np.sqrt(v2x ** 2 + v2y ** 2)
    if mag != 0:
        v2x = v2x / mag
        v2y = v2y / mag

    # The eigenvectors are orthogonal
    v1x = - v2y
    v1y = v2x

    # Compute the eigenvalues
    mu1 = abs(0.5 * (Dxx + Dyy + tmp))
    mu2 = abs(0.5 * (Dxx + Dyy - tmp))

    # Sort eigen values by absolute value abs(Lambda1)<abs(Lambda2)
    if mu1 < mu2:
        Lambda1 = mu1
        Lambda2 = mu2
        I1 = np.hstack((v1x, v1y))
        I2 = np.hstack((v2x, v2y))
    else:
        Lambda1 = mu2
        Lambda2 = mu1
        I1 = np.hstack((v2x, v2y))
        I2 = np.hstack((v1x, v1y))
    return Lambda1, Lambda2, I1, I2


def VectorRotationMatrix(v=None):
    v = np.ravel(v).conj().transpose() / np.sqrt(sum(v ** 2))
    k = np.random.rand(3)
    l = np.hstack(((k[1] * v[2]) - (k[2] * v[1]), (k[2] * v[0]) - (k[0] * v[2]), (k[0] * v[1]) - (k[1] * v[0])))
    l /= np.sqrt(sum(l ** 2))
    k = np.hstack(((l[1] * v[2]) - (l[2] * v[1]), (l[2] * v[0]) - (l[0] * v[2]), (l[0] * v[1]) - (l[1] * v[0])))
    k /= np.sqrt(sum(k ** 2))
    Minv = np.vstack((np.ravel(v), np.ravel(l), np.ravel(k))).transpose()
    M = np.linalg.inv(Minv)
    return M, Minv


def calculateCurvaturesOnPoint(mesh, points, radius_multiplier=1.0):
    """
    Returns the discrete gaussian curvature measure and the discrete mean curvature measure of a sphere centered at a point as detailed in ‘Restricted Delaunay triangulations and normal cycle’- Cohen-Steiner and Morvan.
    Gaussian Curvature is the sum of the vertex defects at all vertices within the radius for each point.
    Mean Curvature is the sum of the angle at all edges contained in the sphere for each point.

    :param mesh: a Mesh() object
    :param points:  ((n, 3) float) – Points in space
    :param radius_multiplier:  (float) – Sphere radius multiplier which should typically be greater than zero. The base sphere radius is calculated as the average mesh faces edge lenght
    :return: G, H (((n,),(n,)) float)
    """
    points = points.reshape((-1, 3))
    radius = 1 * radius_multiplier
    tri_mesh = Trimesh(mesh.vertices, mesh.faces, process=True)
    G = discrete_gaussian_curvature_measure(tri_mesh, points, radius)
    H = discrete_mean_curvature_measure(tri_mesh, points, radius)
    return G, H


def calculateLocalDepth(mesh, seed_points=None):
    """
    Calculate the LocalDepth centered on a seed point (must be vertex indices) using mesh points in a fixed radius
    :param mesh: Mesh Object
    :param seed_points: an array of length N of mesh vertex indices
    :param radius: float
    :return:
    """

    pca = PCA(n_components=3)
    if seed_points is not None:
        vertexLD = np.full(shape=len(seed_points), fill_value=np.nan)
        interval = range(len(seed_points))
    else:
        vertexLD = np.full(shape=len(mesh.vertices), fill_value=np.nan)
        interval = range(len(mesh.vertices))
        seed_points = list(interval)
    for v_idx in interval:
        neigh_vertices = mesh.verticesNeighbours((mesh.verticesNeighbours(mesh.adjacency_list[seed_points[v_idx]]))) # np.where(pairwise_distances(mesh.vertices, mesh.vertices[seed_points[v_idx]].reshape((1, -1)), metric="sqeuclidean") < pow(radius, 2))[0]
        neigh_vertices = mesh.vertices[neigh_vertices]
        if len(neigh_vertices) > 5:
            mass_center = np.mean(neigh_vertices, axis=0)
            neigh_vertices -= mass_center
            pca.fit(neigh_vertices)
            component = pca.components_[2, :]
            vertexLD[v_idx] = np.abs(np.dot(mesh.vertices[seed_points[v_idx]] - mass_center, component))

    while np.isnan(vertexLD).any():
        for v_idx in np.where(np.isnan(vertexLD))[0]:
            neigh_vertices = mesh.adjacency_list[v_idx]
            vertexLD[v_idx] = np.mean(vertexLD[neigh_vertices][~np.isnan(vertexLD[neigh_vertices])])
    return vertexLD


def calculateArbitraryLocalDepth(mesh, seed_points, radius):
    """
    Calculate the LocalDepth centered on a seed point (that can be an arbitrary point in the mesh space) using mesh points in a fixed radius
    :param mesh: Mesh Object
    :param seed_points: a (nx3) array of 3D space points
    :param radius: float
    :return:
    """
    radius = pow(radius, 2)
    pca = PCA(n_components=3)
    vertexLD = np.full(shape=len(seed_points), fill_value=np.nan)
    for v_idx in range(len(seed_points)):
        neigh_vertices = np.where(pairwise_distances(mesh.vertices, seed_points[v_idx].reshape((1, -1)), metric="sqeuclidean") < radius)[0]
        neigh_vertices = mesh.vertices[neigh_vertices]
        while len(neigh_vertices) <= 5:
            radius *= 1.2
            neigh_vertices = np.where(pairwise_distances(mesh.vertices, seed_points[v_idx].reshape((1, -1)), metric="sqeuclidean") < radius)[0]
            neigh_vertices = mesh.vertices[neigh_vertices]
        mass_center = np.mean(neigh_vertices, axis=0)
        neigh_vertices -= mass_center
        pca.fit(neigh_vertices)
        component = pca.components_[2, :]
        vertexLD[v_idx] = np.abs(np.dot(seed_points[v_idx] - mass_center, component))

    # Approccio alternativo: cerco il vertice della mesh più vicino e utilizzo i suoi vicini per il calcolo. DA FINIRE: gestire il caso in cui i vicini non sono abbastanza
    # TODO DA FINIRE: gestire il caso in cui i vicini non sono abbastanza
    # TODO chiedere quale approccio utilizzare
    return vertexLD


def calculateFaceLocalDepth(mesh, faces, radius):
    """
    Calculate the LocalDepth of a mesh face. It is the mean of  Local Depths of face vertices.
    :param mesh: Mesh Object
    :param faces: a list or array of faces(a (nx3) array of point indices)
    :param radius: float
    :return:
    """
    seed_points, reverse_idx = np.unique(faces.flatten(), return_inverse=True)
    LD = calculateLocalDepth(mesh, seed_points, radius)
    LD = LD[reverse_idx]
    LD = LD.reshape((-1, 3))
    return np.mean(LD, axis=1)
