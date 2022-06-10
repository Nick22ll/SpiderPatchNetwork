import open3d as o3d

import trimesh
from trimesh import caching
from MeshCurvatures import *


class Mesh:
    def __init__(self, vertices=None, faces=None, onGPU=False, deviceID=0):

        if onGPU:
            self.device = o3d.core.Device(o3d.core.Device.CUDA, deviceID)
        else:
            self.device = o3d.core.Device(o3d.core.Device.CPU, deviceID)

        self.mesh = o3d.t.geometry.TriangleMesh()
        self.edges = None
        self.faces = None
        self.vertices = None
        self.edge_length = 0
        self.face_adjacency_list = None
        self.vertex_faces = None
        self.adjacency_list = None
        self.vertex_curvatures = None
        self.face_curvatures = None

        if vertices is not None and faces is not None:
            vertices = o3d.utility.Vector3dVector(vertices)
            faces = o3d.utility.Vector3iVector(faces)
            self.mesh = o3d.geometry.TriangleMesh(vertices, faces)
            self.computeDataStructures()

        self.base_color = np.array([0.7, 0.7, 0.7])

    def computeDataStructures(self):
        self.vertices = np.asarray(self.mesh.vertices)
        self.faces = np.asarray(self.mesh.triangles)
        self.mesh.compute_vertex_normals()
        self.mesh.compute_triangle_normals()
        self.mesh.compute_adjacency_list()
        self.computeAdjacenciesLists()
        self.computeEdges()
        self.edge_length = self.averageEdgeLength()

    def computeCurvatures(self):
        self.vertex_curvatures, self.face_curvatures = getCurvatures(self)

    def computeEdges(self):
        """
        Calculates edges of every mesh triangle (face)
        :return edges: an array with dimensions (num_edges, 2), it is an array of mesh vertex indices
        """
        first_edges = self.faces[:, [0, 1]]
        second_edges = self.faces[:, [1, 2]]
        third_edges = self.faces[:, [2, 0]]
        edges = np.sort(np.concatenate((first_edges, second_edges, third_edges), 0), axis=1)
        self.edges = edges

    def computeAdjacenciesLists(self):
        self.vertex_faces = [[] for _ in range(len(self.vertices))]
        for f_idx in range(len(self.faces)):
            for v_idx in self.faces[f_idx]:
                self.vertex_faces[v_idx].append(f_idx)

        self.face_adjacency_list = [[] for _ in range(len(self.faces))]
        for f_idx in range(len(self.faces)):
            for v_idx in self.faces[f_idx]:
                self.face_adjacency_list[f_idx].extend(self.vertex_faces[v_idx])
            self.face_adjacency_list[f_idx] = np.unique(self.face_adjacency_list[f_idx]).tolist()

        self.adjacency_list = [[] for _ in range(len(self.vertices))]
        for v_idx in range(len(self.vertices)):
            for f_idx in self.vertex_faces[v_idx]:
                self.adjacency_list[v_idx].extend(self.faces[f_idx])
            self.adjacency_list[v_idx] = np.unique(self.adjacency_list[v_idx]).tolist()

    def uniqueEdges(self):
        return np.unique(self.edges, axis=0, return_index=True, return_counts=True)

    def edgesUsedTimes(self, used_times=1):
        unique_edges, indices, counts = self.uniqueEdges()
        idx = np.where(counts == used_times)
        return unique_edges[idx]

    def averageEdgeLength(self):
        edges = self.edges
        vertices1 = self.vertices[edges[:, 0]]
        vertices2 = self.vertices[edges[:, 1]]
        distances = np.linalg.norm(vertices1 - vertices2, axis=1)
        return np.mean(distances)

    def faces_normals(self):
        return np.asarray(self.mesh.triangle_normals)

    def verticesNeighbours(self, v_indices):
        neighbours = []
        for v in v_indices:
            neighbours.extend(self.adjacency_list[v])
        return np.unique(neighbours)

    def vertex_normals(self):
        return np.asarray(self.mesh.vertex_normals)

    def getBoundaryVertices(self, neighbors_level=0):
        """
        Return the list of boundary vertices (those vertices that belongs to an edge tha is adjacent to a single triangle)
        :param neighbors_level: parameter that consider boundary vertices also the n-level neighbors of a real boundary vertex
        :return:
        """
        boundary_edges = self.edgesUsedTimes(1)
        boundary_vertices = np.unique(boundary_edges.reshape((-1, 1)))
        for i in range(0, neighbors_level):
            boundary_vertices = np.unique(np.hstack((boundary_vertices, self.verticesNeighbours(boundary_vertices))))
        return boundary_vertices

    def draw(self):
        o3d.visualization.draw_geometries([self.mesh], mesh_show_back_face=True)

    def draw_with_rings(self, rings):
        import matplotlib.cm
        points = np.empty((0, 3))
        colors = np.empty((0, 3))
        faces = np.empty(0, dtype=int)
        cmap = matplotlib.cm.get_cmap()

        for ring_idx in range(len(rings.rings)):
            points = np.concatenate((points, rings.rings[ring_idx].points))
            faces = np.append(faces, rings.rings[ring_idx].faces)
            n_points = len(rings.rings[ring_idx].points) - 1
            for point_idx in range(len(rings.rings[ring_idx].points)):
                colors = np.vstack((colors, np.array(cmap(point_idx / n_points)[0:3])))
        faces = faces[np.where(faces != -1)[0]]
        faces = np.unique(self.faces[faces].reshape(-1, 1))

        self.mesh.paint_uniform_color(self.base_color)
        for f in faces:
            self.mesh.vertex_colors[f] = np.array([0.9, 0.5, 0])
        points = o3d.utility.Vector3dVector(points)
        cloud = o3d.geometry.PointCloud(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([self.mesh, cloud], mesh_show_back_face=True)

    def draw_faces(self, indices, color=np.array([1, 0, 0])):
        faces = np.unique(self.faces[indices].reshape(-1, 1))
        self.mesh.paint_uniform_color(self.base_color)
        for f in faces:
            self.mesh.vertex_colors[f] = color
        o3d.visualization.draw_geometries([self.mesh], mesh_show_back_face=True)

    def drawWithPointCloud(self, points):
        points = o3d.utility.Vector3dVector(points)
        cloud = o3d.geometry.PointCloud(points)
        o3d.visualization.draw_geometries([self.mesh, cloud], mesh_show_back_face=True)

    def draw_with_patches(self, patches):
        for_draw_patches = []
        for patch in patches:
            for_draw_patches.extend(patch.to_draw())
        o3d.visualization.draw_geometries([self.mesh] + for_draw_patches, mesh_show_back_face=True)

    def draw_with_MeshGraph(self, mesh_graph):
        for_draw_patches = []
        for patch in mesh_graph.patches:
            for_draw_patches.extend(patch.to_draw())
        o3d.visualization.draw_geometries([self.mesh] + for_draw_patches, mesh_show_back_face=True)

    def load(self, mesh_path):
        """
        Load a mesh. Format avaiable: .inp), ANSYS msh (.msh), AVS-UCD (.avs), CGNS (.cgns), DOLFIN XML (.xml), Exodus (.e, .exo), FLAC3D (.f3grid), H5M (.h5m), Kratos/MDPA (.mdpa), Medit (.mesh, .meshb), MED/Salome (.med), Nastran (bulk data, .bdf, .fem, .nas), Netgen (.vol, .vol.gz), Neuroglancer precomputed format, Gmsh (format versions 2.2, 4.0, and 4.1, .msh), OBJ (.obj), OFF (.off), PERMAS (.post, .post.gz, .dato, .dato.gz), PLY (.ply), STL (.stl), Tecplot .dat, TetGen .node/.ele, SVG (2D output only) (.svg), SU2 (.su2), UGRID (.ugrid), VTK (.vtk), VTU (.vtu), WKT (TIN) (.wkt), XDMF (.xdmf, .xmf).
        :param mesh_path: the path of the mesh to load
        :return: None
        """
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.computeDataStructures()
        print(f"Mesh loaded")

    def save(self, path):
        """
        Save a mesh.
        :param path: the path with filename and extension (ex. C:/user/file/mesh_filename.off )
        :return: None
        """
        o3d.io.write_triangle_mesh(path, self.mesh)
        print(f"Mesh saved at: {path}")

    def has_adjacency_lists(self):
        return self.adjacency_list is not None

    def has_curvatures(self):
        return self.vertex_curvatures is not None

    def has_edges(self):
        return self.edges is not None


def expandFacet(facet, mesh):
    """
    Expands the facet with all the neighbour faces of the facet
    :param facet: N indices representing the facet to expand
    :param mesh: Mesh object
    :return: a list of face indices representing the expanded facet
    """
    if not isinstance(facet, list) and not isinstance(facet, np.ndarray):
        return np.array(mesh.face_adjacency_list[facet])
    else:
        expanded_facet = []
        for face_idx in facet:
            expanded_facet.extend(mesh.face_adjacency_list[face_idx])
        return np.unique(expanded_facet)


class Meshv2(Trimesh):
    def __init__(self, vertices=None, faces=None):
        super().__init__(vertices=vertices, faces=faces)
        self.face_adjacency_list = None
        self.edge_length = None
        self.base_color = [156, 47, 255, 255]

    def calculateFaceAdjacencyList(self):
        self.face_adjacency_list = [[] for _ in range(len(self.faces[:]))]

        for faces in self.face_adjacency[:]:
            self.face_adjacency_list[faces[0]].append(faces[1])
            self.face_adjacency_list[faces[1]].append(faces[0])

        for i in range(len(self.face_adjacency_list)):
            self.face_adjacency_list[i] = np.unique(self.face_adjacency_list[i])

        self.face_adjacency_list = trimesh.caching.tracked_array(self.face_adjacency_list)

    def getBoundaryVertices(self, neighbors_level=0):
        """
        Return the list of boundary vertices (those vertices that belongs to an edge tha is adjacent to a single triangle)
        :param neighbors_level: parameter that consider boundary vertices also the n-level neighbors of a real boundary vertex
        :return:
        """
        prova = self.edgesUsedTimes(1).reshape((-1, 1))
        boundary_vertices = np.unique(prova)
        for i in range(0, neighbors_level):
            faces = np.where(any(np.isin(self.faces, boundary_vertices), 1) == True)[0]
            boundary_vertices = np.unique(self.faces[faces].reshape((-1, 1)))
        return boundary_vertices

    def averageEdgeLength(self):
        self.edge_length = np.mean(np.array(self.edges_unique_length[:]), axis=0)
        return self.edge_length

    def vertices_faces(self, vertex_indices):
        return self.vertex_faces[np.where(self.vertex_faces[vertex_indices])[0]]

    def faces_normals(self):
        return self.faces_normals[:]

    def neighbourFacesIdx(self, face_idx):
        """
        :param face_idx: index of the face to retrieve neighbours
        :return: list of indices of neighbours
        """
        return self.face_adjacency_list[face_idx]

    def neighbourFaces(self, face_idx):
        """
        :param face_idx: index of the face to retrieve neighbours or (1x3) array representing a face
        :return: neighbours faces
        """
        if face_idx.ndim > 1:
            face_idx = np.where(self.faces == face_idx)[0]
        return self.faces[self.neighbourFacesIdx(face_idx)]

    def uniqueEdges(self):
        return np.unique(np.sort(self.edges[:], axis=1), axis=0, return_index=True, return_counts=True)

    def edgesUsedTimes(self, used_times=1):
        unique_edges, indices, counts = self.uniqueEdges()
        idx = np.where(counts == used_times)
        return unique_edges[idx]

    def drawWithPointCloud(self, points):
        cloud = trimesh.points.PointCloud(points)
        cloud.vertices_color = [0.5, 0.4, 0.7]
        self.visual.face_colors = self.base_color
        scene = trimesh.Scene([self, cloud])
        scene.show(viewer='gl')

    def draw_with_rings(self, rings):
        import matplotlib.cm
        points = np.empty((0, 3))
        colors = np.empty((0, 4))
        faces = np.empty(0, dtype=int)
        cmap = matplotlib.cm.get_cmap()
        scaler = matplotlib.cm.ScalarMappable(cmap=cmap)
        for ring_idx in range(len(rings.rings)):
            points = np.concatenate((points, rings.rings[ring_idx].points))
            faces = np.append(faces, rings.rings[ring_idx].faces)
            n_points = len(rings.rings[ring_idx].points) - 1
            for point_idx in range(len(rings.rings[ring_idx].points)):
                colors = np.vstack((colors, scaler.to_rgba(cmap(point_idx / n_points))))
        faces = faces[np.where(faces != -1)[0]]
        faces = np.unique(self.faces[faces].reshape(-1, 1))

        self.visual.face_colors = self.base_color
        for f in faces:
            self.visual.face_colors[f] = [170, 146, 255, 255]
        cloud = trimesh.PointCloud(points)
        cloud.vertices_colors = colors
        scene = trimesh.Scene([self])
        scene.show()


def loadMeshv2(path):
    mesh = trimesh.load_mesh(path)
    mesh = Meshv2(np.array(mesh.vertices), np.array(mesh.faces))
    mesh.averageEdgeLength()
    mesh.calculateFaceAdjacencyList()
    return mesh


def expandFacetv2(facet, mesh):
    """
    Expands the facet with all the neighbour faces of the facet
    :param facet: N indices representing the facet to expand
    :param mesh: Mesh object
    :return: a list of face indices representing the expanded facet
    """
    if facet.ndim < 1:
        facet = [facet]
    # expanded_facet = array([facet], dtype=int) #Include also the initial facet
    expanded_facet = np.empty(0, dtype=int)
    for face_idx in np.where(facet != -1)[0]:
        expanded_facet = np.hstack((expanded_facet, mesh.face_adjacency_list[face_idx]))
    return np.unique(expanded_facet)
