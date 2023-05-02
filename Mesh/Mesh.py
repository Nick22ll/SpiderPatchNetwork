import copy
import pickle

import open3d as o3d
import numpy as np
import trimesh
from trimesh import caching, Trimesh
from Mesh.MeshCurvatures import *
import matplotlib.cm as cm
import matplotlib.colors as colors


class Mesh:
    def __init__(self, vertices=None, faces=None):
        self.edges = None
        self.faces = None
        self.vertices = None
        self.face_normals = None
        self.vertex_normals = None
        self.edge_length = 0
        self.face_adjacency_list = None
        self.vertex_faces = None
        self.adjacency_list = None
        self.vertex_curvatures = {}
        self.face_curvatures = {}
        self.oriented_bounding_box = {}

        if vertices is not None and faces is not None:
            self.vertices = vertices
            self.faces = faces
            self.computeDataStructures()

        self.base_color = np.array([0.7, 0.7, 0.7])

    def computeDataStructures(self, mesh=None):
        if mesh is None:
            vertices = o3d.utility.Vector3dVector(self.vertices)
            faces = o3d.utility.Vector3iVector(self.faces)
            mesh = o3d.geometry.TriangleMesh(vertices, faces)
        else:
            self.vertices = np.asarray(mesh.vertices)
            self.faces = np.asarray(mesh.triangles)

        mesh.compute_vertex_normals()
        self.vertex_normals = np.asarray(mesh.vertex_normals)

        mesh.compute_triangle_normals()
        self.face_normals = np.asarray(mesh.triangle_normals)

        self.computeAdjacenciesLists()
        self.computeEdges()
        self.edge_length = self.averageEdgeLength()

    def computeCurvatures(self, radius):
        """

        @param radius: (int) The level of radius to compute curvatures - 0 --> 0.1% , 1 --> 0.25%, 2--> 1%, 3 --> 2,5%, 4--> 5%

        @return:
        """
        levels = {0: 0.001, 1: 0.0025, 2: 0.01, 3: 0.025, 4: 0.05}
        self.computeOrientedBoundingBox()
        if isinstance(radius, int) and 0 <= radius < 5:
            self.vertex_curvatures[radius], self.face_curvatures[radius] = getCurvatures(self, np.max(self.oriented_bounding_box["extent"]) * levels[radius])

    def computeEdges(self):
        """
        Calculates edges of every mesh triangle (face)
        @return edges: an array with dimensions (num_edges, 2), it is an array of mesh vertex indices
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

    def computeOrientedBoundingBox(self):
        """
        Computes the oriented bounding box based on the PCA of the convex hull.
        @return:
        """
        vertices = o3d.utility.Vector3dVector(self.vertices)
        faces = o3d.utility.Vector3iVector(self.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, faces)
        oriented_bounding_box = mesh.get_oriented_bounding_box()
        self.oriented_bounding_box["center"] = oriented_bounding_box.center
        self.oriented_bounding_box["extent"] = oriented_bounding_box.extent
        return self.oriented_bounding_box

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

    def verticesNeighbours(self, v_indices):
        neighbours = []
        for v in v_indices:
            neighbours.extend(self.adjacency_list[v])
        return np.unique(neighbours)

    def getBoundaryVertices(self, neighbors_level=0):
        """
        Return the list of boundary vertices (those vertices that belongs to an edge tha is adjacent to a single triangle)
        @param neighbors_level: parameter that consider boundary vertices also the n-level neighbors of a real boundary vertex
        @return:
        """
        boundary_edges = self.edgesUsedTimes(1)
        boundary_vertices = np.unique(boundary_edges.reshape((-1, 1)))
        for i in range(0, neighbors_level):
            boundary_vertices = np.unique(np.hstack((boundary_vertices, self.verticesNeighbours(boundary_vertices))))
        return boundary_vertices

    def draw(self, return_to_draw=False):
        vertices = o3d.utility.Vector3dVector(self.vertices)
        faces = o3d.utility.Vector3iVector(self.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, faces)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        if return_to_draw:
            return mesh
        else:
            o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    def drawWithConcRings(self, concRing, lrf=False):
        vertices = o3d.utility.Vector3dVector(self.vertices)
        faces = o3d.utility.Vector3iVector(self.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, faces)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

        # mesh.translate(-0.5 * np.mean(mesh.vertex_normals, axis=0))

        points = concRing.seed_point.reshape((1, 3))
        faces = concRing.seed_point_face
        norm = colors.Normalize(vmin=0, vmax=len(concRing.rings), clip=True)
        color_map = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('plasma'))
        cmap = cm.get_cmap('gist_rainbow')
        color_points = np.empty((0, 3))

        for ring_idx in range(len(concRing.rings)):
            points = np.concatenate((points, concRing.rings[ring_idx].points))
            faces = np.append(faces, concRing.rings[ring_idx].faces)

            for point_idx in range(len(concRing.rings[ring_idx].points)):
                color_points = np.vstack((color_points, np.array(color_map.to_rgba(ring_idx)[0:3])))

        faces = faces[np.where(faces != -1)[0]]
        faces = np.unique(self.faces[faces].reshape(-1, 1))

        mesh.paint_uniform_color(self.base_color)
        # for f in faces:
        #     mesh.vertex_colors[f] = np.array([0.9, 0.5, 0])
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0.75], (len(points), 1)))

        line_set = o3d.geometry.LineSet()

        if lrf:
            idx = np.argmin(pairwise_distances(concRing.seed_point.reshape((1, -1)), mesh.vertices, metric="sqeuclidean"))
            vertex_normal = mesh.vertex_normals[idx]
            points = [concRing.seed_point, (concRing.seed_point + (concRing.lrf[0] * 3)), (concRing.seed_point + (concRing.lrf[1] * 3)), (concRing.seed_point + (concRing.lrf[2] * 3)), (concRing.seed_point + (vertex_normal * 3))]
            lines = [[0, 1], [0, 2], [0, 3], [0, 4]]
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]])

        o3d.visualization.draw_geometries([mesh, point_cloud, line_set], mesh_show_back_face=True)

    def drawFaces(self, indices, color=np.array([1, 0, 0])):
        vertices = o3d.utility.Vector3dVector(self.vertices)
        faces = o3d.utility.Vector3iVector(self.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, faces)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        faces = np.unique(self.faces[indices].reshape(-1, 1))
        mesh.paint_uniform_color(self.base_color)
        for f in faces:
            mesh.vertex_colors[f] = color
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    def drawWithPointCloud(self, points, colors=None):
        vertices = o3d.utility.Vector3dVector(self.vertices)
        faces = o3d.utility.Vector3iVector(self.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, faces)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        points = o3d.utility.Vector3dVector(points)
        cloud = o3d.geometry.PointCloud(points)
        if colors is not None:
            cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([mesh, cloud], mesh_show_back_face=True)

    def drawFacesWithPointCloud(self, face_indices, points):
        vertices = o3d.utility.Vector3dVector(self.vertices)
        faces = o3d.utility.Vector3iVector(self.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, faces)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        color_points = np.tile(np.array([0, 0, 1]), (len(points), 1))
        points = o3d.utility.Vector3dVector(points)
        cloud = o3d.geometry.PointCloud(points)
        cloud.colors = o3d.utility.Vector3dVector(color_points)
        faces = np.unique(self.faces[face_indices].reshape(-1, 1))
        mesh.paint_uniform_color(self.base_color)
        for f in faces:
            mesh.vertex_colors[f] = np.array([1, 0, 0])
        o3d.visualization.draw_geometries([mesh, cloud], mesh_show_back_face=True)

    def drawWithSuperPatch(self, superPatch):
        vertices = o3d.utility.Vector3dVector(self.vertices)
        faces = o3d.utility.Vector3iVector(self.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, faces)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        for_draw = [mesh]
        for_draw.extend(superPatch.to_draw())
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
        points = np.empty((0, 3))
        color_points = np.empty((0, 3))
        for color_id, concRing in enumerate(superPatch.concentricRings[1:]):
            points = np.concatenate((points, concRing.seed_point.reshape((1, 3))))
            color_points = np.vstack((color_points, colors[color_id]))
            for ring_idx in range(len(concRing.rings)):
                points = np.concatenate((points, concRing.rings[ring_idx].points))
                for point_idx in range(len(concRing.rings[ring_idx].points)):
                    color_points = np.vstack((color_points, colors[color_id]))
        points = o3d.utility.Vector3dVector(points)
        cloud = o3d.geometry.PointCloud(points)
        cloud.colors = o3d.utility.Vector3dVector(color_points)
        for_draw.append(cloud)
        o3d.visualization.draw_geometries(for_draw, mesh_show_back_face=True)

    def drawWithMeshes(self, meshes, translate=False):
        columns = 5
        to_print = []
        vertices = o3d.utility.Vector3dVector(self.vertices)
        faces = o3d.utility.Vector3iVector(self.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, faces)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        to_print.append(copy.deepcopy(mesh))
        for i, mesh in enumerate(meshes):
            vertices = o3d.utility.Vector3dVector(mesh.vertices)
            faces = o3d.utility.Vector3iVector(mesh.faces)
            o3dmesh = o3d.geometry.TriangleMesh(vertices, faces)
            o3dmesh.compute_triangle_normals()
            o3dmesh.compute_vertex_normals()
            if translate:
                mesh.computeOrientedBoundingBox()
                o3dmesh = o3dmesh.translate((mesh.oriented_bounding_box["extent"][0] * ((i + 1) % columns), mesh.oriented_bounding_box["extent"][1] * ((i + 1) % 3), 0))
            to_print.append(copy.deepcopy(o3dmesh))
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        mesh_frame.compute_triangle_normals()
        mesh_frame.compute_vertex_normals()
        to_print.append(copy.deepcopy(mesh_frame))
        o3d.visualization.draw_geometries(to_print, mesh_show_back_face=True)

    def drawWithSpiderPatches(self, patches, colors=None):
        vertices = o3d.utility.Vector3dVector(self.vertices)
        faces = o3d.utility.Vector3iVector(self.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, faces)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

        mesh.translate(-2 * np.mean(mesh.vertex_normals, axis=0))

        for_draw_patches = []

        if colors is None:
            cmap = cm.get_cmap('gist_rainbow')
            colors = [cmap(1 / i)[:-1] for i in range(1, len(patches) + 1)]

        for idx, patch in enumerate(patches):
            for_draw_patches.extend(patch.to_draw(colors[idx] if len(colors) > 0 else None))

        o3d.visualization.draw_geometries([mesh] + for_draw_patches, mesh_show_back_face=True)

    def drawWithMeshGraph(self, mesh_graph):
        vertices = o3d.utility.Vector3dVector(self.vertices)
        faces = o3d.utility.Vector3iVector(self.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, faces)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

        cmap = cm.get_cmap('gist_rainbow')

        start, end = mesh_graph.edges()

        lines = []
        for i, s in enumerate(start):
            lines.append([s, end[i]])

        seed_coords = []
        for_draw_patches = []
        for i, patch in enumerate(mesh_graph.patches):
            seed_coords.append(patch.seed_point)
            for_draw_patches.extend(patch.to_draw(color=cmap(1 / (i + 1))[:-1]))

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(seed_coords)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        o3d.visualization.draw_geometries([mesh, line_set] + for_draw_patches, mesh_show_back_face=True)

    def drawWithMeshGraphs(self, mesh_graphs, return_to_draw=False):
        vertices = o3d.utility.Vector3dVector(self.vertices)
        faces = o3d.utility.Vector3iVector(self.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, faces)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        self.computeOrientedBoundingBox()
        meshes = [mesh]
        for i in range(1, len(mesh_graphs)):
            meshes.append(copy.deepcopy(mesh).translate((self.oriented_bounding_box["extent"][0] * i, 0, 0)))
        for_draw_patches = []
        for i, mesh_graph in enumerate(mesh_graphs):
            for patch in mesh_graph.patches:
                for_draw_patches.extend(patch.to_draw())
                for_draw_patches[-2].translate((self.oriented_bounding_box["extent"][0] * i, 0, 0))
                for_draw_patches[-1].translate((self.oriented_bounding_box["extent"][0] * i, 0, 0))
        if return_to_draw:
            return meshes + for_draw_patches
        else:
            o3d.visualization.draw_geometries(meshes + for_draw_patches, mesh_show_back_face=True)

    def drawWithLD(self, radius):
        vertices = o3d.utility.Vector3dVector(self.vertices)
        faces = o3d.utility.Vector3iVector(self.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, faces)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        vmin = np.min(self.vertex_curvatures[radius]["local_depth"])
        vmax = np.max(self.vertex_curvatures[radius]["local_depth"])
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        color_map = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('plasma'))

        rgbs = color_map.to_rgba(self.vertex_curvatures[radius]["local_depth"])[:, :3]
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs)
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    def drawWithGaussCurv(self, radius):
        vertices = o3d.utility.Vector3dVector(self.vertices)
        faces = o3d.utility.Vector3iVector(self.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, faces)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        curv = self.vertex_curvatures[radius]["gauss_curvature"]
        mean = np.mean(curv)
        norm = colors.Normalize(vmin=mean - 0.05, vmax=mean + 0.05, clip=True)
        color_map = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('plasma'))
        rgbs = color_map.to_rgba(curv)[:, :3]
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs)
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    def drawWithMeanCurv(self, radius):
        vertices = o3d.utility.Vector3dVector(self.vertices)
        faces = o3d.utility.Vector3iVector(self.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, faces)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        curv = self.vertex_curvatures[radius]["mean_curvature"]
        mean = np.mean(curv)
        norm = colors.Normalize(vmin=mean - 0.05, vmax=mean + 0.05, clip=True)
        color_map = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('plasma'))
        rgbs = color_map.to_rgba(curv)[:, :3]
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs)
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    def drawWithCurvedness(self, radius):
        vertices = o3d.utility.Vector3dVector(self.vertices)
        faces = o3d.utility.Vector3iVector(self.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, faces)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        curv = self.vertex_curvatures[radius]["curvedness"]
        mean = np.mean(curv)
        norm = colors.Normalize(vmin=mean - 0.05, vmax=mean + 0.05, clip=True)
        color_map = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('plasma'))
        rgbs = color_map.to_rgba(curv)[:, :3]
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs)
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    def drawWithK2(self, radius):
        vertices = o3d.utility.Vector3dVector(self.vertices)
        faces = o3d.utility.Vector3iVector(self.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, faces)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        curv = self.vertex_curvatures[radius]["K2"]
        mean = np.mean(curv)
        norm = colors.Normalize(vmin=mean - 0.05, vmax=mean + 0.05, clip=True)
        color_map = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('plasma'))
        rgbs = color_map.to_rgba(curv)[:, :3]
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs)
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    def loadFromMeshFile(self, mesh_path, normalize=False):
        """
        Load a mesh. Format available: .inp), ANSYS msh (.msh), AVS-UCD (.avs), CGNS (.cgns), DOLFIN XML (.xml), Exodus (.e, .exo), FLAC3D (.f3grid), H5M (.h5m), Kratos/MDPA (.mdpa), Medit (.mesh, .meshb), MED/Salome (.med), Nastran (bulk data, .bdf, .fem, .nas), Netgen (.vol, .vol.gz), Neuroglancer precomputed format, Gmsh (format versions 2.2, 4.0, and 4.1, .msh), OBJ (.obj), OFF (.off), PERMAS (.post, .post.gz, .dato, .dato.gz), PLY (.ply), STL (.stl), Tecplot .dat, TetGen .node/.ele, SVG (2D output only) (.svg), SU2 (.su2), UGRID (.ugrid), VTK (.vtk), VTU (.vtu), WKT (TIN) (.wkt), XDMF (.xdmf, .xmf).
        @param mesh_path: the path of the mesh to load
        @return: None
        """
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if normalize:
            pc = np.asarray(mesh.vertices)
            centroid = np.mean(pc, axis=0)
            pc = pc - centroid
            m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
            pc /= m
            pc = o3d.utility.Vector3dVector(pc)
            faces = o3d.utility.Vector3iVector(mesh.triangles)
            mesh = o3d.geometry.TriangleMesh(pc, faces)
        self.computeDataStructures(mesh)

    def save(self, path):
        """
        Save a mesh.
        @param path: the path with filename and extension (ex. C:/user/file/mesh_filename.pkl )
        @return: None
        """
        with open(path, "wb") as save_file:
            pickle.dump(self, save_file)
        print(f"Mesh saved at: {path}")

    def has_adjacency_lists(self):
        return self.adjacency_list is not None

    def has_curvatures(self):
        return self.vertex_curvatures

    def has_edges(self):
        return self.edges is not None


def expandFacet(facet, mesh):
    """
    Expands the facet with all the neighbour faces of the facet
    @param facet: N indices representing the facet to expand
    @param mesh: Mesh object
    @return: a list of face indices representing the expanded facet
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
        @param neighbors_level: parameter that consider boundary vertices also the n-level neighbors of a real boundary vertex
        @return:
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
        @param face_idx: index of the face to retrieve neighbours
        @return: list of indices of neighbours
        """
        return self.face_adjacency_list[face_idx]

    def neighbourFaces(self, face_idx):
        """
        @param face_idx: index of the face to retrieve neighbours or (1x3) array representing a face
        @return: neighbours faces
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
