'''
Mesher module for generating mesh data for solver from geometry.
'''
from meshpy.triangle import MeshInfo 
import jax
import jax.numpy as jnp
from utils.utils import *
from src.containers import *
from src.dynamics import *

class Mesher():

    def __init__(self):
        self.points = None
        self.cells = None # index of three points
        self.neighbors = None # index of three cells
        self.faces = None # index of two points
        self.point_markers = None # Point markers for boundary conditions

    def import_meshpy(self, mesh: MeshInfo):
        self.points = jnp.array(mesh.points)
        self.cells = jnp.array(mesh.elements)
        self.faces = jnp.array(mesh.faces)
        self.point_markers = jnp.array(mesh.point_markers)

    def calc_mesh_properties(self):
        self.calc_cell_centers()
        self.calc_face_normals()
        self.calc_face_lengths()
        self.calc_cell_face_indices_and_normals()
        self.calc_cell_face_normal_signs()
        self.calc_face_cell_indices()
        self.calc_point_cell_indices_and_distances()

    def to_env(self,dynamics = D2Q9(tau=None,delta_t=None)):
        cells = Cells(self.cells.shape[0],dynamics)
        faces = Faces(self.faces.shape[0],dynamics)
        nodes = Nodes(self.points.shape[0],dynamics)
        # Fill in the properties
        cells.face_indices = self.cell_face_indices
        cells.face_normals = self.cell_face_normal_signs

        faces.n = self.face_normals
        faces.L = self.face_lengths
        faces.nodes_index = self.faces
        faces.stencil_cells_index = self.face_cell_indices
        faces.stencil_dists = self.face_cell_center_distances

        nodes.cells_index = self.point_cell_indices
        nodes.cell_dists = self.point_cell_center_distances
        nodes.type = self.point_markers

        return cells,faces,nodes

    # Face functions
    def calc_face_normals(self):
        ''' Calculate the unit normal vector for each face defined by two points.'''
        def normal(face):
            return calc_normal(self.points[face[0]], self.points[face[1]])
        self.face_normals = jax.vmap(normal)(self.faces)

    def calc_face_lengths(self):
        ''' Calculate the length of each face defined by two points.'''
        def length(face):
            return calc_dist(self.points[face[0]], self.points[face[1]])
        self.face_lengths = jax.vmap(length)(self.faces)

    # Cell Functions
    def calc_cell_centers(self):
        ''' Calculate the centroid of each triangle cell'''
        def centroid(cell):
            p1, p2, p3 = self.points[cell[0]], self.points[cell[1]], self.points[cell[2]]
            return (p1 + p2 + p3) / 3.0
        self.cell_centers = jax.vmap(centroid)(self.cells)

    def calc_cell_face_indices_and_normals(self):
        '''
        For each cell, find the indices of its three faces and the outward normal of each face.
        '''
        # Precompute face keys for mapping: sorted tuple of point indices for each face
        face_keys = jnp.sort(self.faces, axis=1)
        # For each cell, get its three faces as sorted pairs
        def cell_faces(cell):
            return jnp.stack([
                jnp.sort(jnp.array([cell[0], cell[1]])),
                jnp.sort(jnp.array([cell[1], cell[2]])),
                jnp.sort(jnp.array([cell[2], cell[0]]))
            ])
        cell_face_keys = jax.vmap(cell_faces)(self.cells)  # (num_cells, 3, 2)

        # For each cell face, find its index in self.faces
        def find_face_indices(cell_face_keys):
            # For each face in the cell, find where it matches in face_keys
            def find_idx(face_key):
                # Compare to all face_keys, return index of match
                matches = jnp.all(face_keys == face_key, axis=1)
                return jnp.argmax(matches)
            return jax.vmap(find_idx)(cell_face_keys)
        self.cell_face_indices = jax.vmap(find_face_indices)(cell_face_keys)

        # For each cell, compute outward normals for its faces
        def cell_normals(cell, cell_center):
            faces = [
                (cell[0], cell[1]),
                (cell[1], cell[2]),
                (cell[2], cell[0])
            ]
            normals = []
            for f in faces:
                p0, p1 = self.points[f[0]], self.points[f[1]]
                normal = calc_normal(p0, p1)
                midpoint = (p0 + p1) / 2.0
                to_center = cell_center - midpoint
                normal = jnp.where(jnp.dot(normal, to_center) < 0, -normal, normal)
                normals.append(normal)
            return jnp.stack(normals)
        self.cell_face_normals = jax.vmap(cell_normals)(self.cells, self.cell_centers)

    def calc_cell_face_normal_signs(self):
        '''
        For each cell, compute the sign of the dot product between the cell's face normal
        and the global face normal for each of its faces.
        '''
        def cell_signs(cell_face_indices, cell_face_normals):
            face_normals = self.face_normals[cell_face_indices]  # (3, 2)
            dots = jnp.einsum('ij,ij->i', cell_face_normals, face_normals)
            return jnp.sign(dots)
        self.cell_face_normal_signs = jax.vmap(cell_signs)(
            self.cell_face_indices, self.cell_face_normals
        )

    def calc_face_cell_indices(self):
        '''
        For each face, find the two cells that share it, ordered so that the first cell
        is the upwind cell with respect to the face normal.
        '''
        num_faces = self.faces.shape[0]
        num_cells = self.cells.shape[0]

        # Precompute face keys for mapping: sorted tuple of point indices for each face
        face_keys = jnp.sort(self.faces, axis=1)

        # For each cell, get its three faces as sorted pairs
        def cell_faces(cell):
            return jnp.stack([
                jnp.sort(jnp.array([cell[0], cell[1]])),
                jnp.sort(jnp.array([cell[1], cell[2]])),
                jnp.sort(jnp.array([cell[2], cell[0]]))
            ])
        cell_face_keys = jax.vmap(cell_faces)(self.cells)  # (num_cells, 3, 2)

        # For each face, collect the cells that share it
        # Build a (num_faces, 2) array, fill with -1 for missing neighbors
        face_cell_array = -jnp.ones((num_faces, 2), dtype=int)
        # For each cell, for each of its faces, update face_cell_array
        def update_face_cell_array(face_cell_array, cell_idx, cell_face_keys):
            def update_one(face_cell_array, face_key):
                matches = jnp.all(face_keys == face_key, axis=1)
                face_idx = jnp.argmax(matches)
                # Find first available slot (0 or 1)
                slot0 = face_cell_array[face_idx, 0] == -1
                slot1 = face_cell_array[face_idx, 1] == -1
                face_cell_array = jax.lax.cond(
                    slot0,
                    lambda arr: arr.at[face_idx, 0].set(cell_idx),
                    lambda arr: arr,
                    face_cell_array
                )
                face_cell_array = jax.lax.cond(
                    jnp.logical_and(~slot0, slot1),
                    lambda arr: arr.at[face_idx, 1].set(cell_idx),
                    lambda arr: arr,
                    face_cell_array
                )
                return face_cell_array
            face_cell_array = jax.lax.fori_loop(
                0, 3, lambda i, arr: update_one(arr, cell_face_keys[i]), face_cell_array
            )
            return face_cell_array

        # Use a scan to update face_cell_array for all cells
        def body_fun(carry, x):
            cell_idx, cell_face_keys = x
            carry = update_face_cell_array(carry, cell_idx, cell_face_keys)
            return carry, None
        xs = (jnp.arange(num_cells), cell_face_keys)
        face_cell_array, _ = jax.lax.scan(body_fun, face_cell_array, xs, length=num_cells)

        # For each face, order the two cells and compute distances
        def order_cells_and_distances(face, face_normal, cell_indices):
            midpoint = (self.points[face[0]] + self.points[face[1]]) / 2.0
            c0, c1 = cell_indices
            dists = jnp.array([-1.0, -1.0])
            # Use jax.lax.cond for control flow
            def boundary_case(_):
                c = jnp.where(c0 != -1, c0, c1)
                center = self.cell_centers[c]
                d = jnp.linalg.norm(midpoint - center)
                return (jnp.array([c, -1]), dists.at[0].set(d))
            def interior_case(_):
                center0 = self.cell_centers[c0]
                center1 = self.cell_centers[c1]
                v0 = midpoint - center0
                v1 = midpoint - center1
                dot0 = jnp.dot(face_normal, v0)
                dot1 = jnp.dot(face_normal, v1)
                d0 = jnp.linalg.norm(v0)
                d1 = jnp.linalg.norm(v1)
                cond = dot0 > dot1
                idxs = jnp.where(cond, jnp.array([c0, c1]), jnp.array([c1, c0]))
                dists = jnp.where(cond, jnp.array([d0, d1]), jnp.array([d1, d0]))
                return (idxs, dists)
            is_boundary = (c0 == -1) | (c1 == -1)
            return jax.lax.cond(is_boundary, boundary_case, interior_case, operand=None)

        results = jax.vmap(order_cells_and_distances)(
            self.faces, self.face_normals, face_cell_array
        )
        self.face_cell_indices = results[0]
        self.face_cell_center_distances = results[1]

    # Node functions
    def calc_point_cell_indices_and_distances(self):
        '''
        For each point, find all cells which touch it and the distance from the point to the cell center.
        Fill unused slots with -1.
        '''
        num_points = self.points.shape[0]
        num_cells = self.cells.shape[0]

        # For each point, collect the indices of cells that contain it
        # Build a list of lists, then pad to max length
        point_cell_lists = [[] for _ in range(num_points)]
        for cell_idx, cell in enumerate(self.cells):
            for pt in cell:
                point_cell_lists[pt].append(cell_idx)
        max_len = max(len(lst) for lst in point_cell_lists)
        # Pad with -1
        point_cell_indices = -jnp.ones((num_points, max_len), dtype=int)
        for i, lst in enumerate(point_cell_lists):
            point_cell_indices = point_cell_indices.at[i, :len(lst)].set(jnp.array(lst, dtype=int))

        # For each point and its associated cells, compute distances
        def point_cell_distances(point_idx, cell_indices):
            point = self.points[point_idx]
            dists = -jnp.ones(cell_indices.shape, dtype=float)
            for j, cell_idx in enumerate(cell_indices):
                if cell_idx != -1:
                    center = self.cell_centers[cell_idx]
                    dists = dists.at[j].set(jnp.linalg.norm(point - center))
            return dists

        point_cell_distances_arr = jnp.stack([
            point_cell_distances(i, point_cell_indices[i]) for i in range(num_points)
        ])

        self.point_cell_indices = point_cell_indices
        self.point_cell_center_distances = point_cell_distances_arr