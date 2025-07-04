'''
Mesher module for generating mesh data for solver from geometry.
'''
import jax
import jax.numpy as jnp
import numpy as np
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

    def import_meshpy(self, mesh):
        self.points = np.array(mesh.points,dtype=np.float64)
        self.cells = np.array(mesh.elements,dtype=np.int32)
        self.faces = np.array(mesh.faces,dtype=np.int32)
        self.point_markers = np.array(mesh.point_markers,dtype=np.int32)
        self.enforce_ccw()  # Ensure all triangles are CCW

    def enforce_ccw(self):
        '''
        Ensure all triangle cells are counterclockwise (CCW) ordered.
        If a cell is not CCW, swap two vertices to make it CCW.
        '''
        def signed_area(tri):
            p = self.points[tri]
            return 0.5 * ((p[1,0] - p[0,0]) * (p[2,1] - p[0,1]) - (p[2,0] - p[0,0]) * (p[1,1] - p[0,1]))
        new_cells = []
        for tri in self.cells:
            if signed_area(tri) < 0:
                # Swap two vertices to make CCW
                new_cells.append([tri[0], tri[2], tri[1]])
            else:
                new_cells.append(tri)
        self.cells = np.array(new_cells, dtype=self.cells.dtype)

    def enforce_boundary_face_normals_outward(self):
        '''
        Ensure all boundary face normals point outward from the domain.
        For each boundary face (face with only one adjacent cell),
        check if the normal points outward from the cell. If not, flip the face node order.
        '''
        # First, build a mapping from face to adjacent cells
        face_keys = np.sort(self.faces, axis=1)
        def cell_faces(cell):
            return np.stack([
                np.sort(np.array([cell[0], cell[1]])),
                np.sort(np.array([cell[1], cell[2]])),
                np.sort(np.array([cell[2], cell[0]]))
            ])
        cell_face_keys = np.array([cell_faces(cell) for cell in self.cells])
        face_cell_count = np.zeros(len(self.faces), dtype=int)
        for cell_face_keys_ in cell_face_keys:
            for face_key in cell_face_keys_:
                matches = np.all(face_keys == face_key, axis=1)
                face_idx = np.argmax(matches)
                face_cell_count[face_idx] += 1
        # For each boundary face, check normal direction
        for i, count in enumerate(face_cell_count):
            if count == 1:
                # Find the cell that owns this face
                face = self.faces[i]
                # Find the cell index
                cell_idx = -1
                for idx, cell in enumerate(self.cells):
                    cell_faces_ = cell_faces(cell)
                    for f in cell_faces_:
                        if np.all(np.sort(face) == f):
                            cell_idx = idx
                            break
                    if cell_idx != -1:
                        break
                if cell_idx == -1:
                    continue  # Should not happen
                # Compute normal
                p0, p1 = self.points[face[0]], self.points[face[1]]
                normal = calc_normal(p0, p1)
                cell_center = self.cell_centers[cell_idx]
                midpoint = (p0 + p1) / 2.0
                to_center = cell_center - midpoint
                # If normal points inward, flip face order
                if np.dot(normal, to_center) < 0:
                    continue  # Already outward
                else:
                    # Flip face node order
                    self.faces[i] = [face[1], face[0]]
        # After this, all boundary face normals will point outward

    def calc_face_centers(self):
        '''Calculate the center (midpoint) of each face.'''
        self.face_centers = np.array([
            (self.points[face[0]] + self.points[face[1]]) / 2.0
            for face in self.faces
        ])

    def calc_mesh_properties(self):
        print("Calculating cell centers...")
        self.calc_cell_centers()
        print("Cell centers calculated.")

        # Ensure boundary face normals are outward after cell centers are available
        self.enforce_boundary_face_normals_outward()

        print("Calculating face centers...")
        self.calc_face_centers()
        print("Face centers calculated.")

        print("Calculating face normals...")
        self.calc_face_normals()
        print("Face normals calculated.")

        print("Calculating face lengths...")
        self.calc_face_lengths()
        print("Face lengths calculated.")

        print("Calculating cell face indices and normals...")
        self.calc_cell_face_indices_and_normals()
        print("Cell face indices and normals calculated.")

        print("Calculating cell face normal signs...")
        self.calc_cell_face_normal_signs()
        print("Cell face normal signs calculated.")

        print("Calculating face cell indices...")
        self.calc_face_cell_indices()
        print("Face cell indices calculated.")

        print("Calculating face ghost distances...")
        self.calc_face_ghost_distances()
        print("Face ghost distances calculated.")

        print("Calculating point cell indices and distances...")
        self.calc_point_cell_indices_and_distances()
        print("Point cell indices and distances calculated.")

    def to_env(self, dynamics):
        cells = Cells(self.cells.shape[0], dynamics)
        faces = Faces(self.faces.shape[0], dynamics)
        nodes = Nodes(self.points.shape[0], dynamics)
        # Convert all arrays to jax arrays before assignment
        cells.face_indices = jnp.array(self.cell_face_indices,dtype=jnp.int32)
        cells.face_normals = jnp.array(self.cell_face_normal_signs,dtype=jnp.int32)

        faces.n = jnp.array(self.face_normals,dtype=jnp.float64)
        faces.L = jnp.array(self.face_lengths,dtype=jnp.float64)[...,jnp.newaxis] * 50
        faces.nodes_index = jnp.array(self.faces,dtype=jnp.int32)
        faces.stencil_cells_index = jnp.array(self.face_cell_indices,dtype=jnp.int32)
        faces.stencil_dists = jnp.array(self.face_cell_center_distances,dtype=jnp.float64)*50

        nodes.cells_index = jnp.array(self.point_cell_indices,dtype=jnp.int32)
        nodes.cell_dists = jnp.array(self.point_cell_center_distances,dtype=jnp.float64)
        nodes.cell_dists = nodes.cell_dists.at[jnp.where(nodes.cell_dists > 0)].set(nodes.cell_dists[jnp.where(nodes.cell_dists > 0)] * 50)
        nodes.type = jnp.zeros_like(self.point_markers[..., np.newaxis], dtype=jnp.int32)

        # The following lines assume self.points is a numpy array, so convert to jax for indexing
        points_jax = jnp.array(self.points)
        nodes.type = nodes.type.at[points_jax[:, 1] == 2].set(1)
        nodes.type = nodes.type.at[points_jax[:, 1] == 0].set(1)
        nodes.type = nodes.type.at[points_jax[:, 0] == 2].set(1)
        nodes.type = nodes.type.at[points_jax[:, 0] == 0].set(1)

        nodes.vel = nodes.vel.at[points_jax[:,1] == 2].set(jnp.array([.1, 0]))
        nodes.vel = nodes.vel.at[points_jax[:,0] == 2].set(jnp.array([0, 0]))
        nodes.vel = nodes.vel.at[points_jax[:,0] == 0].set(jnp.array([0, 0]))


        return cells, faces, nodes

    # Face functions
    def calc_face_normals(self):
        ''' Calculate the unit normal vector for each face defined by two points.'''
        def normal(face):
            return calc_normal(self.points[face[0]], self.points[face[1]])
        self.face_normals = np.array([normal(face) for face in self.faces])

    def calc_face_lengths(self):
        ''' Calculate the length of each face defined by two points.'''
        def length(face):
            return calc_dist(self.points[face[0]], self.points[face[1]])
        self.face_lengths = np.array([length(face) for face in self.faces])

    # Cell Functions
    def calc_cell_centers(self):
        ''' Calculate the centroid of each triangle cell'''
        def centroid(cell):
            p1, p2, p3 = self.points[cell[0]], self.points[cell[1]], self.points[cell[2]]
            return (p1 + p2 + p3) / 3.0
        self.cell_centers = np.array([centroid(cell) for cell in self.cells])

    def calc_cell_face_indices_and_normals(self):
        '''
        For each cell, find the indices of its three faces and the outward normal of each face.
        Assumes all cells are CCW ordered (enforced by enforce_ccw).
        '''
        face_keys = np.sort(self.faces, axis=1)
        def cell_faces(cell):
            return np.stack([
                np.sort(np.array([cell[0], cell[1]])),
                np.sort(np.array([cell[1], cell[2]])),
                np.sort(np.array([cell[2], cell[0]]))
            ])
        cell_face_keys = np.array([cell_faces(cell) for cell in self.cells])  # (num_cells, 3, 2)

        def find_face_indices(cell_face_keys):
            def find_idx(face_key):
                matches = np.all(face_keys == face_key, axis=1)
                return np.argmax(matches)
            return np.array([find_idx(face_key) for face_key in cell_face_keys])
        self.cell_face_indices = np.array([find_face_indices(keys) for keys in cell_face_keys])

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
                # For CCW triangles, the outward normal is when dot(normal, to_center) < 0
                if np.dot(normal, to_center) < 0:
                    normals.append(normal)
                else:
                    normals.append(-normal)
            return np.stack(normals)
        self.cell_face_normals = np.array([cell_normals(cell, center) for cell, center in zip(self.cells, self.cell_centers)])

    def calc_cell_face_normal_signs(self):
        '''
        For each cell, compute the sign of the dot product between the cell's face normal
        and the global face normal for each of its faces.
        '''
        def cell_signs(cell_face_indices, cell_face_normals):
            face_normals = self.face_normals[cell_face_indices]  # (3, 2)
            dots = np.einsum('ij,ij->i', cell_face_normals, face_normals)
            return np.sign(dots)
        self.cell_face_normal_signs = np.array([cell_signs(indices, normals) for indices, normals in zip(self.cell_face_indices, self.cell_face_normals)],dtype=np.int32)

    def calc_face_cell_indices(self):
        '''
        For each face, find the two cells that share it, ordered so that the first cell
        is the upwind cell with respect to the face normal.
        Assumes all cells are CCW ordered and face normals are outward.
        '''
        num_faces = self.faces.shape[0]
        face_keys = np.sort(self.faces, axis=1)
        def cell_faces(cell):
            return np.stack([
                np.sort(np.array([cell[0], cell[1]])),
                np.sort(np.array([cell[1], cell[2]])),
                np.sort(np.array([cell[2], cell[0]]))
            ])
        cell_face_keys = np.array([cell_faces(cell) for cell in self.cells])  # (num_cells, 3, 2)

        face_cell_array = -np.ones((num_faces, 2), dtype=int)
        for cell_idx, cell_face_keys_ in enumerate(cell_face_keys):
            for face_key in cell_face_keys_:
                matches = np.all(face_keys == face_key, axis=1)
                face_idx = np.argmax(matches)
                if face_cell_array[face_idx, 0] == -1:
                    face_cell_array[face_idx, 0] = cell_idx
                elif face_cell_array[face_idx, 1] == -1:
                    face_cell_array[face_idx, 1] = cell_idx

        def order_cells_and_distances(face, face_normal, cell_indices):
            midpoint = (self.points[face[0]] + self.points[face[1]]) / 2.0
            c0, c1 = cell_indices
            dists = np.array([-1.0, -1.0])
            if c0 == -1 or c1 == -1:
                c = c0 if c0 != -1 else c1
                center = self.cell_centers[c]
                dist = np.dot(face_normal, center - midpoint)
                if c0 != -1:
                    dists[0] = dist
                else:
                    dists[1] = dist
                return np.array([c0, c1]), dists
            else:
                center0 = self.cell_centers[c0]
                center1 = self.cell_centers[c1]
                v0 = center0 - midpoint
                v1 = center1 - midpoint
                d0 = np.dot(face_normal, v0)
                d1 = np.dot(face_normal, v1)
                # For CCW, outward normal, upwind cell is the one with d < 0
                if d0 < d1:
                    idxs = np.array([c0, c1])
                    dists = np.array([d0, d1])
                else:
                    idxs = np.array([c1, c0])
                    dists = np.array([d1, d0])
                return idxs, dists

        results = [order_cells_and_distances(face, normal, cell_indices)
                   for face, normal, cell_indices in zip(self.faces, self.face_normals, face_cell_array)]
        self.face_cell_indices = np.array([r[0] for r in results])
        self.face_cell_center_distances = np.abs(np.array([r[1] for r in results]))

    def calc_face_ghost_distances(self):
        '''
        For each face with a ghost cell (where one of the stencil indices is -1),
        set the ghost cell's stencil distance equal to the non-ghost cell's stencil distance.
        '''
        dists = self.face_cell_center_distances.copy()
        indices = self.face_cell_indices
        mask0 = indices[:, 0] == -1
        mask1 = indices[:, 1] == -1
        dists[mask0, 0] = dists[mask0, 1]
        dists[mask1, 1] = dists[mask1, 0]
        self.face_cell_center_distances = dists

    # Node functions
    def calc_point_cell_indices_and_distances(self):
        '''
        For each point, find all cells which touch it and the distance from the point to the cell center.
        Fill unused slots with -1.
        '''
        num_points = self.points.shape[0]
        num_cells = self.cells.shape[0]
        point_cell_lists = [[] for _ in range(num_points)]
        for cell_idx, cell in enumerate(self.cells):
            for pt in cell:
                point_cell_lists[pt].append(cell_idx)
        max_len = max(len(lst) for lst in point_cell_lists)
        point_cell_indices = -np.ones((num_points, max_len), dtype=int)
        for i, lst in enumerate(point_cell_lists):
            point_cell_indices[i, :len(lst)] = np.array(lst, dtype=int)

        def point_cell_distances(point_idx, cell_indices):
            point = self.points[point_idx]
            dists = -np.ones(cell_indices.shape, dtype=float)
            for j, cell_idx in enumerate(cell_indices):
                if cell_idx != -1:
                    center = self.cell_centers[cell_idx]
                    dists[j] = np.linalg.norm(point - center)
            return dists

        point_cell_distances_arr = np.stack([
            point_cell_distances(i, point_cell_indices[i]) for i in range(num_points)
        ])

        self.point_cell_indices = point_cell_indices
        self.point_cell_center_distances = point_cell_distances_arr

    def verify_stencil_geometry(self):
        bad_angle_count = 0
        bad_distance_count = 0
        angles = []
        distance_errors = []
        offsets = []
        center_dist_vs_face_dist = []
        for i, (face, n, cells, dists) in enumerate(zip(
                self.faces, self.face_normals, self.face_cell_indices, self.face_cell_center_distances)):
            c0, c1 = cells
            if c0 == -1 or c1 == -1:
                continue  # skip boundary/ghost faces
            center0 = self.cell_centers[c0]
            center1 = self.cell_centers[c1]
            v = center1 - center0
            v_norm = v / np.linalg.norm(v)
            n_norm = n / np.linalg.norm(n)
            cos_theta = np.dot(v_norm, n_norm)
            angle = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi
            angles.append(angle)
            if abs(angle) > 30:  # threshold for bad angle
                bad_angle_count += 1

            # Projected distance check
            d_proj = np.dot(center1 - center0, n_norm)
            d_sum = dists[0] + dists[1]
            distance_error = np.abs(d_proj - d_sum)
            distance_errors.append(distance_error)
            if distance_error > 1e-2:  # threshold for bad distance
                bad_distance_count += 1

            # Face center offset check
            face_center = self.face_centers[i]
            midpoint = (center0 + center1) / 2.0
            offset = np.linalg.norm(face_center - midpoint)
            offsets.append(offset)
            if offset > 1e-2:
                print(f"Warning: Face {i} center offset from cell-center midpoint is {offset:.4e}")

            # Track cell center distance vs face distance sum
            center_dist = np.linalg.norm(center1 - center0)
            face_dist_sum = np.abs(dists[0]) + np.abs(dists[1])
            center_dist_vs_face_dist.append(center_dist - face_dist_sum)

        print(f"Mean angle (deg): {np.mean(angles):.2f}, max: {np.max(angles):.2f}")
        print(f"Faces with angle > 30 deg: {bad_angle_count}")
        print(f"Mean distance error: {np.mean(distance_errors):.4f}, max: {np.max(distance_errors):.4f}")
        print(f"Faces with distance error > 1e-2: {bad_distance_count}")
        print(f"Mean face center offset: {np.mean(offsets):.4e}, max: {np.max(offsets):.4e}")
        print(f"Mean (cell center dist - face dist sum): {np.mean(center_dist_vs_face_dist):.4e}, max: {np.max(center_dist_vs_face_dist):.4e}, min: {np.min(center_dist_vs_face_dist):.4e}")
