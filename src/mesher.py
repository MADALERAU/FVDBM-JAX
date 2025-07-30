'''
Mesher module for generating mesh data for solver from geometry.

This module provides the Mesher class for constructing, analyzing, and verifying 2D triangular meshes for finite volume solvers.
It includes routines for mesh import, geometric property calculation, face/cell/point connectivity, mesh quality verification,
and stencil geometry analysis. All docstrings are compatible with pdoc documentation generation.
'''
import jax
import jax.numpy as jnp
import numpy as np
import time
from utils.utils import *
from src.containers import *
from src.dynamics import *
from src.faces import *

class Mesher():
    '''
    Mesher constructs and analyzes 2D triangular meshes for finite volume solvers.

    Attributes
    ----------
    points : ndarray
        Array of mesh point coordinates.
    cells : ndarray
        Array of triangle vertex indices.
    neighbors : ndarray
        Array of neighboring cell indices for each cell.
    faces : ndarray
        Array of face vertex indices.
    point_markers : ndarray
        Point markers for boundary conditions.
    '''
    def __init__(self):
        '''
        Initialize an empty Mesher object.
        '''
        self.points = None
        self.cells = None # index of three points
        self.neighbors = None # index of three cells
        self.faces = None # index of two points
        self.point_markers = None # Point markers for boundary conditions

    # --- Mesh import and orientation ---
    def import_meshpy(self, mesh):
        '''
        Import mesh data from a meshpy.triangle mesh object.

        Parameters
        ----------
        mesh : meshpy.triangle.MeshInfo
            Mesh object to import.
        '''
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
        # Build a mapping from sorted face (tuple) to owning cell index
        face_to_cell = {}
        for cell_idx, cell in enumerate(self.cells):
            for i in range(3):
                face_tuple = tuple(sorted((cell[i], cell[(i+1)%3])))
                face_to_cell.setdefault(face_tuple, []).append(cell_idx)
        # For each face, check if it's a boundary face (only one adjacent cell)
        for i, face in enumerate(self.faces):
            face_tuple = tuple(sorted(face))
            cell_indices = face_to_cell.get(face_tuple, [])
            if len(cell_indices) == 1:
                cell_idx = cell_indices[0]
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

    # --- Cell geometric properties ---
    def calc_cell_centers(self):
        '''
        Calculate the centroid of each triangle cell.
        '''
        def centroid(cell):
            p1, p2, p3 = self.points[cell[0]], self.points[cell[1]], self.points[cell[2]]
            return (p1 + p2 + p3) / 3.0
        self.cell_centers = np.array([centroid(cell) for cell in self.cells])

    def calc_cell_face_indices_and_normals(self):
        '''
        For each cell, find the indices of its three faces and the outward normal of each face.
        Assumes all cells are CCW ordered (enforced by enforce_ccw).
        Optimized for speed using a face-key-to-index dictionary.
        '''
        # Build a mapping from sorted face (tuple) to face index for O(1) lookup
        face_key_to_index = {tuple(sorted(face)): i for i, face in enumerate(self.faces)}
        def cell_faces(cell):
            return [tuple(sorted([cell[0], cell[1]])),
                    tuple(sorted([cell[1], cell[2]])),
                    tuple(sorted([cell[2], cell[0]]))]
        # For each cell, get the indices of its three faces
        self.cell_face_indices = np.array([
            [face_key_to_index[fk] for fk in cell_faces(cell)]
            for cell in self.cells
        ], dtype=int)

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

    # --- Face geometric properties ---
    def calc_face_centers(self):
        '''
        Calculate the center (midpoint) of each face.
        '''
        self.face_centers = np.array([
            (self.points[face[0]] + self.points[face[1]]) / 2.0
            for face in self.faces
        ])

    def calc_face_normals(self):
        '''
        Calculate the unit normal vector for each face defined by two points.
        '''
        def normal(face):
            return calc_normal(self.points[face[0]], self.points[face[1]])
        self.face_normals = np.array([normal(face) for face in self.faces])

    def calc_face_lengths(self):
        '''
        Calculate the length of each face defined by two points.
        '''
        def length(face):
            return calc_dist(self.points[face[0]], self.points[face[1]])
        self.face_lengths = np.array([length(face) for face in self.faces])

    def calc_face_cell_indices(self):
        '''
        For each face, find the two cells that share it, ordered so that the first cell
        is the upwind cell with respect to the face normal.
        Also computes cc_stencil_dist: the cell-to-face/cell-to-cell distances projected along the stencil norm.
        Assumes all cells are CCW ordered and face normals are outward.
        Optimized for speed using a face-key-to-cell mapping.
        '''
        num_faces = self.faces.shape[0]
        # Build a mapping from face key (tuple of sorted indices) to list of cell indices
        face_key_to_cells = {}
        for cell_idx, cell in enumerate(self.cells):
            for i in range(3):
                face_key = tuple(sorted([cell[i], cell[(i+1)%3]]))
                face_key_to_cells.setdefault(face_key, []).append(cell_idx)
        # For each face, get the two cells that share it
        face_cell_array = -np.ones((num_faces, 2), dtype=int)
        for i, face in enumerate(self.faces):
            face_key = tuple(sorted(face))
            cells = face_key_to_cells.get(face_key, [])
            if len(cells) > 0:
                face_cell_array[i, 0] = cells[0]
            if len(cells) > 1:
                face_cell_array[i, 1] = cells[1]
        # Order cells and distances as before
        def order_cells_and_distances(face, face_normal, cell_indices, stencil_norm):
            midpoint = (self.points[face[0]] + self.points[face[1]]) / 2.0
            c0, c1 = cell_indices
            dists = np.array([-1.0, -1.0])
            cc_stencil = np.array([-1.0, -1.0])
            if c0 == -1 or c1 == -1:
                c = c0 if c0 != -1 else c1
                center = self.cell_centers[c]
                dist = np.dot(face_normal, center - midpoint)
                dist_cc = np.dot(stencil_norm, center - midpoint)
                if c0 != -1:
                    dists[0] = dist
                    cc_stencil[0] = dist_cc
                else:
                    dists[1] = dist
                    cc_stencil[1] = dist_cc
                return np.array([c0, c1]), dists, cc_stencil
            else:
                center0 = self.cell_centers[c0]
                center1 = self.cell_centers[c1]
                v0 = center0 - midpoint
                v1 = center1 - midpoint
                d0 = np.dot(face_normal, v0)
                d1 = np.dot(face_normal, v1)
                d0_cc = np.dot(stencil_norm, v0)
                d1_cc = np.dot(stencil_norm, v1)
                # For CCW, outward normal, upwind cell is the one with d < 0
                if d0 < d1:
                    idxs = np.array([c0, c1])
                    dists = np.array([d0, d1])
                    cc_stencil = np.array([d0_cc, d1_cc])
                else:
                    idxs = np.array([c1, c0])
                    dists = np.array([d1, d0])
                    cc_stencil = np.array([d1_cc, d0_cc])
                return idxs, dists, cc_stencil
        # Calculate stencil norms first if not present
        if not hasattr(self, 'stencil_norms'):
            self.calc_stencil_norms()
        results = [order_cells_and_distances(face, normal, cell_indices, stencil_norm)
                   for face, normal, cell_indices, stencil_norm in zip(
                        self.faces, self.face_normals, face_cell_array, self.stencil_norms)]
        self.face_cell_indices = np.array([r[0] for r in results])
        self.face_cell_center_distances = np.abs(np.array([r[1] for r in results]))
        self.cc_stencil_dist = np.abs(np.array([r[2] for r in results]))

    def calc_face_ghost_distances(self):
        '''
        For each face with a ghost cell (where one of the stencil indices is -1),
        set the ghost cell's stencil distance equal to the non-ghost cell's stencil distance.
        '''
        dists = self.face_cell_center_distances.copy()
        cc_dists = self.cc_stencil_dist.copy()
        indices = self.face_cell_indices
        mask0 = indices[:, 0] == -1
        mask1 = indices[:, 1] == -1
        dists[mask0, 0] = dists[mask0, 1]
        dists[mask1, 1] = dists[mask1, 0]
        cc_dists[mask0, 0] = cc_dists[mask0, 1]
        cc_dists[mask1, 1] = cc_dists[mask1, 0]
        self.face_cell_center_distances = dists
        self.cc_stencil_dist = cc_dists

    # --- Node/cell-to-point connectivity ---
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

    # --- Mesh property calculation and conversion ---
    def calc_mesh_properties(self):
        '''
        Calculate all geometric properties and connectivity for the mesh, including cell centers, face normals,
        face lengths, cell-face connectivity, and stencil geometry. Times and reports each step.
        '''
        print("Calculating cell centers...")
        t0 = time.time()
        self.calc_cell_centers()
        print(f"Cell centers calculated. (Elapsed: {time.time()-t0:.3f}s)")

        # Ensure boundary face normals are outward after cell centers are available
        print("Enforcing boundary face normals outward...")
        t0 = time.time()
        self.enforce_boundary_face_normals_outward()
        print(f"Boundary face normals enforced. (Elapsed: {time.time()-t0:.3f}s)")

        print("Calculating face centers...")
        t0 = time.time()
        self.calc_face_centers()
        print(f"Face centers calculated. (Elapsed: {time.time()-t0:.3f}s)")

        print("Calculating face normals...")
        t0 = time.time()
        self.calc_face_normals()
        print(f"Face normals calculated. (Elapsed: {time.time()-t0:.3f}s)")

        print("Calculating face lengths...")
        t0 = time.time()
        self.calc_face_lengths()
        print(f"Face lengths calculated. (Elapsed: {time.time()-t0:.3f}s)")

        print("Calculating cell face indices and normals...")
        t0 = time.time()
        self.calc_cell_face_indices_and_normals()
        print(f"Cell face indices and normals calculated. (Elapsed: {time.time()-t0:.3f}s)")

        print("Calculating cell face normal signs...")
        t0 = time.time()
        self.calc_cell_face_normal_signs()
        print(f"Cell face normal signs calculated. (Elapsed: {time.time()-t0:.3f}s)")

        print("Calculating stencil norms (cell-to-cell/cell-to-face)...")
        t0 = time.time()
        self.calc_stencil_norms()
        print(f"Stencil norms calculated. (Elapsed: {time.time()-t0:.3f}s)")

        print("Calculating face cell indices...")
        t0 = time.time()
        self.calc_face_cell_indices()
        print(f"Face cell indices calculated. (Elapsed: {time.time()-t0:.3f}s)")

        print("Calculating face ghost distances...")
        t0 = time.time()
        self.calc_face_ghost_distances()
        print(f"Face ghost distances calculated. (Elapsed: {time.time()-t0:.3f}s)")

        print("Calculating point cell indices and distances...")
        t0 = time.time()
        self.calc_point_cell_indices_and_distances()
        print(f"Point cell indices and distances calculated. (Elapsed: {time.time()-t0:.3f}s)")

        print("Calculating face-stencil angles (radians)...")
        t0 = time.time()
        self.calc_face_stencil_angles()
        print(f"Face-stencil angles calculated. (Elapsed: {time.time()-t0:.3f}s)")

    def to_env(self, dynamics, flux_method="upwind"):
        '''
        Convert mesh data to solver environment containers (Cells, Faces, Nodes).

        Parameters
        ----------
        dynamics : object
            Dynamics object for the solver.

        flux_method : str, optional
            The flux method to use for the solver. Defaults to "upwind".
            options: "upwind", "cc".

        Returns
        -------
        tuple
            (cells, faces, nodes) environment containers.
        '''
        cells = Cells(self.cells.shape[0], dynamics)
        # Convert all arrays to jax arrays before assignment
        cells.face_indices = jnp.array(self.cell_face_indices,dtype=jnp.int32)
        cells.face_normals = jnp.array(self.cell_face_normal_signs,dtype=jnp.int32)

        match flux_method:
            case "upwind":
                faces = Faces(self.faces.shape[0], dynamics)

                faces.n = jnp.array(self.face_normals,dtype=jnp.float64)
                faces.L = jnp.array(self.face_lengths,dtype=jnp.float64)[...,jnp.newaxis] * 50
                faces.nodes_index = jnp.array(self.faces,dtype=jnp.int32)
                faces.stencil_cells_index = jnp.array(self.face_cell_indices,dtype=jnp.int32)
                faces.stencil_dists = jnp.array(self.face_cell_center_distances,dtype=jnp.float64)*50
            case "lax_wendroff":
                faces = Faces(self.faces.shape[0], dynamics,flux_scheme='lax_wendroff')

                faces.n = jnp.array(self.face_normals,dtype=jnp.float64)
                faces.L = jnp.array(self.face_lengths,dtype=jnp.float64)[...,jnp.newaxis] * 50
                faces.nodes_index = jnp.array(self.faces,dtype=jnp.int32)
                faces.stencil_cells_index = jnp.array(self.face_cell_indices,dtype=jnp.int32)
                faces.stencil_dists = jnp.array(self.face_cell_center_distances,dtype=jnp.float64)*50
            case "cc_upwind":
                faces = CCStencilFaces(self.faces.shape[0], dynamics)

                faces.n = jnp.array(self.stencil_norms,dtype=jnp.float64)
                faces.alpha = jnp.array(self.face_stencil_angles,dtype=jnp.float64)[...,jnp.newaxis]
                faces.L = jnp.array(self.face_lengths,dtype=jnp.float64)[...,jnp.newaxis] * 50
                faces.nodes_index = jnp.array(self.faces,dtype=jnp.int32)
                faces.stencil_cells_index = jnp.array(self.face_cell_indices,dtype=jnp.int32)
                # Use the stencil-based distances for the stencil scheme
                faces.stencil_dists = jnp.array(self.cc_stencil_dist,dtype=jnp.float64)*50
            case "cc_lax_wendroff":
                faces = CCStencilFaces(self.faces.shape[0], dynamics,flux_scheme = 'lax_wendroff')

                faces.n = jnp.array(self.stencil_norms,dtype=jnp.float64)
                faces.alpha = jnp.array(self.face_stencil_angles,dtype=jnp.float64)[...,jnp.newaxis]
                faces.L = jnp.array(self.face_lengths,dtype=jnp.float64)[...,jnp.newaxis] * 50
                faces.nodes_index = jnp.array(self.faces,dtype=jnp.int32)
                faces.stencil_cells_index = jnp.array(self.face_cell_indices,dtype=jnp.int32)
                # Use the stencil-based distances for the stencil scheme
                faces.stencil_dists = jnp.array(self.cc_stencil_dist,dtype=jnp.float64)*50
            case _:
                raise ValueError(f"Unsupported flux method: {flux_method}")

        nodes = Nodes(self.points.shape[0], dynamics)

        nodes.cells_index = jnp.array(self.point_cell_indices,dtype=jnp.int32)
        nodes.cell_dists = jnp.array(self.point_cell_center_distances,dtype=jnp.float64)
        nodes.cell_dists = nodes.cell_dists.at[jnp.where(nodes.cell_dists > 0)].set(nodes.cell_dists[jnp.where(nodes.cell_dists > 0)] * 50)
        nodes.type = jnp.zeros_like(self.point_markers[..., np.newaxis], dtype=jnp.int32)

        # Temporary: Set node types based on point coordinates
        # This is a placeholder and should be replaced with actual logic for setting node types
        points_jax = jnp.array(self.points)
        nodes.type = nodes.type.at[points_jax[:, 1] == 2].set(1)
        nodes.type = nodes.type.at[points_jax[:, 1] == 0].set(1)
        nodes.type = nodes.type.at[points_jax[:, 0] == 2].set(1)
        nodes.type = nodes.type.at[points_jax[:, 0] == 0].set(1)

        nodes.vel = nodes.vel.at[points_jax[:,1] == 2].set(jnp.array([.1, 0]))
        nodes.vel = nodes.vel.at[points_jax[:,0] == 2].set(jnp.array([0, 0]))
        nodes.vel = nodes.vel.at[points_jax[:,0] == 0].set(jnp.array([0, 0]))


        return cells, faces, nodes

    # --- Mesh quality and stencil verification ---
    def verify_stencil_geometry(self):
        '''
        Verify mesh stencil geometry by checking face normal alignment, distance errors, face center offsets,
        and consistency between face-normal and stencil-normal projected distances.

        Reports summary statistics and counts of problematic faces, including:
        - Angle between cell center vector and face normal
        - Projected distance error (face normal)
        - Face center offset
        - Difference and sign mismatch between face-normal and stencil-normal projected distances
        '''
        bad_angle_count = 0
        bad_distance_count = 0
        bad_offset_count = 0
        angles = []
        distance_errors = []
        offsets = []
        center_dist_vs_face_dist = []
        percent_distance_errors = []
        percent_offsets = []
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

            # Projected distance check (face normal)
            d_proj = np.dot(center1 - center0, n_norm)
            d_sum = dists[0] + dists[1]
            distance_error = np.abs(d_proj - d_sum)
            face_length = self.face_lengths[i]
            percent_distance_error = 100.0 * distance_error / face_length if face_length > 0 else 0.0
            distance_errors.append(distance_error)
            percent_distance_errors.append(percent_distance_error)
            if percent_distance_error > 1.0:  # threshold for bad percent error
                bad_distance_count += 1

            # Face center offset check
            face_center = self.face_centers[i]
            midpoint = (center0 + center1) / 2.0
            offset = np.linalg.norm(face_center - midpoint)
            percent_offset = 100.0 * offset / face_length if face_length > 0 else 0.0
            offsets.append(offset)
            percent_offsets.append(percent_offset)
            if percent_offset > 10.0:
                bad_offset_count += 1

            # Track cell center distance vs face distance sum (percent)
            center_dist = np.linalg.norm(center1 - center0)
            face_dist_sum = np.abs(dists[0]) + np.abs(dists[1])
            percent_center_vs_face = 100.0 * (center_dist - face_dist_sum) / face_length if face_length > 0 else 0.0
            center_dist_vs_face_dist.append(percent_center_vs_face)

        # --- New: Stencil distance checks summary (percentages) ---
        stencil_dist_diffs = []
        percent_stencil_dist_diffs = []
        stencil_dist_sign_mismatches = []
        stencil_dist_large_diffs = []
        stencil_dist_diff_threshold = 1.0  # percent
        for i, (face, n, cells, dists, cc_stencil) in enumerate(zip(
                self.faces, self.face_normals, self.face_cell_indices, self.face_cell_center_distances, self.cc_stencil_dist)):
            c0, c1 = cells
            if c0 == -1 or c1 == -1:
                continue  # skip boundary/ghost faces
            center0 = self.cell_centers[c0]
            center1 = self.cell_centers[c1]
            midpoint = (center0 + center1) / 2.0
            n_norm = n / np.linalg.norm(n)
            stencil_norm = self.stencil_norms[i] / np.linalg.norm(self.stencil_norms[i])
            face_length = self.face_lengths[i]
            # Projected distances from cell centers to midpoint
            d0_face = np.dot(n_norm, center0 - midpoint)
            d1_face = np.dot(n_norm, center1 - midpoint)
            d0_stencil = np.dot(stencil_norm, center0 - midpoint)
            d1_stencil = np.dot(stencil_norm, center1 - midpoint)
            # Compare face-normal and stencil-normal projected distances
            diff0 = np.abs(d0_face - d0_stencil)
            diff1 = np.abs(d1_face - d1_stencil)
            percent_diff0 = 100.0 * diff0 / face_length if face_length > 0 else 0.0
            percent_diff1 = 100.0 * diff1 / face_length if face_length > 0 else 0.0
            stencil_dist_diffs.extend([diff0, diff1])
            percent_stencil_dist_diffs.extend([percent_diff0, percent_diff1])
            # Check for sign mismatch
            if np.sign(d0_face) != np.sign(d0_stencil):
                stencil_dist_sign_mismatches.append((i, c0, 'd0'))
            if np.sign(d1_face) != np.sign(d1_stencil):
                stencil_dist_sign_mismatches.append((i, c1, 'd1'))
            # Track large differences (percent)
            if percent_diff0 > stencil_dist_diff_threshold:
                stencil_dist_large_diffs.append((i, c0, percent_diff0))
            if percent_diff1 > stencil_dist_diff_threshold:
                stencil_dist_large_diffs.append((i, c1, percent_diff1))

        # --- Print all results at the end ---
        print(f"Mean angle (deg): {np.mean(angles):.2f}, max: {np.max(angles):.2f}")
        print(f"Faces with angle > 30 deg: {bad_angle_count}")
        print(f"Mean distance error: {np.mean(distance_errors):.4f}, max: {np.max(distance_errors):.4f}")
        print(f"Mean distance error (% of face length): {np.mean(percent_distance_errors):.2f}%, max: {np.max(percent_distance_errors):.2f}%")
        print(f"Faces with distance error > 1% of face length: {bad_distance_count}")
        print(f"Mean face center offset: {np.mean(offsets):.4e}, max: {np.max(offsets):.4e}")
        print(f"Mean face center offset (% of face length): {np.mean(percent_offsets):.2f}%, max: {np.max(percent_offsets):.2f}%")
        print(f"Faces with face center offset > 10% of face length: {bad_offset_count}")
        print(f"Mean (cell center dist - face dist sum) as % of face length: {np.mean(center_dist_vs_face_dist):.4e}%, max: {np.max(center_dist_vs_face_dist):.4e}%, min: {np.min(center_dist_vs_face_dist):.4e}%")
        print(f"\nStencil distance checks:")
        print(f"Mean |face-normal dist - stencil-normal dist|: {np.mean(stencil_dist_diffs):.4e}, max: {np.max(stencil_dist_diffs):.4e}")
        print(f"Mean |face-normal dist - stencil-normal dist| (% of face length): {np.mean(percent_stencil_dist_diffs):.2f}%, max: {np.max(percent_stencil_dist_diffs):.2f}%")
        print(f"Faces with |face-normal dist - stencil-normal dist| > {stencil_dist_diff_threshold:.1f}%: {len(stencil_dist_large_diffs)}")
        print(f"Faces with sign mismatch between face-normal and stencil-normal projected distances: {len(stencil_dist_sign_mismatches)}")

    # --- Stencil and angle calculations ---
    def calc_stencil_norms(self):
        '''
        For each face, calculate the stencil norm:
        - For interior faces (with two adjacent cells), the normalized vector from cell0 center to cell1 center.
        - For boundary faces (with one adjacent cell), the normalized vector from the cell center to the face center.
        Ensures stencil norm points in the same direction as the face normal (dot product positive).
        Stores result in self.stencil_norms (shape: [num_faces, 2])
        '''
        # Build a mapping from sorted face (tuple) to list of cell indices
        face_key_to_cells = {}
        for cell_idx, cell in enumerate(self.cells):
            for i in range(3):
                face_key = tuple(sorted([cell[i], cell[(i+1)%3]]))
                face_key_to_cells.setdefault(face_key, []).append(cell_idx)
        stencil_norms = np.zeros_like(self.face_normals)
        for i, face in enumerate(self.faces):
            face_key = tuple(sorted(face))
            cells = face_key_to_cells.get(face_key, [])
            if len(cells) == 2:
                c0, c1 = cells
                v = self.cell_centers[c1] - self.cell_centers[c0]
            elif len(cells) == 1:
                c = cells[0]
                v = self.face_centers[i] - self.cell_centers[c]
            else:
                v = np.zeros(2)
            norm = np.linalg.norm(v)
            if norm > 0:
                stencil_norms[i] = v / norm
            else:
                stencil_norms[i] = v  # zero vector if degenerate
        # Ensure stencil norm and face normal are aligned (dot > 0)
        face_norms = self.face_normals / np.linalg.norm(self.face_normals, axis=1, keepdims=True)
        stencil_norms_unit = stencil_norms / (np.linalg.norm(stencil_norms, axis=1, keepdims=True) + 1e-14)
        dots = np.einsum('ij,ij->i', face_norms, stencil_norms_unit)
        flip_mask = dots < 0
        print(f"Flipping {np.sum(flip_mask)} stencil norms to align with face normals.")
        stencil_norms[flip_mask] *= -1
        self.stencil_norms = stencil_norms

    def calc_face_stencil_angles(self):
        '''
        For each face, calculate the angle (in radians) between the face normal and the stencil norm.
        Stores result in self.face_stencil_angles (shape: [num_faces,])
        '''
        if not hasattr(self, 'stencil_norms'):
            self.calc_stencil_norms()
        face_norms = self.face_normals / np.linalg.norm(self.face_normals, axis=1, keepdims=True)
        stencil_norms = self.stencil_norms / np.linalg.norm(self.stencil_norms, axis=1, keepdims=True)
        dots = np.einsum('ij,ij->i', face_norms, stencil_norms)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)  # radians
        self.face_stencil_angles = angles
