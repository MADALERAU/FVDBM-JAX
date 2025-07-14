"""
MeshRefiner: Mesh quality analysis and adaptive mesh improvement routines for 2D triangular meshes.

This module provides the MeshRefiner class, which offers mesh quality metrics, reporting, and a suite of mesh improvement operations
such as smoothing, edge flipping, patch remeshing, selective refinement, and coarsening. The main entry point is the `improve` method,
which iteratively enhances mesh quality according to user-specified thresholds and options.

Designed for use with meshpy.triangle meshes. All routines are compatible with pdoc documentation generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from meshpy import triangle
from scipy.spatial import Delaunay
import time


class MeshRefiner:
    """
    MeshRefiner provides mesh quality analysis and adaptive mesh improvement for 2D triangular meshes.

    Parameters
    ----------
    mesh : meshpy.triangle.MeshInfo or compatible mesh object
        The mesh to be refined and improved.
    """
    def __init__(self, mesh,holes=None):
        """
        Initialize the MeshRefiner with a mesh object.

        Parameters
        ----------
        mesh : meshpy.triangle.MeshInfo or compatible mesh object
            The mesh to be refined and improved.
        """
        self.mesh = mesh
        self.holes = holes

    # --- Quality metrics and reporting ---
    @staticmethod
    def _aspect_ratio_single(cell_points):
        """
        Compute the aspect ratio of a single triangle.

        Parameters
        ----------
        cell_points : ndarray, shape (3, 2)
            The coordinates of the triangle's vertices.

        Returns
        -------
        float
            The aspect ratio (max edge / min edge).
        """
        edges = np.linalg.norm(cell_points - np.roll(cell_points, -1, axis=0), axis=1)
        return np.max(edges) / np.min(edges)

    @staticmethod
    def _skewness_single(cell_points):
        """
        Compute the skewness (max angle deviation from 60°) of a single triangle.

        Parameters
        ----------
        cell_points : ndarray, shape (3, 2)
            The coordinates of the triangle's vertices.

        Returns
        -------
        float
            The maximum absolute deviation from 60 degrees among the triangle's angles.
        """
        v = np.roll(cell_points, -1, axis=0) - cell_points
        a = np.roll(v, 2, axis=0)
        b = -v
        dot = np.sum(a * b, axis=1)
        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b, axis=1)
        cos_theta = dot / (norm_a * norm_b)
        angles = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi
        return np.max(np.abs(angles - 60))

    def compute_aspect_ratios(self, points, cells):
        """
        Compute aspect ratios for all triangles in the mesh using vectorized numpy operations.

        Parameters
        ----------
        points : ndarray, shape (n_points, 2)
            Array of mesh point coordinates.
        cells : ndarray, shape (n_cells, 3)
            Array of triangle vertex indices.

        Returns
        -------
        ndarray
            Array of aspect ratios for each triangle.
        """
        cell_points = points[cells]  # (n_cells, 3, 2)
        # Vectorized edge calculation
        edges = np.linalg.norm(cell_points - np.roll(cell_points, -1, axis=1), axis=2)  # (n_cells, 3)
        aspect_ratios = np.max(edges, axis=1) / np.min(edges, axis=1)
        return aspect_ratios

    def compute_skewness(self, points, cells):
        """
        Compute skewness (max angle deviation from 60°) for all triangles in the mesh using vectorized numpy operations.

        Parameters
        ----------
        points : ndarray, shape (n_points, 2)
            Array of mesh point coordinates.
        cells : ndarray, shape (n_cells, 3)
            Array of triangle vertex indices.

        Returns
        -------
        ndarray
            Array of skewness values for each triangle.
        """
        cell_points = points[cells]  # (n_cells, 3, 2)
        v = np.roll(cell_points, -1, axis=1) - cell_points  # (n_cells, 3, 2)
        a = np.roll(v, 2, axis=1)
        b = -v
        dot = np.sum(a * b, axis=2)
        norm_a = np.linalg.norm(a, axis=2)
        norm_b = np.linalg.norm(b, axis=2)
        cos_theta = dot / (norm_a * norm_b)
        angles = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi  # (n_cells, 3)
        skewness = np.max(np.abs(angles - 60), axis=1)
        return skewness

    def show_mesh_quality(self, aspect_thresh=3.0, skew_thresh=30.0):
        """
        Display histograms of aspect ratio and skewness for the mesh, and print summary statistics.

        Parameters
        ----------
        aspect_thresh : float
            Threshold for reporting bad aspect ratios.
        skew_thresh : float
            Threshold for reporting bad skewness (degrees).
        """
        points = np.array(self.mesh.points)
        cells = np.array(self.mesh.elements)
        aspect_ratios = self.compute_aspect_ratios(points, cells)
        skewness = self.compute_skewness(points, cells)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].hist(aspect_ratios, bins=30, color='skyblue', edgecolor='k')
        axs[0].set_title('Aspect Ratio Histogram')
        axs[0].set_xlabel('Aspect Ratio')
        axs[0].set_ylabel('Count')
        axs[1].hist(skewness, bins=30, color='salmon', edgecolor='k')
        axs[1].set_title('Skewness Histogram')
        axs[1].set_xlabel('Max Angle Deviation (deg)')
        axs[1].set_ylabel('Count')
        plt.tight_layout()
        plt.show()
        bad_aspect = np.sum(aspect_ratios > aspect_thresh)
        bad_skew = np.sum(skewness > skew_thresh)
        print(f"Number of cells with aspect ratio > {aspect_thresh}: {bad_aspect}")
        print(f"Number of cells with skewness > {skew_thresh}°: {bad_skew}")
        print(f"Maximum skewness: {np.max(skewness):.2f}")

    def _triangle_angles(self, p):
        """
        Compute the angles of a triangle in degrees.

        Parameters
        ----------
        p : ndarray, shape (3, 2)
            The coordinates of the triangle's vertices.

        Returns
        -------
        tuple
            (angles array, min angle, max angle)
        """
        p = np.array(p)
        v = np.roll(p, -1, axis=0) - p
        a = np.roll(v, 2, axis=0)
        b = -v
        dot = np.sum(a * b, axis=1)
        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b, axis=1)
        cos_theta = dot / (norm_a * norm_b)
        angles = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi
        return np.array(angles), float(np.min(angles)), float(np.max(angles))

    def _grading_ok(self, points, cell, grading_ratio=2.0):
        """
        Check if the grading (max/min edge ratio) of a triangle is within the allowed ratio (vectorized).

        Parameters
        ----------
        points : ndarray
            Array of mesh point coordinates.
        cell : array-like
            Indices of the triangle's vertices.
        grading_ratio : float
            Maximum allowed ratio of longest to shortest edge.

        Returns
        -------
        bool
            True if grading is acceptable, False otherwise.
        """
        verts = points[cell]
        edge_lengths = np.linalg.norm(verts - np.roll(verts, -1, axis=0), axis=1)
        return np.max(edge_lengths) / np.min(edge_lengths) <= grading_ratio

    # --- Mesh operations ---
    def flip_edges(self, verbose=False):
        """
        Perform edge flipping to improve minimum angles in the mesh.

        Parameters
        ----------
        verbose : bool
            If True, print the number of edge flips performed.

        Returns
        -------
        int
            Number of edges flipped.
        """
        points = np.array(self.mesh.points)
        elements = np.array(self.mesh.elements)
        edge_to_cells = {}
        for idx, tri in enumerate(elements):
            for i in range(3):
                edge = tuple(sorted((tri[i], tri[(i+1)%3])))
                edge_to_cells.setdefault(edge, []).append(idx)
        flipped = 0
        for edge, cells in edge_to_cells.items():
            if len(cells) != 2:
                continue
            t1, t2 = elements[cells[0]], elements[cells[1]]
            opp1 = [v for v in t1 if v not in edge][0]
            opp2 = [v for v in t2 if v not in edge][0]
            def min_angle(tri):
                p = points[list(tri)]
                v = [p[(i+1)%3] - p[i] for i in range(3)]
                angles = []
                for i in range(3):
                    a = v[i-1]
                    b = -v[i]
                    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                    angle = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi
                    angles.append(angle)
                return min(angles)
            old_min = min(min_angle(t1), min_angle(t2))
            new_min = min(min_angle([opp1, opp2, edge[0]]), min_angle([opp1, opp2, edge[1]]))
            if new_min > old_min + 1e-8:
                elements[cells[0]] = [opp1, opp2, edge[0]]
                elements[cells[1]] = [opp1, opp2, edge[1]]
                flipped += 1
        if flipped > 0:
            info = triangle.MeshInfo()
            info.set_points(self.mesh.points)
            info.set_facets(self.mesh.facets)
            if self.holes is not None and len(self.holes) > 0:
                info.set_holes(self.holes)
            self.mesh = triangle.build(
                info,
                min_angle=30,
                generate_faces=True,
                generate_neighbor_lists=True,
                attributes=True,
                volume_constraints=True
            )
            if verbose:
                print(f"  Edge flips performed: {flipped}")
        return flipped

    def laplacian_smooth(self, points, facets, iterations=1):
        """
        Perform Laplacian smoothing on mesh points (excluding boundary nodes) using vectorized numpy operations.

        Parameters
        ----------
        points : array-like
            List or array of mesh point coordinates.
        facets : array-like
            List of mesh facets (boundary edges).
        iterations : int
            Number of smoothing iterations.

        Returns
        -------
        list
            List of smoothed point coordinates.
        """
        points = np.array(points)
        n_points = points.shape[0]
        # Build neighbor list
        neighbors = [[] for _ in range(n_points)]
        for facet in facets:
            for i in range(len(facet)):
                a, b = facet[i], facet[(i+1)%len(facet)]
                neighbors[a].append(b)
                neighbors[b].append(a)
        # Identify boundary nodes
        boundary = np.zeros(n_points, dtype=bool)
        for facet in facets:
            for idx in facet:
                boundary[idx] = True
        for _ in range(iterations):
            new_points = points.copy()
            for i in range(n_points):
                if boundary[i]:
                    continue
                nbs = neighbors[i]
                if nbs:
                    new_points[i] = np.mean(points[nbs], axis=0)
            points = new_points
        return [tuple(pt) for pt in points]

    def angle_based_smooth(self, iterations=1):
        """
        Perform angle-based smoothing to maximize minimum triangle angles.

        Parameters
        ----------
        iterations : int
            Number of smoothing iterations.
        """
        points = np.array(self.mesh.points)
        elements = np.array(self.mesh.elements)
        point_tris = {i: [] for i in range(len(points))}
        for idx, tri in enumerate(elements):
            for v in tri:
                point_tris[v].append(idx)
        boundary = set()
        for facet in self.mesh.facets:
            for idx in facet:
                boundary.add(idx)
        for _ in range(iterations):
            for i in range(len(points)):
                if i in boundary:
                    continue
                tris = [elements[j] for j in point_tris[i]]
                best_pt = points[i]
                best_min_angle = min(
                    min(self._triangle_angles(points[tri])[0]) for tri in tris
                )
                neighbors = set()
                for tri in tris:
                    for v in tri:
                        if v != i:
                            neighbors.add(v)
                if neighbors:
                    candidate = np.mean(points[list(neighbors)], axis=0)
                    test_points = points.copy()
                    test_points[i] = candidate
                    min_angle = min(
                        min(self._triangle_angles(test_points[tri])[0]) for tri in tris
                    )
                    if min_angle > best_min_angle:
                        points[i] = candidate
        info = triangle.MeshInfo()
        info.set_points(points.tolist())
        info.set_facets(self.mesh.facets)
        if self.holes is not None and len(self.holes) > 0:
            info.set_holes(self.holes)
        self.mesh = triangle.build(
            info,
            min_angle=30,
            generate_faces=True,
            generate_neighbor_lists=True,
            attributes=True,
            volume_constraints=True
        )

    def selective_refine(self, points, facets, cells, aspect_thresh, skew_thresh, min_area, max_refine=10):
        """
        Refine the mesh by adding centroids of worst-quality triangles (vectorized).

        Parameters
        ----------
        points : array-like
            List of mesh point coordinates.
        facets : array-like
            List of mesh facets.
        cells : array-like
            List of triangle vertex indices.
        aspect_thresh : float
            Aspect ratio threshold for refinement.
        skew_thresh : float
            Skewness threshold for refinement.
        min_area : float
            Minimum area for a triangle to be considered for refinement.
        max_refine : int
            Maximum number of new points to add per call.

        Returns
        -------
        list
            Updated list of mesh points.
        """
        points_arr = np.asarray(points)
        cells_arr = np.asarray(cells)
        p = points_arr[cells_arr]
        area = 0.5 * np.abs(
            p[:,0,0]*(p[:,1,1]-p[:,2,1]) +
            p[:,1,0]*(p[:,2,1]-p[:,0,1]) +
            p[:,2,0]*(p[:,0,1]-p[:,1,1])
        )
        skew = self.compute_skewness(points_arr, cells_arr)
        aspect = self.compute_aspect_ratios(points_arr, cells_arr)
        mask = (area >= min_area)
        qualities = np.stack([skew, aspect], axis=1)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            return list(points)
        # Sort by skew, then aspect
        sort_idx = np.lexsort((aspect[idxs], skew[idxs]))[::-1]
        worst_cells = cells_arr[idxs][sort_idx][:max_refine]
        centroids = np.mean(points_arr[worst_cells], axis=1)
        # Only add centroids not already in points
        new_points = [tuple(c) for c in centroids if tuple(c) not in points]
        return list(points) + new_points

    def patch_remesh(self, worst_cell_idx, patch_radius=1, verbose=True):
        """
        Remesh a local patch around the worst-quality triangle.

        Parameters
        ----------
        worst_cell_idx : int
            Index of the worst-quality triangle.
        patch_radius : int
            Number of adjacency layers to include in the patch.
        verbose : bool
            If True, print the number of triangles remeshed.
        """
        points = np.array(self.mesh.points)
        elements = np.array(self.mesh.elements)
        patch_tris = set([worst_cell_idx])
        for _ in range(patch_radius):
            new_tris = set()
            for tri_idx in patch_tris:
                tri = elements[tri_idx]
                for i in range(3):
                    edge = set([tri[i], tri[(i+1)%3]])
                    for j, other in enumerate(elements):
                        if j != tri_idx and len(edge & set(other)) >= 2:
                            new_tris.add(j)
            patch_tris |= new_tris
        patch_tris = list(patch_tris)
        patch_pts_idx = set()
        for tri_idx in patch_tris:
            patch_pts_idx.update(elements[tri_idx])
        patch_pts_idx = list(patch_pts_idx)
        patch_points = points[patch_pts_idx]
        mask = np.ones(len(elements), dtype=bool)
        mask[patch_tris] = False
        new_elements = elements[mask]
        if len(patch_points) >= 3:
            delaunay = Delaunay(patch_points)
            for simplex in delaunay.simplices:
                new_elements = np.vstack([new_elements, [patch_pts_idx[i] for i in simplex]])
        info = triangle.MeshInfo()
        info.set_points([tuple(pt) for pt in points])
        info.set_facets(self.mesh.facets)
        if self.holes is not None and len(self.holes) > 0:
            info.set_holes(self.holes)
        self.mesh = triangle.build(
            info,
            min_angle=30,
            generate_faces=True,
            generate_neighbor_lists=True,
            attributes=True,
            volume_constraints=True
        )
        if verbose:
            print(f"Patch remeshed: {len(patch_tris)} triangles replaced.")

    def relocate_bad_points(self, points, cells, bad_cells, move_fraction=0.5):
        """
        Move nodes of bad triangles toward their centroid, skipping boundary nodes. Uses numpy masking for speed.

        Parameters
        ----------
        points : array-like
            List of mesh point coordinates.
        cells : array-like
            List of triangle vertex indices.
        bad_cells : list
            List of bad triangle vertex indices.
        move_fraction : float
            Fraction of the distance to move toward the centroid (0=no move, 1=to centroid).

        Returns
        -------
        list
            Updated list of mesh points.
        """
        points = np.array(points)
        n_points = points.shape[0]
        # Identify boundary nodes
        boundary_nodes = np.zeros(n_points, dtype=bool)
        for facet in self.mesh.facets:
            for idx in facet:
                boundary_nodes[idx] = True
        for cell in bad_cells:
            verts = points[cell]
            centroid = np.mean(verts, axis=0)
            for idx in cell:
                if boundary_nodes[idx]:
                    continue
                points[idx] = (1 - move_fraction) * points[idx] + move_fraction * centroid
        return [tuple(pt) for pt in points]

    def coarsen_mesh(self, min_edge_length=0.05, max_iterations=5, verbose=True):
        """
        Coarsen the mesh by collapsing short edges and removing degenerate triangles.

        Parameters
        ----------
        min_edge_length : float
            Minimum allowed edge length; edges shorter than this will be collapsed.
        max_iterations : int
            Maximum number of coarsening passes.
        verbose : bool
            If True, print progress information.

        Returns
        -------
        meshpy.triangle.MeshInfo
            The coarsened mesh object.
        """
        for it in range(max_iterations):
            points = np.array(self.mesh.points)
            elements = np.array(self.mesh.elements)
            edge_to_cells = {}
            for idx, tri in enumerate(elements):
                for i in range(3):
                    edge = tuple(sorted((tri[i], tri[(i+1)%3])))
                    edge_to_cells.setdefault(edge, []).append(idx)
            # Find short edges
            short_edges = []
            for edge in edge_to_cells:
                p0, p1 = points[list(edge)]
                length = np.linalg.norm(p0 - p1)
                if length < min_edge_length:
                    short_edges.append(edge)
            if not short_edges:
                if verbose:
                    print(f"Coarsening stopped at iteration {it}: no short edges left.")
                break
            # Collapse edges
            collapsed = set()
            for edge in short_edges:
                if edge[0] in collapsed or edge[1] in collapsed:
                    continue
                # Collapse edge to midpoint
                midpoint = (points[edge[0]] + points[edge[1]]) / 2
                points[edge[0]] = midpoint
                points[edge[1]] = midpoint
                collapsed.update(edge)
            # Remove duplicate points
            unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
            new_elements = []
            for tri in elements:
                new_tri = [int(inverse[v]) for v in tri]
                # Skip degenerate triangles
                if len(set(new_tri)) == 3:
                    new_elements.append(new_tri)
            # Rebuild mesh
            info = triangle.MeshInfo()
            info.set_points([tuple(pt) for pt in unique_points])
            info.set_facets(self.mesh.facets)
            self.mesh = triangle.build(
                info,
                min_angle=30,
                generate_faces=True,
                generate_neighbor_lists=True,
                attributes=True,
                volume_constraints=True
            )
            if verbose:
                print(f"Coarsening iteration {it+1}: {len(short_edges)} edges collapsed.")
        return self.mesh

    # --- Main refinement loop ---
    def improve(self, aspect_thresh=3.0, skew_thresh=30.0, min_area=1e-12, max_iter=10,
                smoothing=True, edge_flipping=True, angle_smoothing=True, selective_refinement=True,
                max_refine=10, patch_remesh=True, patch_radius=1, verbose=True,
                global_remesh_interval=5, grading_ratio=2.0, max_volume=None, move_fraction=0.5):
        """
        Iteratively improve mesh quality using smoothing, edge flipping, patch remeshing, and refinement.

        Parameters
        ----------
        aspect_thresh : float
            Aspect ratio threshold for identifying bad triangles.
        skew_thresh : float
            Skewness threshold (degrees) for identifying bad triangles.
        min_area : float
            Minimum area for a triangle to be considered in quality checks.
        max_iter : int
            Maximum number of improvement iterations.
        smoothing : bool
            If True, apply Laplacian smoothing each iteration.
        edge_flipping : bool
            If True, perform edge flipping each iteration.
        angle_smoothing : bool
            If True, perform angle-based smoothing each iteration.
        selective_refinement : bool
            If True, relocate bad points instead of global refinement.
        max_refine : int
            Maximum number of new points to add per iteration (if not relocating).
        patch_remesh : bool
            If True, perform patch remeshing for very bad triangles.
        patch_radius : int
            Number of adjacency layers for patch remeshing.
        verbose : bool
            If True, print progress information.
        global_remesh_interval : int
            Perform global remeshing every N iterations.
        grading_ratio : float
            Maximum allowed edge grading ratio.
        max_volume : float or None
            Maximum allowed triangle area for remeshing (None for unlimited).
        move_fraction : float
            Fraction of the distance to move bad points toward centroid.

        Returns
        -------
        meshpy.triangle.MeshInfo
            The improved mesh object.
        """
        element_counts = []
        for it in range(max_iter):
            t0 = time.time()
            points = list(self.mesh.points)
            facets = list(self.mesh.facets)
            cells = np.array(self.mesh.elements)
            points_arr = np.array(points)
            bad_cells = []
            skewness = []
            for cell in cells:
                verts = points_arr[cell]
                area = 0.5 * abs(
                    (verts[0][0]*(verts[1][1]-verts[2][1]) +
                     verts[1][0]*(verts[2][1]-verts[0][1]) +
                     verts[2][0]*(verts[0][1]-verts[1][1]))
                )
                if area < min_area:
                    continue
                skew = self.compute_skewness(points_arr, [cell])[0]
                aspect = self.compute_aspect_ratios(points_arr, [cell])[0]
                skewness.append(skew)
                # Use points_arr instead of points for _grading_ok
                if (skew > skew_thresh or aspect > aspect_thresh) and self._grading_ok(points_arr, cell, grading_ratio):
                    bad_cells.append(cell)
            max_skew = np.max(skewness) if skewness else 0
            num_elements = len(self.mesh.elements)
            element_counts.append(num_elements)
            if not bad_cells:
                iter_time = time.time() - t0
                if verbose:
                    print(f"Iteration {it}: {len(bad_cells)} bad triangles, max skewness: {max_skew:.2f}, elements: {num_elements}, elapsed: {iter_time:.3f}s")
                    print("Mesh quality acceptable.")
                break
            if patch_remesh and max_skew > skew_thresh * 2:
                self.patch_remesh(np.argmax(skewness), patch_radius=patch_radius, verbose=verbose)
            elif selective_refinement:
                points = self.relocate_bad_points(points, cells, bad_cells, move_fraction=move_fraction)
            else:
                new_points = []
                for cell in bad_cells:
                    verts = points_arr[cell]
                    centroid = tuple(np.mean(verts, axis=0))
                    if centroid not in points and centroid not in new_points:
                        new_points.append(centroid)
                points.extend(new_points)
            if smoothing:
                points = self.laplacian_smooth(points, facets, iterations=2)
            if global_remesh_interval and (it+1) % global_remesh_interval == 0:
                info = triangle.MeshInfo()
                info.set_points(points)
                info.set_facets(facets)
                if self.holes is not None and len(self.holes) > 0:
                    info.set_holes(self.holes)
                self.mesh = triangle.build(
                    info,
                    min_angle=30,
                    max_volume=max_volume,
                    generate_faces=True,
                    generate_neighbor_lists=True,
                    attributes=True,
                    volume_constraints=True
                )
                if verbose:
                    print(f"Global remeshing performed at iteration {it+1}")
            else:
                info = triangle.MeshInfo()
                info.set_points(points)
                info.set_facets(facets)
                if self.holes is not None and len(self.holes) > 0:
                    info.set_holes(self.holes)
                self.mesh = triangle.build(
                    info,
                    min_angle=30,
                    generate_faces=True,
                    generate_neighbor_lists=True,
                    attributes=True,
                    volume_constraints=True
                )
            if edge_flipping:
                self.flip_edges(verbose=verbose)
            if angle_smoothing:
                self.angle_based_smooth(iterations=1)
            iter_time = time.time() - t0
            if verbose:
                print(f"Iteration {it+1}: {len(bad_cells)} bad triangles, max skewness: {max_skew:.2f}, elements: {num_elements}, elapsed: {iter_time:.3f}s")
        if verbose:
            print(f"Element counts per iteration: {element_counts}")
        return self.mesh