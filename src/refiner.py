import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from meshpy import triangle
from scipy.spatial import Delaunay

class MeshRefiner:
    def __init__(self, mesh):
        self.mesh = mesh

    def compute_aspect_ratios(self, points, cells):
        p = points[cells]  # (n_cells, 3, 2)
        edges = jnp.linalg.norm(p - jnp.roll(p, -1, axis=1), axis=2)  # (n_cells, 3)
        aspect = jnp.max(edges, axis=1) / jnp.min(edges, axis=1)
        return np.array(aspect)

    def compute_skewness(self, points, cells):
        p = points[cells]  # (n_cells, 3, 2)
        v = jnp.roll(p, -1, axis=1) - p  # (n_cells, 3, 2)
        a = jnp.roll(v, 2, axis=1)
        b = -v
        dot = jnp.sum(a * b, axis=2)
        norm_a = jnp.linalg.norm(a, axis=2)
        norm_b = jnp.linalg.norm(b, axis=2)
        cos_theta = dot / (norm_a * norm_b)
        angles = jnp.arccos(jnp.clip(cos_theta, -1, 1)) * 180 / jnp.pi
        skew = jnp.max(jnp.abs(angles - 60), axis=1)
        return np.array(skew)

    def show_mesh_quality(self, aspect_thresh=3.0, skew_thresh=30.0):
        points = np.array(self.mesh.points)
        cells = np.array(self.mesh.elements)
        aspect_ratios = self.compute_aspect_ratios(jnp.array(points), jnp.array(cells))
        skewness = self.compute_skewness(jnp.array(points), jnp.array(cells))
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

    def flip_edges(self, verbose=False):
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

    def angle_based_smooth(self, iterations=1):
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
        self.mesh = triangle.build(
            info,
            min_angle=30,
            generate_faces=True,
            generate_neighbor_lists=True,
            attributes=True,
            volume_constraints=True
        )

    def _triangle_angles(self, p):
        p = jnp.array(p)
        v = jnp.roll(p, -1, axis=0) - p
        a = jnp.roll(v, 2, axis=0)
        b = -v
        dot = jnp.sum(a * b, axis=1)
        norm_a = jnp.linalg.norm(a, axis=1)
        norm_b = jnp.linalg.norm(b, axis=1)
        cos_theta = dot / (norm_a * norm_b)
        angles = jnp.arccos(jnp.clip(cos_theta, -1, 1)) * 180 / jnp.pi
        return np.array(angles), float(jnp.min(angles)), float(jnp.max(angles))

    def laplacian_smooth(self, points, facets, iterations=1):
        points = jnp.array(points)
        edge_map = {i: set() for i in range(points.shape[0])}
        for facet in facets:
            for i in range(len(facet)):
                a, b = facet[i], facet[(i+1)%len(facet)]
                edge_map[a].add(b)
                edge_map[b].add(a)
        boundary = set()
        for facet in facets:
            for idx in facet:
                boundary.add(idx)
        for _ in range(iterations):
            new_points = points.copy()
            for i in range(points.shape[0]):
                if i in boundary:
                    continue
                neighbors = list(edge_map[i])
                if neighbors:
                    new_points = new_points.at[i].set(jnp.mean(points[jnp.array(neighbors)], axis=0))
            points = new_points
        return [tuple(pt) for pt in np.array(points)]

    def selective_refine(self, points, facets, cells, aspect_thresh, skew_thresh, min_area, max_refine=10):
        points_arr = jnp.array(points)
        cells_arr = jnp.array(cells)
        p = points_arr[cells_arr]
        area = 0.5 * jnp.abs(
            p[:,0,0]*(p[:,1,1]-p[:,2,1]) +
            p[:,1,0]*(p[:,2,1]-p[:,0,1]) +
            p[:,2,0]*(p[:,0,1]-p[:,1,1])
        )
        skew = self.compute_skewness(points_arr, cells_arr)
        aspect = self.compute_aspect_ratios(points_arr, cells_arr)
        mask = (area >= min_area)
        qualities = list(zip(skew[mask], aspect[mask], np.where(mask)[0], cells_arr[mask]))
        qualities.sort(reverse=True, key=lambda x: (x[0], x[1]))
        worst_cell = qualities[0][3] if qualities else None
        new_points = []
        included = set()
        if worst_cell is not None:
            verts = points_arr[worst_cell]
            centroid = tuple(jnp.mean(verts, axis=0))
            new_points.append(centroid)
            included.add(tuple(worst_cell))
        for q in qualities[1:max_refine]:
            cell = q[3]
            if tuple(cell) in included:
                continue
            verts = points_arr[cell]
            centroid = tuple(jnp.mean(verts, axis=0))
            if centroid not in points and centroid not in new_points:
                new_points.append(centroid)
                included.add(tuple(cell))
        points = list(points) + new_points
        return points

    def patch_remesh(self, worst_cell_idx, patch_radius=1, verbose=True):
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

    def _grading_ok(self, points, cell, grading_ratio=2.0):
        points = np.array(points)
        verts = points[cell]
        edge_lengths = [np.linalg.norm(verts[i] - verts[(i+1)%3]) for i in range(3)]
        if max(edge_lengths) / min(edge_lengths) > grading_ratio:
            return False
        return True
    
    def relocate_bad_points(self, points, cells, bad_cells, move_fraction=0.5):
        """
        Move the nodes of bad triangles toward their centroid,
        but do NOT move boundary nodes.
        move_fraction: 0.0 (no move), 1.0 (move directly to centroid)
        """
        points = np.array(points)
        # Identify boundary nodes
        boundary_nodes = set()
        for facet in self.mesh.facets:
            for idx in facet:
                boundary_nodes.add(idx)
        for cell in bad_cells:
            verts = points[cell]
            centroid = np.mean(verts, axis=0)
            for idx in cell:
                if idx in boundary_nodes:
                    continue  # Skip boundary nodes
                points[idx] = (1 - move_fraction) * points[idx] + move_fraction * centroid
        return [tuple(pt) for pt in points]

    def improve(self, aspect_thresh=3.0, skew_thresh=30.0, min_area=1e-12, max_iter=10,
                smoothing=True, edge_flipping=True, angle_smoothing=True, selective_refinement=True,
                max_refine=10, patch_remesh=True, patch_radius=1, verbose=True,
                global_remesh_interval=5, grading_ratio=2.0, max_volume=None, move_fraction=0.5):
        element_counts = []
        for it in range(max_iter):
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
                if (skew > skew_thresh or aspect > aspect_thresh) and self._grading_ok(points, cell, grading_ratio):
                    bad_cells.append(cell)
            max_skew = np.max(skewness) if skewness else 0
            num_elements = len(self.mesh.elements)
            element_counts.append(num_elements)
            if verbose:
                print(f"Iteration {it}: {len(bad_cells)} bad triangles, max skewness: {max_skew:.2f}, elements: {num_elements}")
            if not bad_cells:
                if verbose:
                    print("Mesh quality acceptable.")
                break
            if patch_remesh and max_skew > skew_thresh * 2:
                worst_idx = np.argmax(skewness)
                self.patch_remesh(worst_idx, patch_radius=patch_radius, verbose=verbose)
                continue
            if selective_refinement:
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
        if verbose:
            print(f"Element counts per iteration: {element_counts}")
        return self.mesh

    def coarsen_mesh(self, min_edge_length=0.05, max_iterations=5, verbose=True):
        """
        Coarsen the mesh by collapsing short edges to smooth out the mesh.
        min_edge_length: minimum allowed edge length; edges shorter than this will be collapsed.
        max_iterations: maximum number of passes.
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