"""Structured gmsh quad meshes of a 2D square RVE with a central (elliptical) fibre.

The RVE is the square [-L/2, L/2]^2 (L = rve_size, default 1) with a central
elliptical inclusion of semi-axes a (along x) and b (along y); a = b gives the
circular fibre of RVE/RVE.jpg. Both phases are meshed: the fibre and the matrix.

Layout (O-grid, 8 sectors of 45 degrees, like 4 copies of the quarter plate):
    - 8 rays at 0, 45, ..., 315 degrees run from a small core rectangle inside
      the fibre, through the fibre boundary, to the RVE edge midpoints (blue dots
      in RVE.jpg) and corners (red dots).
    - Each sector has a fibre patch (core -> interface) and a matrix patch
      (interface -> RVE edge); the core rectangle is one more patch.
Every patch is a gmsh transfinite surface with fixed divisions, so node count,
element count and connectivity are identical for every fibre shape and size;
only node positions change. Opposite RVE edges carry matching nodes and the
.msh file contains gmsh periodic constraints (left->right, bottom->top), ready
for periodic boundary conditions in homogenisation.

Geo-FNO view of the mesh:
    coords [T,R,2]     the ring from the core boundary to the RVE edge, a
                       structured, angularly periodic grid (seam NOT duplicated):
                       i=0 on the +x axis, i increasing counter-clockwise,
                       j=0 on the core boundary, j=interface_j on the fibre
                       boundary, j=R-1 on the RVE edge. Use periodic=True in
                       PlaneStressGeoFNO and plane_stress_fields.
    core_coords [C,C,2] the core rectangle inside the fibre, grid ordered
                       (x index, y index). Its boundary is ring j=0.
    nodes [n,2]        every node (point-cloud view, e.g. elasticity/elas_geofno.py).

Usage (from the repository root):
    python3 RVE/gmsh_rve_mesh.py --plot                    # example shapes
    python3 RVE/gmsh_rve_mesh.py --samples 500 --circle --seed 0 --out rve_meshes
"""

import argparse
import math
import os

import gmsh
import numpy as np

# Ray directions in max-norm form; scaled they give core, fibre and edge points.
_DIRECTIONS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def build_rve_mesh(a, b, *, rve_size=1.0, max_semi_axis=0.45, n_arc=9, n_fibre=9,
                   n_matrix=17, progression=1.05, core_ratio=0.6, msh_path=None):
    """Mesh one RVE and return a dict of numpy arrays.

    a, b: fibre semi-axes along x and y, each in (0, max_semi_axis].
    rve_size: side L of the square RVE, centred at the origin.
    n_arc: nodes along each 45-degree sector edge (T = 8 * (n_arc - 1),
    core grid C = 2 * (n_arc - 1) + 1).
    n_fibre: nodes from core boundary to fibre boundary; n_matrix: nodes from
    fibre boundary to RVE edge (R = n_fibre + n_matrix - 1). progression > 1
    refines both phases towards the fibre/matrix interface.
    core_ratio: the core corners sit at core_ratio x the fibre point on the
    45-degree ray, in (0, 1).
    """
    half = rve_size / 2
    if not 0 < max_semi_axis < half:
        raise ValueError(f"max_semi_axis must be in (0, {half}).")
    if not (0 < a <= max_semi_axis and 0 < b <= max_semi_axis):
        raise ValueError(f"Semi-axes must be in (0, {max_semi_axis}]; got a={a}, b={b}.")
    if min(n_arc, n_fibre, n_matrix) < 2 or progression <= 0 or not 0 < core_ratio < 1:
        raise ValueError("n_arc, n_fibre, n_matrix must be >= 2, progression > 0, "
                         "core_ratio in (0, 1).")

    s = math.sqrt(0.5)
    core = [(core_ratio * s * a * u, core_ratio * s * b * v) for u, v in _DIRECTIONS]
    fibre = [(a * (s if u and v else 1) * u, b * (s if u and v else 1) * v)
             for u, v in _DIRECTIONS]

    def make_arcs(fibre_tags, center):
        geo = gmsh.model.geo
        if a == b:
            return [geo.addCircleArc(fibre_tags[k], center, fibre_tags[(k + 1) % 8])
                    for k in range(8)]
        major = fibre_tags[0] if a >= b else fibre_tags[2]
        return [geo.addEllipseArc(fibre_tags[k], center, major, fibre_tags[(k + 1) % 8])
                for k in range(8)]

    mesh = mesh_rve(f"rve_a{a:g}_b{b:g}", core, fibre, make_arcs, rve_size=rve_size,
                    n_arc=n_arc, n_fibre=n_fibre, n_matrix=n_matrix,
                    progression=progression, msh_path=msh_path)
    mesh.update(a=a, b=b, label=f"a={a:g}, b={b:g}")
    return mesh


def mesh_rve(name, core_xy, fibre_xy, make_arcs, *, rve_size=1.0, n_arc=9, n_fibre=9,
             n_matrix=17, progression=1.05, ray_progression=None, msh_path=None):
    """Mesh the 8-sector O-grid of one RVE and return a dict of numpy arrays.

    core_xy, fibre_xy: 8 (x, y) points on the rays at 0, 45, ..., 315 degrees
    (towards the RVE edge midpoints and corners): the core corners/mid-sides
    and the points where the rays cross the fibre boundary.
    make_arcs(fibre_tags, center_tag): adds the 8 fibre-boundary curves to
    gmsh.model.geo, curve k from fibre point k to fibre point k+1
    (counter-clockwise), and returns their tags.
    ray_progression: optional (fibre[8], matrix[8]) growth ratios away from the
    interface, one per ray; overrides progression.
    Shared by every fibre shape, so all meshes have the same topology.
    """
    half = rve_size / 2
    gmsh.model.add(name)
    geo = gmsh.model.geo
    center = geo.addPoint(0, 0, 0)
    core = [geo.addPoint(x, y, 0) for x, y in core_xy]
    fibre = [geo.addPoint(x, y, 0) for x, y in fibre_xy]
    edge = [geo.addPoint(half * u, half * v, 0) for u, v in _DIRECTIONS]

    nxt = [(k + 1) % 8 for k in range(8)]
    core_lines = [geo.addLine(core[k], core[nxt[k]]) for k in range(8)]
    arcs = make_arcs(fibre, center)
    outer = [geo.addLine(edge[k], edge[nxt[k]]) for k in range(8)]
    fibre_rays = [geo.addLine(core[k], fibre[k]) for k in range(8)]    # core -> interface
    matrix_rays = [geo.addLine(fibre[k], edge[k]) for k in range(8)]   # interface -> edge

    # Counter-clockwise loops: inner edge, outward ray, outer edge back, inward ray.
    core_surface = geo.addPlaneSurface([geo.addCurveLoop(core_lines)])
    fibre_surfaces, matrix_surfaces = [], []
    for k in range(8):
        loop = geo.addCurveLoop([fibre_rays[k], arcs[k], -fibre_rays[nxt[k]], -core_lines[k]])
        fibre_surfaces.append(geo.addPlaneSurface([loop]))
        loop = geo.addCurveLoop([matrix_rays[k], outer[k], -matrix_rays[nxt[k]], -arcs[k]])
        matrix_surfaces.append(geo.addPlaneSurface([loop]))
    geo.synchronize()

    mesh = gmsh.model.mesh
    for curve in core_lines + arcs + outer:
        mesh.setTransfiniteCurve(curve, n_arc)
    fibre_growth, matrix_growth = ray_progression or ([progression] * 8, [progression] * 8)
    for curve, growth in zip(fibre_rays, fibre_growth):
        mesh.setTransfiniteCurve(curve, n_fibre, "Progression", 1 / growth)
    for curve, growth in zip(matrix_rays, matrix_growth):
        mesh.setTransfiniteCurve(curve, n_matrix, "Progression", growth)
    mesh.setTransfiniteSurface(core_surface, "Left", [core[1], core[3], core[5], core[7]])
    for k in range(8):
        mesh.setTransfiniteSurface(fibre_surfaces[k], "Left",
                                   [core[k], fibre[k], fibre[nxt[k]], core[nxt[k]]])
        mesh.setTransfiniteSurface(matrix_surfaces[k], "Left",
                                   [fibre[k], edge[k], edge[nxt[k]], fibre[nxt[k]]])
    for surface in [core_surface] + fibre_surfaces + matrix_surfaces:
        mesh.setRecombine(2, surface)

    # Periodic node matching: right = left + (L, 0), top = bottom + (0, L).
    sides = {"right": [outer[7], outer[0]], "top": [outer[1], outer[2]],
             "left": [outer[4], outer[3]], "bottom": [outer[5], outer[6]]}
    mesh.setPeriodic(1, sides["right"], sides["left"],
                     [1, 0, 0, rve_size, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    mesh.setPeriodic(1, sides["top"], sides["bottom"],
                     [1, 0, 0, 0, 0, 1, 0, rve_size, 0, 0, 1, 0, 0, 0, 0, 1])

    # Physical groups for FEM: edges for periodic/kinematic BCs, the
    # fibre/matrix interface, corners to remove rigid-body motion, and phases.
    for name, curves in sides.items():
        gmsh.model.addPhysicalGroup(1, curves, name=name)
    gmsh.model.addPhysicalGroup(1, arcs, name="interface")
    gmsh.model.addPhysicalGroup(0, [edge[1], edge[3], edge[5], edge[7]], name="corners")
    gmsh.model.addPhysicalGroup(2, [core_surface] + fibre_surfaces, name="fibre")
    gmsh.model.addPhysicalGroup(2, matrix_surfaces, name="matrix")

    mesh.generate(2)
    if msh_path:
        gmsh.write(msh_path)

    # Nodes referenced by quads (this skips the unmeshed centre point).
    quad_type = gmsh.model.mesh.getElementType("Quadrangle", 1)
    elem_types, _, _ = mesh.getElements(2)
    if list(elem_types) != [quad_type]:
        raise RuntimeError(f"Expected only linear quads, got element types {elem_types}.")
    quad_tags, quad_phase, elem_ids = [], [], []
    fibre_set = set([core_surface] + fibre_surfaces)
    for surface in [core_surface] + fibre_surfaces + matrix_surfaces:
        _, tags, nodes = mesh.getElements(2, surface)
        quad_tags.append(nodes[0].reshape(-1, 4))
        quad_phase.append(np.full(len(tags[0]), 1 if surface in fibre_set else 0))
        elem_ids.append(tags[0])
    quad_tags = np.concatenate(quad_tags).astype(np.int64)
    quad_phase = np.concatenate(quad_phase).astype(np.int8)
    node_tags, xyz, _ = mesh.getNodes()
    xy_by_tag = dict(zip(node_tags.tolist(), xyz.reshape(-1, 3)[:, :2]))
    used = np.unique(quad_tags)
    xy = np.array([xy_by_tag[t] for t in used.tolist()])
    index = {t: i for i, t in enumerate(used.tolist())}
    quads = np.vectorize(index.get)(quad_tags)

    def entity_nodes(dim, tags):
        found = set()
        for tag in tags:
            found.update(mesh.getNodes(dim, tag, includeBoundary=True)[0].tolist())
        return np.array(sorted(index[t] for t in found))

    core_boundary = entity_nodes(1, core_lines)
    core_nodes = entity_nodes(2, [core_surface])
    qualities = mesh.getElementQualities(np.concatenate(elem_ids), "minSJ")
    gmsh.model.remove()

    ring, core_grid = _structured_order(xy, quads, core_boundary, core_nodes)
    expected_ring = (8 * (n_arc - 1), n_fibre + n_matrix - 1)
    expected_core = (2 * (n_arc - 1) + 1,) * 2
    if (ring.shape != expected_ring or core_grid.shape != expected_core
            or len(np.union1d(ring.ravel(), core_grid.ravel())) != len(xy)):
        raise RuntimeError(f"Mesh is not a {expected_ring} ring + {expected_core} core grid; "
                           f"got {ring.shape} + {core_grid.shape}.")

    corners = xy[quads]
    area = 0.5 * np.sum(corners[:, :, 0] * np.roll(corners[:, :, 1], -1, axis=1)
                        - np.roll(corners[:, :, 0], -1, axis=1) * corners[:, :, 1], axis=1)
    edges = np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=-1)
    return {"size": rve_size, "nodes": xy, "quads": quads,
            "quad_phase": quad_phase, "ring_index": ring, "core_index": core_grid,
            "coords": xy[ring], "core_coords": xy[core_grid], "interface_j": n_fibre - 1,
            "periodic_x": _periodic_pairs(xy, ring[:, -1], axis=0, half=half),
            "periodic_y": _periodic_pairs(xy, ring[:, -1], axis=1, half=half),
            "min_scaled_jacobian": float(qualities.min()), "min_area": float(area.min()),
            "max_edge_ratio": float((edges.max(1) / edges.min(1)).max())}


def _structured_order(xy, quads, core_boundary, core_nodes):
    """Recover the [T,R] ring and [C,C] core node indices from the connectivity.

    Ring 0 is the core boundary, walked counter-clockwise from the +x axis.
    Every node in ring j has exactly one neighbour outside the core and rings
    0..j: its node in ring j+1. The core is walked row by row from its
    bottom-left corner in the same way, restricted to core nodes.
    """
    neighbours = [set() for _ in range(len(xy))]
    for q in quads:
        for m in range(4):
            u, v = q[m], q[(m + 1) % 4]
            neighbours[u].add(v)
            neighbours[v].add(u)

    boundary = set(core_boundary.tolist())
    start = min(boundary, key=lambda n: (xy[n, 0] <= 0, abs(xy[n, 1])))
    prev, cur = start, max(neighbours[start] & boundary, key=lambda n: xy[n, 1])
    ring = [start]
    while cur != start:
        ring.append(cur)
        prev, cur = cur, next(n for n in neighbours[cur] & boundary if n != prev)

    rings, seen = [ring], set(core_nodes.tolist())
    while True:
        outward = []
        for n in rings[-1]:
            candidates = neighbours[n] - seen - set(rings[-1])
            if len(candidates) != 1:
                break
            outward.append(candidates.pop())
        if len(outward) != len(ring):
            break
        seen.update(rings[-1])
        rings.append(outward)

    # Ring 0 passes the core corners every quarter turn; its stretch from the
    # 225- to the 315-degree corner is the bottom core row, left to right.
    core = set(core_nodes.tolist())
    quarter = len(ring) // 4
    row = (ring + ring[:1])[5 * quarter // 2: 7 * quarter // 2 + 1]
    rows, seen = [row], set()
    while True:
        upward = []
        for n in rows[-1]:
            candidates = (neighbours[n] & core) - seen - set(rows[-1])
            if len(candidates) != 1:
                break
            upward.append(candidates.pop())
        if len(upward) != len(row):
            break
        seen.update(rows[-1])
        rows.append(upward)
    return np.array(rings).T, np.array(rows).T  # [T,R], [C,C] as (x index, y index)


def _periodic_pairs(xy, edge_nodes, *, axis, half):
    """Pairs (minus side, plus side) of edge nodes that differ only along `axis`."""
    other = 1 - axis
    minus = sorted((n for n in edge_nodes if abs(xy[n, axis] + half) < 1e-9 * half),
                   key=lambda n: xy[n, other])
    plus = sorted((n for n in edge_nodes if abs(xy[n, axis] - half) < 1e-9 * half),
                  key=lambda n: xy[n, other])
    pairs = np.array([minus, plus]).T
    if len(minus) != len(plus) or not np.allclose(xy[pairs[:, 0], other], xy[pairs[:, 1], other],
                                                   atol=1e-9 * half):
        raise RuntimeError("Opposite RVE edges do not have matching nodes.")
    return pairs


def plot_meshes(meshes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    cols = min(len(meshes), 3)
    rows = math.ceil(len(meshes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    for ax, m in zip(axes.flat, meshes):
        colours = np.where(m["quad_phase"][:, None] == 1, [0.55, 0.55, 0.55], [0.56, 0.70, 0.69])
        ax.add_collection(PolyCollection(m["nodes"][m["quads"]], facecolors=colours,
                                         edgecolor="#1f3b3a", linewidth=0.3))
        for j, colour in ((0, "C1"), (m["interface_j"], "C3")):
            ax.plot(*np.vstack([m["coords"][:, j], m["coords"][:1, j]]).T,
                    color=colour, linewidth=1.2)
        ax.plot(*m["coords"][0].T, color="C0", linewidth=1.2)   # i = 0 seam ray
        pad = 0.02 * m["size"]
        ax.set_xlim(-m["size"] / 2 - pad, m["size"] / 2 + pad)
        ax.set_ylim(-m["size"] / 2 - pad, m["size"] / 2 + pad)
        ax.set_aspect("equal")
        ax.set_title(f"{m['label']}: {len(m['nodes'])} nodes, "
                     f"{len(m['quads'])} quads", fontsize=10)
    for ax in axes.flat[len(meshes):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_dataset(meshes, path, **extra):
    """Check that all meshes share one topology, save them as an .npz and report.

    extra: per-dataset arrays such as shape parameters, saved alongside.
    """
    ref = meshes[0]
    shared = ("quads", "quad_phase", "ring_index", "core_index", "periodic_x", "periodic_y")
    for m in meshes:
        if len(m["nodes"]) != len(ref["nodes"]) or not all(
                np.array_equal(m[key], ref[key]) for key in shared):
            raise RuntimeError(f"Mesh {m['label']} differs in topology.")
        if m["min_scaled_jacobian"] <= 0 or m["min_area"] <= 0:
            raise RuntimeError(f"Mesh {m['label']} has inverted elements.")

    np.savez(path, **extra,
             coords=np.stack([m["coords"] for m in meshes]),            # [N,T,R,2]
             core_coords=np.stack([m["core_coords"] for m in meshes]),  # [N,C,C,2]
             nodes=np.stack([m["nodes"] for m in meshes]),              # [N,nodes,2]
             interface_j=ref["interface_j"], **{key: ref[key] for key in shared})
    print(f"{len(meshes)} meshes written to {os.path.dirname(path) or '.'}/")
    print(f"Every mesh: {len(ref['nodes'])} nodes, {len(ref['quads'])} quads "
          f"({int(ref['quad_phase'].sum())} fibre), identical connectivity; "
          f"ring [T,R] = {list(ref['ring_index'].shape)}, core [C,C] = "
          f"{list(ref['core_index'].shape)}, interface at j = {ref['interface_j']}")
    print("Min scaled Jacobian over all meshes: "
          f"{min(m['min_scaled_jacobian'] for m in meshes):.3f}; max element edge ratio: "
          f"{max(m['max_edge_ratio'] for m in meshes):.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--a", type=float, nargs="*", default=[0.2, 0.3, 0.4, 0.4, 0.25, 0.45],
                        help="fibre semi-axis along x (one value per mesh)")
    parser.add_argument("--b", type=float, nargs="*", default=[0.2, 0.3, 0.4, 0.25, 0.4, 0.45],
                        help="fibre semi-axis along y (one value per mesh)")
    parser.add_argument("--samples", type=int, default=0,
                        help="if > 0, draw this many random (a, b) instead of --a/--b")
    parser.add_argument("--range", type=float, nargs=2, default=(0.1, 0.45),
                        help="uniform range for random semi-axes")
    parser.add_argument("--circle", action="store_true",
                        help="with --samples: circular fibres (b = a)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rve-size", type=float, default=1.0, help="side of the square RVE")
    parser.add_argument("--max-semi-axis", type=float, default=0.45)
    parser.add_argument("--n-arc", type=int, default=9)
    parser.add_argument("--n-fibre", type=int, default=9)
    parser.add_argument("--n-matrix", type=int, default=17)
    parser.add_argument("--progression", type=float, default=1.05)
    parser.add_argument("--core-ratio", type=float, default=0.6)
    parser.add_argument("--out", default="rve_meshes")
    parser.add_argument("--plot", action="store_true", help="save a figure of the first 9 meshes")
    args = parser.parse_args()

    if args.samples > 0:
        rng = np.random.default_rng(args.seed)
        pairs = rng.uniform(*args.range, size=(args.samples, 2)).round(4)
        if args.circle:
            pairs[:, 1] = pairs[:, 0]
    else:
        if len(args.a) != len(args.b):
            parser.error("--a and --b need the same number of values.")
        pairs = np.column_stack([args.a, args.b])

    os.makedirs(args.out, exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        meshes = []
        for k, (a, b) in enumerate(pairs):
            meshes.append(build_rve_mesh(
                float(a), float(b), rve_size=args.rve_size,
                max_semi_axis=args.max_semi_axis, n_arc=args.n_arc, n_fibre=args.n_fibre,
                n_matrix=args.n_matrix, progression=args.progression,
                core_ratio=args.core_ratio,
                msh_path=os.path.join(args.out, f"rve_{k:04d}.msh")))
    finally:
        gmsh.finalize()

    save_dataset(meshes, os.path.join(args.out, "rve_meshes.npz"),
                 a=pairs[:, 0], b=pairs[:, 1], rve_size=args.rve_size,
                 volume_fraction=math.pi * pairs[:, 0] * pairs[:, 1] / args.rve_size ** 2)
    if args.plot:
        plot_meshes(meshes[:9], os.path.join(args.out, "rve_meshes.png"))
        print(f"Figure: {os.path.join(args.out, 'rve_meshes.png')}")


if __name__ == "__main__":
    main()
