"""Structured gmsh quad meshes of a doubly symmetric plate with an elliptical hole.

The full plate is plate_size x plate_size (default 200 x 200 mm) with a central
elliptical hole. Only one quarter is modelled: the square [0,L] x [0,L],
L = plate_size/2, with a quarter-ellipse hole (semi-axes a along x, b along y) centred at the origin corner. The x = 0 and
y = 0 edges are symmetry planes. A diagonal from the hole to the far corner splits
the quarter into 2 transfinite (mapped) patches. Every patch has fixed divisions,
so node count, element count and connectivity are identical for every hole shape
and size; only node positions change.

The mesh is also returned as a structured [T,R,2] array: i=0 on the x-axis
symmetry edge, i=T-1 on the y-axis symmetry edge (angle increasing
counter-clockwise), j=0 on the hole and j=R-1 on the outer edges x=L, y=L.

Usage (from the repository root):
    python3 plane_stress_geofno/gmsh_plate_mesh.py --a 40 --b 20 --plot  # 200x200 plate
    python3 plane_stress_geofno/gmsh_plate_mesh.py --samples 100 --seed 0 --out plate_meshes
"""

import argparse
import math
import os

import gmsh
import numpy as np


def build_plate_mesh(a, b, *, plate_size=200.0, max_semi_axis=50.0, n_arc=17,
                     n_radial=48, progression=1.05, msh_path=None):
    """Mesh one quarter plate and return a dict of numpy arrays.

    a, b: hole semi-axes along x and y, each in (0, max_semi_axis].
    plate_size: side of the full plate; the modelled quarter is half of it.
    n_arc: nodes along each 45-degree patch edge (T = 2 * (n_arc - 1) + 1).
    n_radial: nodes from hole to outer edge (R). progression > 1 grows the
    radial cell size away from the hole.
    """
    size = plate_size / 2
    if not 0 < max_semi_axis < size:
        raise ValueError(f"max_semi_axis must be in (0, {size}).")
    if not (0 < a <= max_semi_axis and 0 < b <= max_semi_axis):
        raise ValueError(f"Semi-axes must be in (0, {max_semi_axis}]; got a={a}, b={b}.")
    if n_arc < 2 or n_radial < 2 or progression <= 0:
        raise ValueError("n_arc and n_radial must be >= 2, progression > 0.")

    gmsh.model.add(f"quarter_plate_a{a:g}_b{b:g}")
    geo = gmsh.model.geo
    s = math.sqrt(0.5)
    center = geo.addPoint(0, 0, 0)
    hole = [geo.addPoint(a, 0, 0), geo.addPoint(a * s, b * s, 0), geo.addPoint(0, b, 0)]
    edge = [geo.addPoint(size, 0, 0), geo.addPoint(size, size, 0), geo.addPoint(0, size, 0)]
    major = hole[0] if a >= b else hole[2]

    arcs, outer = [], []
    for k in range(2):
        if a == b:
            arcs.append(geo.addCircleArc(hole[k], center, hole[k + 1]))
        else:
            arcs.append(geo.addEllipseArc(hole[k], center, major, hole[k + 1]))
        outer.append(geo.addLine(edge[k], edge[k + 1]))
    radial = [geo.addLine(hole[k], edge[k]) for k in range(3)]  # oriented hole -> edge

    surfaces = []
    for k in range(2):
        loop = geo.addCurveLoop([arcs[k], radial[k + 1], -outer[k], -radial[k]])
        surfaces.append(geo.addPlaneSurface([loop]))
    geo.synchronize()

    for curve in arcs + outer:
        gmsh.model.mesh.setTransfiniteCurve(curve, n_arc)
    for curve in radial:
        gmsh.model.mesh.setTransfiniteCurve(curve, n_radial, "Progression", progression)
    for k, surface in enumerate(surfaces):
        gmsh.model.mesh.setTransfiniteSurface(
            surface, "Left", [hole[k], hole[k + 1], edge[k + 1], edge[k]])
        gmsh.model.mesh.setRecombine(2, surface)

    # Boundary groups: symmetry planes, loaded/free outer edges, and the hole.
    groups = {"hole": arcs, "sym_y0": [radial[0]], "sym_x0": [radial[2]],
              "right": [outer[0]], "top": [outer[1]]}
    for name, curves in groups.items():
        gmsh.model.addPhysicalGroup(1, curves, name=name)
    gmsh.model.addPhysicalGroup(2, surfaces, name="plate")

    gmsh.model.mesh.generate(2)
    if msh_path:
        gmsh.write(msh_path)

    # Nodes referenced by quads (this skips the unmeshed ellipse-centre point).
    elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(2)
    quad_type = gmsh.model.mesh.getElementType("Quadrangle", 1)
    if list(elem_types) != [quad_type]:
        raise RuntimeError(f"Expected only linear quads, got element types {elem_types}.")
    quad_tags = elem_nodes[0].reshape(-1, 4).astype(np.int64)
    node_tags, xyz, _ = gmsh.model.mesh.getNodes()
    xy_by_tag = dict(zip(node_tags.tolist(), xyz.reshape(-1, 3)[:, :2]))
    used = np.unique(quad_tags)
    xy = np.array([xy_by_tag[t] for t in used.tolist()])
    index = {t: i for i, t in enumerate(used.tolist())}
    quads = np.vectorize(index.get)(quad_tags)
    hole_tags = set()
    for curve in arcs:
        hole_tags.update(gmsh.model.mesh.getNodes(1, curve, includeBoundary=True)[0].tolist())
    hole_nodes = np.array(sorted(index[t] for t in hole_tags))

    qualities = gmsh.model.mesh.getElementQualities(elem_tags[0], "minSJ")
    gmsh.model.remove()

    grid = _structured_order(xy, quads, hole_nodes)
    expected = (2 * (n_arc - 1) + 1, n_radial)
    if grid.shape != expected or len(np.unique(grid)) != len(xy):
        raise RuntimeError(f"Mesh is not a full {expected} structured grid; got {grid.shape}.")
    return {"a": a, "b": b, "size": size, "nodes": xy, "quads": quads, "grid_index": grid,
            "coords": xy[grid], "min_scaled_jacobian": float(qualities.min())}


def _structured_order(xy, quads, hole_nodes):
    """Recover [T,R] node indices from the structured connectivity.

    Ring 0 is the hole, walked from the x-axis to the y-axis. Every node in ring
    j has exactly one neighbour outside rings j-1 and j: its node in ring j+1.
    """
    neighbours = [set() for _ in range(len(xy))]
    for q in quads:
        for m in range(4):
            u, v = q[m], q[(m + 1) % 4]
            neighbours[u].add(v)
            neighbours[v].add(u)

    ring_set = set(hole_nodes.tolist())
    start = min(ring_set, key=lambda n: (abs(xy[n, 1]), -xy[n, 0]))
    ring, prev, cur = [start], None, start
    while True:
        step = [n for n in neighbours[cur] if n in ring_set and n != prev]
        if not step:
            break
        prev, cur = cur, step[0]
        ring.append(cur)

    rings, inner = [ring], set()
    while True:
        current = set(rings[-1])
        outward = []
        for n in rings[-1]:
            candidates = neighbours[n] - current - inner
            if len(candidates) != 1:
                break
            outward.append(candidates.pop())
        if len(outward) != len(ring):
            break
        inner = current
        rings.append(outward)
    return np.array(rings).T  # [T,R]


def plot_meshes(meshes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    cols = min(len(meshes), 3)
    rows = math.ceil(len(meshes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    for ax, m in zip(axes.flat, meshes):
        ax.add_collection(PolyCollection(m["nodes"][m["quads"]], facecolor="#8fb3b0",
                                         edgecolor="#1f3b3a", linewidth=0.3))
        ax.plot(*m["coords"][:, 0].T, color="C3", linewidth=1.2)
        pad = 0.02 * m["size"]
        ax.set_xlim(-pad, m["size"] + pad)
        ax.set_ylim(-pad, m["size"] + pad)
        ax.set_aspect("equal")
        ax.set_title(f"a={m['a']:g}, b={m['b']:g}: {len(m['nodes'])} nodes, "
                     f"{len(m['quads'])} quads", fontsize=10)
    for ax in axes.flat[len(meshes):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--a", type=float, nargs="*", default=[25.0, 50.0, 15.0, 50.0, 5.0, 50.0],
                        help="hole semi-axis along x (one value per mesh)")
    parser.add_argument("--b", type=float, nargs="*", default=[25.0, 20.0, 40.0, 5.0, 50.0, 50.0],
                        help="hole semi-axis along y (one value per mesh)")
    parser.add_argument("--samples", type=int, default=0,
                        help="if > 0, draw this many random (a, b) instead of --a/--b")
    parser.add_argument("--range", type=float, nargs=2, default=(5.0, 50.0),
                        help="uniform range for random semi-axes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plate-size", type=float, default=200.0,
                        help="side of the full plate (the quarter modelled is half)")
    parser.add_argument("--max-semi-axis", type=float, default=50.0)
    parser.add_argument("--n-arc", type=int, default=17)
    parser.add_argument("--n-radial", type=int, default=48)
    parser.add_argument("--progression", type=float, default=1.05)
    parser.add_argument("--out", default="plate_meshes")
    parser.add_argument("--plot", action="store_true", help="save a figure of the first 9 meshes")
    args = parser.parse_args()

    if args.samples > 0:
        rng = np.random.default_rng(args.seed)
        pairs = rng.uniform(*args.range, size=(args.samples, 2)).round(3)
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
            meshes.append(build_plate_mesh(
                float(a), float(b), plate_size=args.plate_size,
                max_semi_axis=args.max_semi_axis, n_arc=args.n_arc,
                n_radial=args.n_radial, progression=args.progression,
                msh_path=os.path.join(args.out, f"plate_{k:04d}.msh")))
    finally:
        gmsh.finalize()

    reference = meshes[0]["quads"]
    for m in meshes:
        if len(m["nodes"]) != len(meshes[0]["nodes"]) or not np.array_equal(m["quads"], reference):
            raise RuntimeError(f"Mesh a={m['a']}, b={m['b']} differs in topology.")
        if m["min_scaled_jacobian"] <= 0:
            raise RuntimeError(f"Mesh a={m['a']}, b={m['b']} has inverted elements.")

    np.savez(os.path.join(args.out, "plate_meshes.npz"),
             a=pairs[:, 0], b=pairs[:, 1], plate_size=args.plate_size,
             coords=np.stack([m["coords"] for m in meshes]),      # [N,T,R,2]
             nodes=np.stack([m["nodes"] for m in meshes]),        # [N,nodes,2]
             quads=reference, grid_index=meshes[0]["grid_index"])
    print(f"{len(meshes)} meshes written to {args.out}/")
    print(f"Every mesh: {len(meshes[0]['nodes'])} nodes, {len(reference)} quads, "
          f"identical connectivity; grid [T,R] = {list(meshes[0]['grid_index'].shape)}")
    print("Min scaled Jacobian over all meshes: "
          f"{min(m['min_scaled_jacobian'] for m in meshes):.3f}")
    if args.plot:
        plot_meshes(meshes[:9], os.path.join(args.out, "plate_meshes.png"))
        print(f"Figure: {os.path.join(args.out, 'plate_meshes.png')}")


if __name__ == "__main__":
    main()
