"""Structured gmsh quad meshes of a 2D square RVE with a rotated, non-circular fibre.

Fibre shapes (RVE/Fiber_shapes.png), each with length 2a along its major axis,
width 2b (a >= b, aspect ratio a/b) and rotation theta:
    ellipse       semi-axes a, b
    stadium       rectangle with two semicircular caps of radius b
    rounded_rect  2a x 2b rectangle with corner radius R_f (0 < R_f <= b)
theta is the angle of the major axis, in degrees, measured counter-clockwise
from the +x axis. The major length 2a is at most max_length x the RVE side
(default 60%), the aspect ratio a/b is at most max_fibre_aspect (default 2)
and the rounded_rect corner radius R_f / b lies in corner_range (default 0.2-0.6).

The mesh uses the same 8-sector O-grid as gmsh_rve_mesh.py (it calls
gmsh_rve_mesh.mesh_rve). The 8 rays at 0, 45, ..., 315 degrees cross the fibre
boundary at the fibre points. Each fibre point is joined to the RVE edge
midpoint or corner on its ray and, inward, to a core quadrilateral. Node
count, connectivity, grid ordering, physical groups, periodic constraints and
the .npz layout are therefore identical to gmsh_rve_mesh.py. With the same
mesh options, meshes from both scripts can be mixed in one dataset.

Usage (from the repository root):
    python3 RVE/gmsh_rve_shapes.py --plot                                   # example shapes
    python3 RVE/gmsh_rve_shapes.py --samples 500 --seed 0 --out rve_shape_meshes
    python3 RVE/gmsh_rve_shapes.py --samples 500 --shapes stadium --aspect-range 1.5 2 --length-range 0.4 0.6
"""

import argparse
import math
import os

import gmsh
import numpy as np

from gmsh_rve_mesh import _DIRECTIONS, mesh_rve, plot_meshes, save_dataset

SHAPES = ("ellipse", "stadium", "rounded_rect")


def fibre_sdf(shape, a, b, theta, corner_radius, x, y):
    """Negative inside the fibre, positive outside (exact distance except ellipse)."""
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    u, v = np.abs(c * x + s * y), np.abs(-s * x + c * y)   # local axes, first quadrant
    if shape == "ellipse":
        return np.hypot(u / a, v / b) - 1
    if shape == "stadium":
        return np.hypot(np.maximum(u - (a - b), 0), v) - b
    qx, qy = u - (a - corner_radius), v - (b - corner_radius)
    return (np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
            + np.minimum(np.maximum(qx, qy), 0) - corner_radius)


def fibre_area(shape, a, b, corner_radius):
    if shape == "ellipse":
        return math.pi * a * b
    if shape == "stadium":
        return math.pi * b * b + 4 * b * (a - b)
    return 4 * a * b - (4 - math.pi) * corner_radius ** 2


def ray_points(shape, a, b, theta, corner_radius, phi):
    """Points where rays from the centre at angles phi (radians) leave the fibre."""
    direction = np.stack([np.cos(phi), np.sin(phi)], axis=-1)
    lo, hi = np.zeros(phi.shape), np.full(phi.shape, 2.0 * a)
    for _ in range(60):  # the fibre is convex, so each ray crosses it once
        mid = (lo + hi) / 2
        inside = fibre_sdf(shape, a, b, theta, corner_radius,
                           mid * direction[..., 0], mid * direction[..., 1]) < 0
        lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
    return (lo + hi)[..., None] / 2 * direction


def build_shape_mesh(shape, a, b, theta=0.0, corner_radius=0.0, *, rve_size=1.0, max_length=0.6,
                     max_fibre_aspect=2.0, min_gap=0.04, n_arc=17, n_fibre=13, n_matrix=13,
                     max_growth=1.1, core_ratio=0.6, corner_range=(0.2, 0.6),
                     n_spline=40, msh_path=None):
    """Mesh one RVE with the given fibre and return a dict of numpy arrays.

    shape: one of SHAPES. a >= b > 0: half length and half width of the fibre.
    theta: major-axis angle in degrees from +x. corner_radius: R_f for
    rounded_rect, with R_f / b in corner_range (a sub-range of (0, 1]);
    ignored otherwise.
    max_length: largest allowed major length 2a, as a fraction of rve_size.
    max_fibre_aspect: largest allowed aspect ratio a/b.
    min_gap: smallest allowed distance between the fibre and the RVE edge.
    n_spline: points per sector used to describe stadium/rounded_rect
    boundaries (gmsh splines through exact boundary points).
    Radial grading is set per ray: the first cell at the interface is as long
    as the local interface spacing (square cells), growing away from it by at
    most max_growth per cell, smoothed over neighbouring rays. Thin gaps get
    uniform cells.
    The remaining options are those of gmsh_rve_mesh.build_rve_mesh. Defaults
    are finer than gmsh_rve_mesh.py because elongated fibres have a longer
    boundary; pass the same n_arc/n_fibre/n_matrix to both scripts to mix them.
    """
    half = rve_size / 2
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}; got {shape!r}.")
    if not 0 < b <= a:
        raise ValueError(f"Need 0 < b <= a; got a={a}, b={b}.")
    if a / b > max_fibre_aspect * (1 + 1e-9):
        raise ValueError(f"Aspect ratio a/b = {a / b:.3g} exceeds {max_fibre_aspect:g}.")
    if 2 * a > max_length * rve_size * (1 + 1e-9):
        raise ValueError(f"Major length 2a = {2 * a:g} exceeds {max_length:g} x RVE size "
                         f"({max_length * rve_size:g}).")
    if shape != "rounded_rect":
        corner_radius = 0.0
    elif not (0 < corner_range[0] <= corner_range[1] <= 1 and corner_range[0] * (1 - 1e-9)
              <= corner_radius / b <= corner_range[1] * (1 + 1e-9)):
        raise ValueError(f"rounded_rect needs corner_radius / b in [{corner_range[0]:g}, "
                         f"{corner_range[1]:g}]; got {corner_radius / b:.3g}.")
    if min(n_arc, n_fibre, n_matrix) < 2 or max_growth < 1 or not 0 < core_ratio < 1:
        raise ValueError("n_arc, n_fibre, n_matrix must be >= 2, max_growth >= 1, "
                         "core_ratio in (0, 1).")
    outline = ray_points(shape, a, b, theta, corner_radius, np.linspace(0, 2 * math.pi, 1441))
    if np.abs(outline).max() > half - min_gap:
        raise ValueError(f"Fibre {shape} a={a:g}, b={b:g}, theta={theta:g} does not fit "
                         f"inside the RVE with a gap of {min_gap:g}.")

    # Fibre points on the 8 rays; core corners at core_ratio x the fibre point on
    # the diagonal rays, core mid-sides where the core edges cross the axis rays.
    fibre = ray_points(shape, a, b, theta, corner_radius, np.arange(8) * math.pi / 4)
    core = core_ratio * fibre
    for k in range(0, 8, 2):
        p, q = core[k - 1], core[k + 1]
        direction = np.array(_DIRECTIONS[k], dtype=float)
        t, _ = np.linalg.solve(np.column_stack([q - p, -direction]), -p)
        core[k] = p + t * (q - p)

    # Interface spacing at each ray (mean of the two neighbouring sectors).
    sector = [_arc_length(shape, a, b, theta, corner_radius, k * math.pi / 4,
                          (k + 1) * math.pi / 4) for k in range(8)]
    spacing = [(sector[k - 1] + sector[k]) / 2 / (n_arc - 1) for k in range(8)]
    edge = half * np.array(_DIRECTIONS, dtype=float)
    fibre_growth = [_growth(np.linalg.norm(fibre[k] - core[k]), spacing[k], n_fibre - 1,
                            max_growth) for k in range(8)]
    matrix_growth = [_growth(np.linalg.norm(edge[k] - fibre[k]), spacing[k], n_matrix - 1,
                             max_growth) for k in range(8)]
    # Smooth across neighbouring rays: abrupt grading changes fold cells at sharp tips.
    ray_progression = tuple([(q[k - 1] + 2 * q[k] + q[(k + 1) % 8]) / 4 for k in range(8)]
                            for q in (fibre_growth, matrix_growth))

    def make_arcs(fibre_tags, center):
        geo = gmsh.model.geo
        nxt = [(k + 1) % 8 for k in range(8)]
        if shape == "ellipse" and a == b:
            return [geo.addCircleArc(fibre_tags[k], center, fibre_tags[nxt[k]]) for k in range(8)]
        if shape == "ellipse":
            major = geo.addPoint(a * math.cos(math.radians(theta)),
                                 a * math.sin(math.radians(theta)), 0)
            return [geo.addEllipseArc(fibre_tags[k], center, major, fibre_tags[nxt[k]])
                    for k in range(8)]
        arcs = []
        for k in range(8):
            phi = _resample(shape, a, b, theta, corner_radius, k * math.pi / 4,
                            (k + 1) * math.pi / 4, n_spline)
            inner = [geo.addPoint(x, y, 0)
                     for x, y in ray_points(shape, a, b, theta, corner_radius, phi[1:-1])]
            arcs.append(geo.addSpline([fibre_tags[k]] + inner + [fibre_tags[nxt[k]]]))
        return arcs

    mesh = mesh_rve(f"rve_{shape}_a{a:g}_b{b:g}_t{theta:g}", [tuple(p) for p in core],
                    [tuple(p) for p in fibre], make_arcs, rve_size=rve_size, n_arc=n_arc,
                    n_fibre=n_fibre, n_matrix=n_matrix, ray_progression=ray_progression,
                    msh_path=msh_path)
    interface = mesh["coords"][:, mesh["interface_j"]]
    error = np.abs(fibre_sdf(shape, a, b, theta, corner_radius, interface[:, 0], interface[:, 1]))
    mesh.update(shape=shape, a=a, b=b, theta=theta, corner_radius=corner_radius,
                boundary_error=float(error.max() if shape != "ellipse" else 0.0),
                label=f"{shape} a={a:.3g} b={b:.3g} θ={theta:.0f}°")
    return mesh


def _arc_length(shape, a, b, theta, corner_radius, phi0, phi1):
    xy = ray_points(shape, a, b, theta, corner_radius, np.linspace(phi0, phi1, 2001))
    return float(np.hypot(*np.diff(xy, axis=0).T).sum())


def _growth(length, first, cells, max_growth):
    """Geometric ratio q with first * (1 + q + ... + q^(cells-1)) = length, in [1, max_growth]."""
    if first * cells >= length or max_growth == 1:
        return 1.0
    lo, hi = 1.0, max_growth
    if first * (hi ** cells - 1) / (hi - 1) <= length:
        return hi
    for _ in range(60):
        q = (lo + hi) / 2
        lo, hi = (q, hi) if first * (q ** cells - 1) / (q - 1) < length else (lo, q)
    return (lo + hi) / 2


def _resample(shape, a, b, theta, corner_radius, phi0, phi1, n):
    """n ray angles in [phi0, phi1] spaced by arc length plus turning angle.

    Uniform arc length alone would leave tight rounded corners with very few
    spline points; weighting the turning angle keeps them well resolved.
    """
    phi = np.linspace(phi0, phi1, 2001)
    xy = ray_points(shape, a, b, theta, corner_radius, phi)
    step = np.diff(xy, axis=0)
    heading = np.unwrap(np.arctan2(step[:, 1], step[:, 0]))
    measure = np.concatenate([[0], np.cumsum(np.hypot(*step.T)
                                             + 0.25 * b * np.abs(np.diff(heading, prepend=heading[0])))])
    return np.interp(np.linspace(0, measure[-1], n), measure, phi)


def _random_fibre(rng, args):
    """Draw (shape, a, b, theta, corner_radius) from the sampling ranges."""
    shape = str(rng.choice(args.shapes))
    a = rng.uniform(*args.length_range) * args.rve_size / 2
    b = a / rng.uniform(*args.aspect_range)
    rf = rng.uniform(*args.corner_range) if shape == "rounded_rect" else 0.0
    return shape, round(a, 5), round(b, 5), round(rng.uniform(*args.theta_range), 3), \
        round(rf * b, 5)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", nargs="*",
                        default=["ellipse", "stadium", "rounded_rect", "ellipse", "stadium",
                                 "rounded_rect", "ellipse", "stadium", "rounded_rect"],
                        choices=SHAPES, help="fibre shape (one value per mesh)")
    parser.add_argument("--a", type=float, nargs="*",
                        default=[0.3, 0.3, 0.27, 0.25, 0.28, 0.3, 0.22, 0.3, 0.28],
                        help="half major length (2a <= max_length x RVE size)")
    parser.add_argument("--b", type=float, nargs="*",
                        default=[0.18, 0.15, 0.18, 0.25, 0.17, 0.15, 0.11, 0.15, 0.25],
                        help="half width (a/2 <= b <= a with the default aspect limit)")
    parser.add_argument("--theta", type=float, nargs="*",
                        default=[30, 60, 15, 0, 135, 45, 90, 100, 20],
                        help="major-axis angle in degrees from +x")
    parser.add_argument("--corner-radius", type=float, nargs="*",
                        default=[0, 0, 0.05, 0, 0, 0.03, 0, 0, 0.08],
                        help="R_f for rounded_rect (ignored for other shapes)")
    parser.add_argument("--samples", type=int, default=0,
                        help="if > 0, draw this many random fibres instead of the lists above")
    parser.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=SHAPES,
                        help="with --samples: shapes to draw from (uniformly)")
    parser.add_argument("--length-range", type=float, nargs=2, default=(0.3, 0.6),
                        help="with --samples: major length 2a as a fraction of the RVE size")
    parser.add_argument("--aspect-range", type=float, nargs=2, default=(1.0, 2.0),
                        help="with --samples: aspect ratio a/b range")
    parser.add_argument("--theta-range", type=float, nargs=2, default=(0.0, 180.0),
                        help="with --samples: rotation range in degrees")
    parser.add_argument("--corner-range", type=float, nargs=2, default=(0.2, 0.6),
                        help="allowed (and, with --samples, drawn) R_f / b for rounded_rect")
    parser.add_argument("--min-quality", type=float, default=0.3,
                        help="with --samples: redraw fibres whose min scaled Jacobian is lower")
    parser.add_argument("--max-element-ratio", type=float, default=6.0,
                        help="with --samples: redraw fibres whose largest element edge ratio "
                             "(longest / shortest edge) is higher")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rve-size", type=float, default=1.0, help="side of the square RVE")
    parser.add_argument("--max-length", type=float, default=0.6,
                        help="largest major length 2a as a fraction of the RVE size")
    parser.add_argument("--max-fibre-aspect", type=float, default=2.0,
                        help="largest fibre aspect ratio a/b")
    parser.add_argument("--min-gap", type=float, default=0.04,
                        help="smallest distance between fibre and RVE edge")
    parser.add_argument("--n-arc", type=int, default=17)
    parser.add_argument("--n-fibre", type=int, default=13)
    parser.add_argument("--n-matrix", type=int, default=13)
    parser.add_argument("--max-growth", type=float, default=1.1,
                        help="largest radial cell growth ratio away from the interface")
    parser.add_argument("--core-ratio", type=float, default=0.6)
    parser.add_argument("--n-spline", type=int, default=40)
    parser.add_argument("--out", default="rve_shape_meshes")
    parser.add_argument("--plot", action="store_true", help="save a figure of the first 9 meshes")
    args = parser.parse_args()

    if args.samples <= 0:
        lists = [args.shape, args.a, args.b, args.theta, args.corner_radius]
        if len({len(values) for values in lists}) != 1:
            parser.error("--shape, --a, --b, --theta and --corner-radius need the same "
                         "number of values.")
        fibres = list(zip(*lists))
    if args.samples > 0 and args.length_range[1] > args.max_length:
        parser.error("--length-range upper bound must not exceed --max-length.")
    if args.samples > 0 and not (1 <= args.aspect_range[0] <= args.aspect_range[1]
                                 <= args.max_fibre_aspect):
        parser.error("--aspect-range must lie within [1, --max-fibre-aspect].")
    options = dict(rve_size=args.rve_size, max_length=args.max_length,
                   max_fibre_aspect=args.max_fibre_aspect, min_gap=args.min_gap,
                   n_arc=args.n_arc,
                   n_fibre=args.n_fibre, n_matrix=args.n_matrix, max_growth=args.max_growth,
                   core_ratio=args.core_ratio, corner_range=tuple(args.corner_range),
                   n_spline=args.n_spline)

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    rejected = 0
    try:
        meshes = []
        for k in range(args.samples if args.samples > 0 else len(fibres)):
            msh_path = os.path.join(args.out, f"rve_{k:04d}.msh")
            if args.samples <= 0:
                meshes.append(build_shape_mesh(*fibres[k], **options, msh_path=msh_path))
                continue
            while True:  # redraw fibres that do not fit or give poor elements
                if rejected > 20 * args.samples:
                    raise RuntimeError("Too many rejected fibres; relax the sampling ranges.")
                try:
                    mesh = build_shape_mesh(*_random_fibre(rng, args), **options,
                                            msh_path=msh_path)
                except ValueError:
                    rejected += 1
                    continue
                if (mesh["min_scaled_jacobian"] >= args.min_quality and mesh["min_area"] > 0
                        and mesh["max_edge_ratio"] <= args.max_element_ratio):
                    break
                rejected += 1
            meshes.append(mesh)
    finally:
        gmsh.finalize()

    save_dataset(meshes, os.path.join(args.out, "rve_meshes.npz"),
                 shape=np.array([m["shape"] for m in meshes]),
                 shape_id=np.array([SHAPES.index(m["shape"]) for m in meshes]),
                 a=np.array([m["a"] for m in meshes]), b=np.array([m["b"] for m in meshes]),
                 theta=np.array([m["theta"] for m in meshes]),
                 corner_radius=np.array([m["corner_radius"] for m in meshes]),
                 aspect_ratio=np.array([m["a"] / m["b"] for m in meshes]),
                 volume_fraction=np.array([fibre_area(m["shape"], m["a"], m["b"],
                                                      m["corner_radius"]) for m in meshes])
                 / args.rve_size ** 2,
                 rve_size=args.rve_size)
    for m in meshes:
        if (m["min_scaled_jacobian"] < args.min_quality
                or m["max_edge_ratio"] > args.max_element_ratio):
            print(f"Warning: {m['label']} has min scaled Jacobian "
                  f"{m['min_scaled_jacobian']:.2f}, max edge ratio {m['max_edge_ratio']:.1f}")
    if args.samples > 0:
        print(f"Redrawn fibres (no fit, min scaled Jacobian < {args.min_quality} or edge "
              f"ratio > {args.max_element_ratio}): {rejected}")
    print("Max interface node distance from the exact fibre boundary: "
          f"{max(m['boundary_error'] for m in meshes):.2e}")
    if args.plot:
        plot_meshes(meshes[:9], os.path.join(args.out, "rve_meshes.png"))
        print(f"Figure: {os.path.join(args.out, 'rve_meshes.png')}")


if __name__ == "__main__":
    main()
