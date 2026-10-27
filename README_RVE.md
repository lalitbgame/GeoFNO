# 2D RVE mesh generator for Geo-FNO

`gmsh_rve_mesh.py` builds structured gmsh quad meshes of the square RVE in
`RVE.jpg`. The RVE is a square cell `[-L/2, L/2]²` (default `L = 1`) with a
central circular or elliptical fibre (semi-axes `a` along x, `b` along y).
**Both phases are meshed**: the fibre and the matrix.

Every mesh has the same node count, element count and connectivity, so a set
of meshes forms a Geo-FNO dataset. The grid follows the same conventions as
`plane_stress_geofno/` (the plate with a hole), so its model and stress
postprocessing work on it directly.

![Example RVE meshes](rve_meshes_example.png)

## Folder layout

| File | Purpose |
|---|---|
| `RVE.jpg` | Reference RVE shape (square cell, central fibre). |
| `gmsh_rve_mesh.py` | Mesh generator (CLI + `build_rve_mesh`). Needs only `gmsh` and `numpy` (`matplotlib` for `--plot`). |
| `rve_meshes_example.png` | Output of `--plot` with the default example shapes. |
| `Fiber_shapes.png` | Reference non-circular fibre shapes (stadium, ellipse, rounded rectangle). |
| `gmsh_rve_shapes.py` | Mesh generator for rotated non-circular fibres (CLI + `build_shape_mesh`); see [Non-circular fibres](#non-circular-fibres-gmsh_rve_shapespy). Imports the shared mesher from `gmsh_rve_mesh.py`. |
| `rve_shapes_example.png` | Output of `gmsh_rve_shapes.py --plot`. |
| `rve_shape_examples/` | The 9 example RVEs with different fibre shapes: `.msh` files, `rve_meshes.npz`, figure. |

## Installation

```bash
python3 -m pip install gmsh numpy matplotlib
# for the Geo-FNO model and stress postprocessing:
python3 -m pip install -r plane_stress_geofno/requirements.txt
```

## Quick start

Run from the **repository root**:

```bash
# 6 example shapes (circles and ellipses) + figure
python3 RVE/gmsh_rve_mesh.py --plot

# 500 random circular fibres, radius 0.1-0.45
python3 RVE/gmsh_rve_mesh.py --samples 500 --circle --range 0.1 0.45 --seed 0 --out rve_meshes

# 500 random ellipses
python3 RVE/gmsh_rve_mesh.py --samples 500 --range 0.1 0.45 --seed 0 --out rve_meshes
```

From Python:

```python
import sys, gmsh
sys.path.insert(0, "RVE")
from gmsh_rve_mesh import build_rve_mesh

gmsh.initialize()
mesh = build_rve_mesh(0.3, 0.3, msh_path="rve.msh")   # circular fibre, r = 0.3
gmsh.finalize()
mesh["coords"].shape, mesh["core_coords"].shape, mesh["nodes"].shape
# (64, 25, 2) (17, 17, 2) (1825, 2)
```

## Mesh layout

The mesh is an O-grid made of 8 sectors of 45°, like four copies of the quarter plate:

- 8 straight rays at 0°, 45°, …, 315° run from a small **core rectangle**
  inside the fibre, through the fibre boundary, to the RVE **edge midpoints**
  and **corners** (the blue and red points in `RVE.jpg`).
- Each sector has one **fibre patch** (core → interface) and one **matrix
  patch** (interface → RVE edge). The core rectangle is one more patch.
- All 17 patches are gmsh transfinite surfaces recombined into quads, with fixed
  divisions. Only node positions change with `a` and `b`.

The mesh is continuous and conforming: nodes on patch boundaries are shared.
The core rectangle is part of the fibre material. It is split off only so the
element divisions stay fixed. The **fibre/matrix interface** is the only
material boundary, and it is perfectly bonded (shared nodes).

In the figure:

| Line | Meaning |
|---|---|
| grey cells | fibre (`quad_phase = 1`) |
| green cells | matrix (`quad_phase = 0`) |
| red curve | fibre/matrix interface, ring `j = interface_j` |
| orange rectangle | core boundary, ring `j = 0` (start of the ring grid) |
| blue line | grid seam `i = 0` on the +x axis (not a mesh boundary) |

Default sizes:

| Quantity | Formula | Default |
|---|---|---|
| Ring grid `T × R` | `8·(n_arc−1) × (n_fibre+n_matrix−1)` | 64 × 25 |
| Core grid `C × C` | `C = 2·(n_arc−1)+1` | 17 × 17 |
| Nodes | `T·R + (C−2)²` | **1,825** |
| Quads | `T·(R−1) + (C−1)²` | **1,792** (768 fibre) |

## Command line

```bash
python3 RVE/gmsh_rve_mesh.py [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--a A [A ...]`, `--b B [B ...]` | six example shapes | explicit semi-axes, one pair per mesh |
| `--samples N` | 0 | if > 0, draw N random `(a, b)` pairs instead |
| `--range LO HI` | 0.1 0.45 | uniform range for random semi-axes |
| `--circle` | off | with `--samples`: circular fibres (`b = a`) |
| `--seed S` | 0 | random seed |
| `--rve-size` | 1.0 | side `L` of the square RVE |
| `--max-semi-axis` | 0.45 | largest allowed semi-axis (must be < `L/2`) |
| `--n-arc` | 9 | nodes per 45° sector edge |
| `--n-fibre` | 9 | radial nodes, core boundary → interface |
| `--n-matrix` | 17 | radial nodes, interface → RVE edge |
| `--progression` | 1.05 | refinement towards the interface (both phases) |
| `--core-ratio` | 0.6 | core corner position relative to the fibre point on the 45° ray, in (0, 1) |
| `--out DIR` | `rve_meshes` | output folder |
| `--plot` | off | save `rve_meshes.png` with the first 9 meshes |

Keep the mesh options fixed for a whole dataset. The script stops with an
error if any mesh differs in topology or has an inverted element.

## Outputs (in `--out`)

| File | Contents |
|---|---|
| `rve_0000.msh`, … | one gmsh mesh per fibre shape, with physical groups and `$Periodic` constraints |
| `rve_meshes.npz` | all meshes in grid and point-cloud form (below) |
| `rve_meshes.png` | with `--plot` only |

`rve_meshes.npz`:

| Key | Shape (defaults) | Meaning |
|---|---|---|
| `coords` | `[N,64,25,2]` | ring grid: `i=0` on the +x axis, `i` counter-clockwise, seam not duplicated; `j=0` core boundary, `j=interface_j` fibre boundary, `j=R−1` RVE edge |
| `core_coords` | `[N,17,17,2]` | core rectangle grid, `(x index, y index)`; its boundary is ring `j=0` |
| `nodes` | `[N,1825,2]` | all nodes in mesh order (point-cloud view) |
| `quads` | `[1792,4]` | shared connectivity, counter-clockwise, indices into `nodes` |
| `quad_phase` | `[1792]` | 1 = fibre, 0 = matrix |
| `ring_index` | `[64,25]` | ring grid position → node index (`coords = nodes[ring_index]`) |
| `core_index` | `[17,17]` | core grid position → node index |
| `periodic_x` | `[17,2]` | node pairs (left, right) with equal y |
| `periodic_y` | `[17,2]` | node pairs (bottom, top) with equal x |
| `interface_j` | scalar | ring index of the fibre/matrix interface (8) |
| `a`, `b`, `volume_fraction` | `[N]` | fibre semi-axes and `π·a·b / L²` |
| `rve_size` | scalar | `L` |

Physical groups in every `.msh`:

| Group | Dim | Location | Typical use |
|---|---|---|---|
| `fibre` | 2 | inclusion (core + fibre patches) | fibre material |
| `matrix` | 2 | everything outside the fibre | matrix material |
| `interface` | 1 | fibre boundary | post-processing interface stress |
| `left`, `right`, `bottom`, `top` | 1 | RVE edges | periodic or kinematic BCs |
| `corners` | 0 | four RVE corners | fix rigid-body motion |

Opposite edges have matching nodes (`right = left + (L, 0)`,
`top = bottom + (0, L)`), and the `.msh` file stores this as gmsh periodic
constraints.

## Using the dataset with Geo-FNO

1. Solve each `.msh` in your FEM code (for example, periodic BCs with an
   applied macroscopic strain). Assign `fibre` and `matrix` properties.
2. Store the nodal results in the node order of `nodes` (the `.msh` order of
   the nodes used by quads). Check that your FEM converter does not renumber
   nodes.
3. Reorder to the grids:

```python
import numpy as np
d = np.load("rve_meshes/rve_meshes.npz")
u_ring = u_nodes[:, d["ring_index"]]    # [N,64,25,2]
u_core = u_nodes[:, d["core_index"]]    # [N,17,17,2]
```

4. Train the known-mapping model on the ring grid. The angular axis wraps around,
   so use `periodic=True`:

```python
import torch
from plane_stress_geofno import PlaneStressGeoFNO, plane_stress_fields

coords = torch.tensor(d["coords"], dtype=torch.float32)                  # [N,T,R,2]
phase = (torch.arange(coords.shape[2]) <= int(d["interface_j"])).float()
phase = phase.expand(*coords.shape[:3])[..., None]                      # 1 = fibre
model = PlaneStressGeoFNO(condition_channels=1, modes_theta=16, modes_radial=8,
                          periodic=True)
u = model(coords, phase)                                                # [N,T,R,2]
```

With different fibre and matrix materials, pass per-node `E` and `ν`
(`[B,T,R]`) to `plane_stress_fields(..., periodic=True)`. At the interface
row, the strain is a one-mesh-row difference across a material jump. Treat
stresses on that row with care.

**About the core.** The core rectangle (`core_coords`) is not part of the
`[T,R]` grid. A single rectangular grid that follows a smooth inclusion boundary
would need a degenerate 180° cell, so the core is kept separate. You can cover it
in one of two ways:

- train a second small FNO on `core_coords` (a plain `[C,C]` grid, non-periodic), or
- use the point-cloud view `nodes` with the learned-deformation Geo-FNO in
  `elasticity/elas_geofno.py`, which covers every node.

## Verification

Checked with gmsh 4.15, NumPy 2.5, PyTorch (CPU):

- 300 random elliptical fibres (semi-axes 0.1–0.45): all had 1,825 nodes and
  1,792 quads with identical connectivity, and the minimum scaled Jacobian was 0.745.
- Opposite edge nodes matched to about 1e-12.
- An affine displacement on the ring grid gave exact strain (error ~1e-16)
  through `plane_stress_fields(..., periodic=True)`.
- `PlaneStressGeoFNO(periodic=True)` ran forward and backward on the ring grid
  with a phase condition channel.

## Non-circular fibres (`gmsh_rve_shapes.py`)

This script meshes the fibre shapes in `Fiber_shapes.png`. Each shape has length
`2a` along its major axis, width `2b` (`a ≥ b`, aspect ratio `a/b`) and a
rotation `θ`:

| Shape | Definition |
|---|---|
| `ellipse` | semi-axes `a`, `b` |
| `stadium` | `2(a−b) × 2b` rectangle with two semicircular caps of radius `b` |
| `rounded_rect` | `2a × 2b` rectangle with corner radius `R_f`, `0.2 b ≤ R_f ≤ 0.6 b` (default) |

`θ` is the angle of the major axis, in degrees, measured **counter-clockwise
from the +x axis**. If you measure the angle from the y axis instead, as the
sketch may suggest, use `θ_x = 90° − θ_y`.

**Size and shape limits.** Every fibre, explicit or random, must satisfy:

| Limit | Default | Option |
|---|---|---|
| major length `2a ≤ max_length × L` | 60% of the RVE side (`a ≤ 0.3` for `L = 1`) | `--max-length` |
| aspect ratio `1 ≤ a/b ≤ max_fibre_aspect` | 2 (`b ≥ a/2`) | `--max-fibre-aspect` |
| `rounded_rect` corner radius `R_f / b` in `corner_range` | 0.2–0.6 (`R_f` from 0.015 L to 0.18 L with the limits above) | `--corner-range` |

`build_shape_mesh` raises a `ValueError` for a fibre outside these limits, for
example `Major length 2a = 0.64 exceeds 0.6 x RVE size (0.6).` or
`Aspect ratio a/b = 2.14 exceeds 2.` or
`rounded_rect needs corner_radius / b in [0.2, 0.6]; got 0.133.` Random
sampling stays inside them.

![Example non-circular fibre meshes](rve_shapes_example.png)

```bash
python3 RVE/gmsh_rve_shapes.py --plot --out RVE/rve_shape_examples         # the 9 example fibres
python3 RVE/gmsh_rve_shapes.py --samples 500 --seed 0 --out rve_shape_meshes   # mixed shapes
python3 RVE/gmsh_rve_shapes.py --samples 500 --shapes stadium --aspect-range 1.5 2 --length-range 0.4 0.6
python3 RVE/gmsh_rve_shapes.py --shape rounded_rect --a 0.3 --b 0.15 --theta 30 --corner-radius 0.05
```

`rve_shape_examples/` holds the output of the first command: the 9 example
RVEs above as `rve_0000.msh` … `rve_0008.msh` (in the order of the figure), with
`rve_meshes.npz` and `rve_meshes.png`. These examples have element edge ratios of
at most 4.3 and a minimum scaled Jacobian of 0.76.

**Same approach, same dataset format.** The script calls the same
`mesh_rve` function as `gmsh_rve_mesh.py`. That function builds the 8-sector
O-grid, the transfinite patches, the physical groups, the periodic constraints
and the grid ordering. Only the fibre geometry is different:

- The fibre points are where the 8 fixed rays (0°, 45°, …) cross the fibre
  boundary. The rays still end at the RVE edge midpoints and corners.
- The core is a quadrilateral with its corners at `core_ratio ×` the fibre points on
  the diagonal rays. It rotates with the fibre.
- Ellipse boundaries are exact gmsh ellipse arcs. Stadium and rounded-rectangle
  boundaries are gmsh splines through `--n-spline` exact boundary points per
  sector, with extra points at the rounded corners. Interface nodes lie within
  about 3e-5 `L` of the exact boundary.

**Divisions and element shape.** Elongated fibres have a longer boundary than a
circle of the same area, so this script uses finer default divisions than
`gmsh_rve_mesh.py`. It also sets the radial grading separately on each ray, so
the elements keep a moderate aspect ratio:

| | `gmsh_rve_mesh.py` | `gmsh_rve_shapes.py` |
|---|---|---|
| `n_arc`, `n_fibre`, `n_matrix` | 9, 9, 17 | **17, 13, 13** |
| Ring `[T,R]`, core `[C,C]` | `[64,25]`, `[17,17]` | **`[128,25]`, `[33,33]`** |
| Nodes, quads | 1,825, 1,792 | **4,161, 4,096** (2,560 fibre) |
| Interface row `interface_j` | 8 | **12** |
| Radial grading | one `progression` (1.05) on every ray | per ray (below), at most `max_growth` (1.1) |

- **Per-ray grading.** On each ray, the first cell at the interface is as
  long as the local interface node spacing, so it is close to square. Cells grow
  geometrically away from the interface by at most `--max-growth`. The ratios
  are smoothed over neighbouring rays (weights 1-2-1), because abrupt changes
  fold cells at sharp tips. In thin gaps the cells are uniform.
- **Element-shape filter.** `--max-element-ratio` (default 6) is the largest allowed
  element edge ratio (longest / shortest edge). Random fibres above it, or
  below `--min-quality`, are redrawn. Explicit fibres are kept, but the script
  prints a warning. The remaining stretched elements come from the geometry,
  not the divisions: a fibre tip within a few hundredths of `L` of the RVE edge,
  a very thin fibre, or a rounded-rectangle corner with a small radius.

Measured before the size and shape limits were added, on 150 random fibres
(mixed shapes, volume fraction 0.1–0.5, aspect ratio 1–3, any rotation), as
element edge ratios. The limits remove the long, thin and near-edge fibres
that caused most stretched elements:

| Setting | 99th percentile | worst | fibres with an element > 6 |
|---|---|---|---|
| old: 9/9/17, progression 1.05 | 9.7 | 25.4 | 39% |
| new: 17/13/13, per-ray grading | 3.4 | 8.6 | 2% (redrawn by the filter) |

The mesh still uses the same `mesh_rve` function, ray layout, physical
groups, periodic constraints, grid ordering and `.npz` layout as
`gmsh_rve_mesh.py`. **To mix circular and non-circular fibres in one dataset,
generate both with the same `--n-arc/--n-fibre/--n-matrix`.** For example, run
`gmsh_rve_mesh.py --n-arc 17 --n-fibre 13 --n-matrix 13`, or circles from this
script (`--shapes ellipse --aspect-range 1 1`). The grading moves nodes but
never changes the topology. Everything in
[Using the dataset with Geo-FNO](#using-the-dataset-with-geo-fno) applies.
Use `interface_j` from the `.npz` rather than a hard-coded 8, and choose
`modes_theta` for `T = 128` (for example 16–32).

Options, in addition to `--rve-size`, `--n-arc`, `--n-fibre`, `--n-matrix`,
`--core-ratio`, `--seed`, `--out` and `--plot` (same meaning as in
`gmsh_rve_mesh.py`):

| Option | Default | Meaning |
|---|---|---|
| `--shape`, `--a`, `--b`, `--theta`, `--corner-radius` | 9 example fibres | explicit fibres, one value per mesh in each list (`--corner-radius` is ignored except for `rounded_rect`) |
| `--samples N` | 0 | if > 0, draw N random fibres instead |
| `--shapes` | all three | shapes to draw from (uniformly) |
| `--max-length` | 0.6 | largest major length `2a` / `L` |
| `--max-fibre-aspect` | 2 | largest aspect ratio `a/b` |
| `--length-range` | 0.3 0.6 | random major length `2a` / `L` (within `--max-length`) |
| `--aspect-range` | 1 2 | random aspect ratio `a/b` (within `--max-fibre-aspect`) |
| `--theta-range` | 0 180 | rotation in degrees |
| `--corner-range` | 0.2 0.6 | allowed `R_f / b` for `rounded_rect`; random fibres are drawn uniformly from it |
| `--min-gap` | 0.04 | smallest distance between the fibre and the RVE edge |
| `--max-growth` | 1.1 | largest radial cell growth ratio (replaces `--progression`) |
| `--min-quality` | 0.3 | redraw random fibres with a lower minimum scaled Jacobian |
| `--max-element-ratio` | 6 | redraw random fibres with a larger element edge ratio (longest / shortest edge) |
| `--n-spline` | 40 | spline points per sector (stadium, rounded rectangle) |

Random fibres are drawn by major length and aspect ratio, so the volume
fraction is an output. It is stored in the `.npz`; with the defaults it is
between about 0.035 (thin ellipse, `2a = 0.3 L`) and 0.36
(near-square rounded rectangle, `2a = 0.6 L`). Fibres that give poor elements are redrawn, and the
script prints how many. Check the final `volume_fraction` and `aspect_ratio`
distribution in the `.npz`.

Extra `.npz` keys (per sample): `shape` (string), `shape_id` (0 ellipse,
1 stadium, 2 rounded_rect), `a`, `b`, `theta`, `corner_radius`, `aspect_ratio`,
`volume_fraction` (exact shape area / `L²`).

Verified before the size and shape limits were added, on 500 random fibres
(volume fraction 0.1–0.5, aspect ratio 1–3). All had 4,161 nodes and 4,096
quads with identical connectivity, and every element was convex. The minimum
scaled Jacobian was 0.345. Element edge ratios had a mean of 1.44, a 99th
percentile of 3.39 and a maximum of 5.98. Interface nodes were within 2.5e-5 `L`
of the exact shape, and the periodic edges matched to 1e-12. An affine displacement gave exact
strain through `plane_stress_fields(..., periodic=True)`, and
`PlaneStressGeoFNO(periodic=True)` ran forward and backward.

## Limitations

- One central fibre: axis-aligned in `gmsh_rve_mesh.py`, rotated in `gmsh_rve_shapes.py`. Fibre shapes must be convex (each ray from the centre crosses the boundary once). Multiple fibres are not supported.
- Thin ligaments (semi-axis near `L/2`) compress the matrix cells. Check FEM
  convergence for the largest fibres.
- The core region needs separate handling for the known-mapping model (see above).
