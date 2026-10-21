"""Analytic maps from a rectangular computational grid to a perforated plate."""

import math

import torch


def computational_grid(n_theta, n_radial, *, periodic=True, device=None, dtype=torch.float32):
    """Return [T,R,2] coordinates (angular fraction, radial fraction).

    periodic=True (full ring): the angular seam is NOT duplicated.
    periodic=False (e.g. quarter model): both angular endpoints ARE included.
    Radial endpoints are always included.
    """
    if n_theta < (8 if periodic else 3) or n_radial < 3:
        raise ValueError("Use at least 8 (periodic) or 3 angular nodes and 3 radial nodes.")
    if periodic:
        angular = torch.arange(n_theta, device=device, dtype=dtype) / n_theta
    else:
        angular = torch.linspace(0, 1, n_theta, device=device, dtype=dtype)
    radial = torch.linspace(0, 1, n_radial, device=device, dtype=dtype)
    return torch.stack(torch.meshgrid(angular, radial, indexing="ij"), dim=-1)


def make_plate_mesh(n_theta=128, n_radial=48, *, plate_size=(2.0, 2.0),
                    hole="circle", hole_size=(0.25, 0.25), exponent=4.0,
                    rotation=0.0, radial_power=1.0, hole_radius=None, quarter=False,
                    device=None, dtype=torch.float32):
    """Return physical coordinates [T,R,2], centered at (0,0).

    quarter=True returns only the doubly symmetric quarter x >= 0, y >= 0:
    i=0 on the y=0 symmetry edge, i=-1 on the x=0 edge. Use periodic=False in
    the model and in plane_stress_fields for such grids.

    hole_size contains semi-axes. `hole` is circle, ellipse, or superellipse.
    A callable hole_radius(theta) overrides the named hole and must return a
    positive radius per angle. It describes a star-shaped hole, not point data.
    radial_power > 1 clusters nodes near the hole. Angles are in radians.
    """
    width, height = plate_size
    a, b = hole_size
    if min(width, height, a, b, radial_power) <= 0 or exponent < 2:
        raise ValueError("Sizes/power must be positive; superellipse exponent must be >= 2.")
    if hole not in {"circle", "ellipse", "superellipse"}:
        raise ValueError("hole must be circle, ellipse, or superellipse.")
    if hole_radius is None and hole == "circle" and a != b:
        raise ValueError("A circle requires equal hole semi-axes.")
    if quarter and rotation != 0:
        raise ValueError("A rotated hole is not doubly symmetric; use quarter=False.")
    grid = computational_grid(n_theta, n_radial, periodic=not quarter,
                              device=device, dtype=dtype)
    theta = (math.pi / 2 if quarter else 2 * math.pi) * grid[:, 0, 0]
    direction = torch.stack((theta.cos(), theta.sin()), dim=-1)
    # Intersect rays with the axis-aligned rectangular outer boundary.
    eps = torch.finfo(dtype).eps
    outer = torch.minimum((width / 2) / direction[:, 0].abs().clamp_min(eps),
                          (height / 2) / direction[:, 1].abs().clamp_min(eps))
    if hole_radius is not None:
        inner = torch.as_tensor(hole_radius(theta), device=device, dtype=dtype)
        if inner.shape != theta.shape:
            raise ValueError("hole_radius(theta) must return shape [n_theta].")
    else:
        angle = theta - rotation
        p = exponent if hole == "superellipse" else 2.0
        inner = ((angle.cos().abs() / a).pow(p)
                 + (angle.sin().abs() / b).pow(p)).pow(-1 / p)
    if not torch.isfinite(inner).all() or not ((inner > 0) & (inner < outer)).all():
        raise ValueError("The hole must lie strictly inside the plate along every sampled ray.")
    fraction = grid[..., 1].pow(radial_power)
    radius = inner[:, None] * (1 - fraction) + outer[:, None] * fraction
    return radius[..., None] * direction[:, None, :]
