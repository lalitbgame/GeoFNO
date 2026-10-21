"""Small-strain isotropic plane stress from mapped-grid finite differences."""

import torch


def _derivatives(field, periodic):
    """Differentiate [B,T,R,C] on uniform computational coordinates.

    Angular: periodic centred differences on a full ring, otherwise second-order
    differences that are one-sided at the two open edges (e.g. symmetry planes).
    """
    nt, nr = field.shape[1:3]
    if periodic:
        angular = (torch.roll(field, -1, 1) - torch.roll(field, 1, 1)) * (nt / 2)
    else:
        angular = torch.gradient(field, spacing=(1 / (nt - 1),), dim=(1,), edge_order=2)[0]
    radial = torch.gradient(field, spacing=(1 / (nr - 1),), dim=(2,), edge_order=2)[0]
    return angular, radial


def plane_stress_fields(displacement, coords, young_modulus, poisson_ratio, *,
                        periodic=True, jacobian_tolerance=1e-10):
    """Return strain/stress [xx,yy,xy] and von Mises [B,T,R].

    strain[...,2] is engineering shear gamma_xy, stress[...,2] is tau_xy.
    Material scalars or fields broadcastable to [B,T,R] are accepted. Use
    [B,1,1] for one scalar per sample. No physical derivatives are taken by
    autograd with respect to model input coordinates: these are spatial finite
    differences of the resulting field, transformed using the mesh Jacobian.
    Use periodic=False for an open angular axis such as a quarter model.
    """
    if displacement.shape != coords.shape or coords.ndim != 4 or coords.shape[-1] != 2:
        raise ValueError("displacement and coords must both have shape [B,T,R,2].")
    if coords.shape[1] < (8 if periodic else 3) or coords.shape[2] < 3:
        raise ValueError("At least 8 (periodic) or 3 angular and 3 radial nodes are required.")
    if jacobian_tolerance <= 0:
        raise ValueError("jacobian_tolerance must be positive.")
    young = torch.as_tensor(young_modulus, dtype=coords.dtype, device=coords.device)
    nu = torch.as_tensor(poisson_ratio, dtype=coords.dtype, device=coords.device)
    if not torch.isfinite(young).all() or not (young > 0).all():
        raise ValueError("Young's modulus must be finite and positive.")
    if not torch.isfinite(nu).all() or not ((nu > -1) & (nu < 0.5)).all():
        raise ValueError("For stable isotropic elasticity use -1 < nu < 0.5.")
    a, b = _derivatives(coords, periodic)
    du, dv = _derivatives(displacement, periodic)
    # a=(x_xi,y_xi), b=(x_eta,y_eta); invert the coordinate Jacobian.
    determinant = a[..., 0] * b[..., 1] - b[..., 0] * a[..., 1]
    if not torch.isfinite(determinant).all() or (determinant.abs() <= jacobian_tolerance).any():
        raise ValueError("Mesh has a singular/near-singular discrete Jacobian.")
    if not ((determinant > 0).all() or (determinant < 0).all()):
        raise ValueError("Mesh orientation changes: check for folded elements.")
    grad_x = (du * b[..., 1:] - dv * a[..., 1:]) / determinant[..., None]
    grad_y = (dv * a[..., :1] - du * b[..., :1]) / determinant[..., None]
    exx, eyy = grad_x[..., 0], grad_y[..., 1]
    gamma = grad_y[..., 0] + grad_x[..., 1]
    factor = young / (1 - nu.square())
    sxx = factor * (exx + nu * eyy)
    syy = factor * (eyy + nu * exx)
    txy = young / (2 * (1 + nu)) * gamma
    stress = torch.stack(torch.broadcast_tensors(sxx, syy, txy), dim=-1)
    mises = (sxx.square() - sxx * syy + syy.square() + 3 * txy.square()).clamp_min(0).sqrt()
    return {"strain": torch.stack((exx, eyy, gamma), dim=-1),
            "stress": stress, "von_mises": mises}
