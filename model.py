"""FFT operator on a known body-fitted parameterization; no training side effects."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import computational_grid


class SpectralConv2d(nn.Module):
    """Low-frequency complex channel mixing on (angle, radius) axes."""

    def __init__(self, width, modes_theta, modes_radial):
        super().__init__()
        if min(width, modes_theta, modes_radial) < 1:
            raise ValueError("Width and mode counts must be positive.")
        self.modes_theta, self.modes_radial = modes_theta, modes_radial
        shape = (width, width, modes_theta, modes_radial)
        self.positive = nn.Parameter(torch.randn(*shape, dtype=torch.cfloat) / width)
        self.negative = nn.Parameter(torch.randn(*shape, dtype=torch.cfloat) / width)

    def forward(self, x):
        batch, channels, nt, nr = x.shape
        mt, mr = self.modes_theta, self.modes_radial
        if 2 * mt > nt or mr > nr // 2 + 1:
            raise ValueError("Fourier modes exceed grid capacity or angular blocks overlap.")
        spectrum = torch.fft.rfft2(x)
        result = spectrum.new_zeros(batch, channels, nt, nr // 2 + 1)
        result[:, :, :mt, :mr] = torch.einsum(
            "bctr,cotr->botr", spectrum[:, :, :mt, :mr], self.positive)
        result[:, :, -mt:, :mr] = torch.einsum(
            "bctr,cotr->botr", spectrum[:, :, -mt:, :mr], self.negative)
        return torch.fft.irfft2(result, s=(nt, nr))


class PlaneStressGeoFNO(nn.Module):
    """Predict [ux,uy] on supplied physical mesh nodes.

    coords: [B,T,R,2], ordered in angle and from hole to plate.
    conditions: [B,T,R,C], user-defined material/loading/BC features.
    periodic=True: full ring, the angular axis wraps around (no angular padding,
    cos/sin angle features). periodic=False: an open angular axis such as a
    quarter model between two symmetry edges; it is zero-padded like the radial
    axis so the FFT does not couple the two edges, and the angle feature is a
    plain 0..1 coordinate.
    Geometry and displacement use a consistent user-chosen system of units.
    No boundary conditions are imposed automatically.
    """

    def __init__(self, condition_channels=0, width=32, modes_theta=16,
                 modes_radial=12, depth=4, radial_padding=8, periodic=True):
        super().__init__()
        if condition_channels < 0 or depth < 1 or radial_padding < 0:
            raise ValueError("Invalid condition count, depth, or padding.")
        self.condition_channels = condition_channels
        self.radial_padding = radial_padding
        self.angular_padding = 0 if periodic else radial_padding
        self.periodic = periodic
        # X,Y plus angular encoding (cos/sin if periodic, else xi) and radial eta.
        self.lift = nn.Linear((5 if periodic else 4) + condition_channels, width)
        self.spectral = nn.ModuleList([
            SpectralConv2d(width, modes_theta, modes_radial) for _ in range(depth)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.project = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 2))

    def forward(self, coords, conditions=None):
        if coords.ndim != 4 or coords.shape[-1] != 2:
            raise ValueError("coords must have shape [batch,theta,radial,2].")
        if not coords.is_floating_point():
            raise ValueError("coords must be floating point.")
        batch, nt, nr, _ = coords.shape
        grid = computational_grid(nt, nr, periodic=self.periodic,
                                  device=coords.device, dtype=coords.dtype)
        if self.periodic:
            angle = 2 * math.pi * grid[..., :1]
            position = torch.cat((angle.cos(), angle.sin(), grid[..., 1:]), dim=-1)
        else:
            position = grid
        features = [coords, position.unsqueeze(0).expand(batch, -1, -1, -1)]
        if self.condition_channels:
            expected = (batch, nt, nr, self.condition_channels)
            if conditions is None or conditions.shape != expected:
                raise ValueError(f"conditions must have shape {expected}.")
            features.append(conditions)
        elif conditions is not None:
            raise ValueError("Configure condition_channels before supplying conditions.")
        x = self.lift(torch.cat(features, dim=-1)).permute(0, 3, 1, 2)
        x = F.pad(x, (0, self.radial_padding, 0, self.angular_padding))
        for index, (spectral, local) in enumerate(zip(self.spectral, self.local)):
            x = spectral(x) + local(x)
            if index < len(self.spectral) - 1:
                x = F.gelu(x)
        x = x[..., :nt, :nr]
        return self.project(x.permute(0, 2, 3, 1))
