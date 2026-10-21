"""Body-fitted Geo-FNO for a plate containing one central hole."""

from .geometry import computational_grid, make_plate_mesh
from .model import PlaneStressGeoFNO
from .physics import plane_stress_fields

__all__ = ["computational_grid", "make_plate_mesh", "PlaneStressGeoFNO", "plane_stress_fields"]
