"""Untrained forward demonstration, not a physical solution or dataset."""

import torch

from .geometry import make_plate_mesh
from .model import PlaneStressGeoFNO
from .physics import plane_stress_fields


def main():
    torch.manual_seed(7)
    meshes = [make_plate_mesh(64, 24, hole=name, hole_size=size)
              for name, size in [("circle", (0.25, 0.25)),
                                 ("ellipse", (0.35, 0.2)),
                                 ("superellipse", (0.3, 0.25))]]
    coords = torch.stack(meshes)
    # Illustrative convention only: [scaled E, nu, scaled traction_x, traction_y].
    conditions = coords.new_tensor([1.0, 0.3, 0.01, 0.0]).expand(3, 64, 24, 4)
    model = PlaneStressGeoFNO(condition_channels=4, width=16,
                              modes_theta=8, modes_radial=8).eval()
    with torch.no_grad():
        displacement = model(coords, conditions)
        fields = plane_stress_fields(displacement, coords, 1.0, 0.3)
    print("UNTRAINED model: outputs have no physical predictive validity.")
    print("Coordinates:", tuple(coords.shape))
    print("Displacement [ux,uy]:", tuple(displacement.shape))
    for name, value in fields.items():
        print(f"{name}: {tuple(value.shape)}, finite={bool(torch.isfinite(value).all())}")

    # Doubly symmetric quarter of a 200 x 200 plate: open angular axis.
    quarter = torch.stack([make_plate_mesh(33, 48, plate_size=(200.0, 200.0), hole="ellipse",
                                           hole_size=size, quarter=True)
                           for size in [(50.0, 20.0), (20.0, 50.0)]]) / 100
    model = PlaneStressGeoFNO(width=16, modes_theta=8, modes_radial=8, periodic=False).eval()
    with torch.no_grad():
        displacement = model(quarter)
        fields = plane_stress_fields(displacement, quarter, 1.0, 0.3, periodic=False)
    print("Quarter model coordinates:", tuple(quarter.shape),
          "von Mises finite:", bool(torch.isfinite(fields["von_mises"]).all()))


if __name__ == "__main__":
    main()
