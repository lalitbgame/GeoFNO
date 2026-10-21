"""Run from repository root: python -m unittest plane_stress_geofno.test_core."""

import unittest

import torch

from .geometry import make_plate_mesh
from .model import PlaneStressGeoFNO
from .physics import plane_stress_fields


class CoreTests(unittest.TestCase):
    def test_mesh_boundaries(self):
        for hole in ("circle", "ellipse", "superellipse"):
            xy = make_plate_mesh(64, 16, hole=hole, dtype=torch.float64)
            a = b = 0.25
            p = 4 if hole == "superellipse" else 2
            inner = (xy[:, 0, 0].abs() / a).pow(p) + (xy[:, 0, 1].abs() / b).pow(p)
            torch.testing.assert_close(inner, torch.ones_like(inner))
            torch.testing.assert_close(xy[:, -1].abs().amax(-1), torch.ones(64, dtype=xy.dtype))
            self.assertFalse(torch.equal(xy[0], xy[-1]))
        with self.assertRaises(ValueError):
            make_plate_mesh(hole_size=(2, 2))

    def test_rotated_and_custom_holes(self):
        xy = make_plate_mesh(64, 12, plate_size=(3.0, 2.0), hole="ellipse",
                             hole_size=(0.4, 0.2), rotation=0.4, dtype=torch.float64)
        inner = xy[:, 0]
        angle = inner.new_tensor(0.4)
        x = inner[:, 0] * angle.cos() + inner[:, 1] * angle.sin()
        y = -inner[:, 0] * angle.sin() + inner[:, 1] * angle.cos()
        torch.testing.assert_close((x / 0.4).square() + (y / 0.2).square(),
                                   torch.ones_like(x))
        shape = lambda theta: 0.25 * (1 + 0.1 * torch.cos(3 * theta))
        custom = make_plate_mesh(64, 12, hole_radius=shape)
        theta = torch.arange(64) * (2 * torch.pi / 64)
        torch.testing.assert_close(custom[:, 0].norm(dim=-1), shape(theta))

    def test_affine_displacement_has_exact_constant_strain(self):
        xy = make_plate_mesh(64, 16, hole="ellipse", hole_size=(0.3, 0.2),
                             dtype=torch.float64).unsqueeze(0)
        matrix = torch.tensor([[0.02, 0.03], [-0.01, -0.004]], dtype=xy.dtype)
        u = xy @ matrix.T + xy.new_tensor([0.7, -0.4])
        fields = plane_stress_fields(u, xy, 200.0, 0.3)
        strain = xy.new_tensor([0.02, -0.004, 0.02]).expand(1, 64, 16, 3)
        torch.testing.assert_close(fields["strain"], strain)
        sxx = 200 / (1 - 0.3**2) * (0.02 + 0.3 * -0.004)
        syy = 200 / (1 - 0.3**2) * (-0.004 + 0.3 * 0.02)
        txy = 200 / (2 * 1.3) * 0.02
        expected = xy.new_tensor([sxx, syy, txy]).expand_as(strain)
        torch.testing.assert_close(fields["stress"], expected)
        vm = (sxx**2 - sxx * syy + syy**2 + 3 * txy**2)**0.5
        torch.testing.assert_close(fields["von_mises"], torch.full_like(u[..., 0], vm))

    def test_rigid_rotation_and_translation_have_zero_stress(self):
        xy = make_plate_mesh(32, 12).unsqueeze(0)
        u = torch.stack((-0.01 * xy[..., 1], 0.01 * xy[..., 0]), -1) + 0.1
        fields = plane_stress_fields(u, xy, 10.0, 0.2)
        torch.testing.assert_close(fields["stress"], torch.zeros_like(fields["stress"]),
                                   atol=2e-5, rtol=0)

    def test_forward_backward_and_resolution(self):
        model = PlaneStressGeoFNO(condition_channels=2, width=8,
                                  modes_theta=4, modes_radial=4, depth=2)
        for nt, nr in ((24, 9), (32, 12)):
            xy = make_plate_mesh(nt, nr).unsqueeze(0).expand(2, -1, -1, -1)
            conditions = torch.zeros(2, nt, nr, 2)
            model.zero_grad(set_to_none=True)
            u = model(xy, conditions)
            self.assertEqual(u.shape, xy.shape)
            fields = plane_stress_fields(u, xy, 1.0, 0.3)
            (u.square().mean() + fields["stress"].square().mean()).backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        with self.assertRaises(ValueError):
            model(xy)
        with self.assertRaises(ValueError):
            PlaneStressGeoFNO(modes_theta=20)(xy)

    def test_quarter_mesh_boundaries(self):
        xy = make_plate_mesh(33, 12, plate_size=(200.0, 200.0), hole="ellipse",
                             hole_size=(50.0, 20.0), quarter=True, dtype=torch.float64)
        zeros = torch.zeros(12, dtype=xy.dtype)
        torch.testing.assert_close(xy[0, :, 1], zeros)                    # y=0 symmetry edge
        torch.testing.assert_close(xy[-1, :, 0], zeros, atol=1e-12, rtol=0)  # x=0 edge
        torch.testing.assert_close(xy[:, -1].amax(-1), torch.full((33,), 100.0, dtype=xy.dtype))
        inner = (xy[:, 0, 0] / 50).square() + (xy[:, 0, 1] / 20).square()
        torch.testing.assert_close(inner, torch.ones_like(inner))
        with self.assertRaises(ValueError):
            make_plate_mesh(33, 12, hole="ellipse", hole_size=(0.3, 0.2),
                            rotation=0.2, quarter=True)

    def test_quarter_affine_strain_needs_open_angular_axis(self):
        xy = make_plate_mesh(33, 16, plate_size=(200.0, 200.0), hole="ellipse",
                             hole_size=(50.0, 20.0), radial_power=1.5, quarter=True,
                             dtype=torch.float64).unsqueeze(0)
        matrix = torch.tensor([[0.02, 0.03], [-0.01, -0.004]], dtype=xy.dtype)
        u = xy @ matrix.T
        strain = xy.new_tensor([0.02, -0.004, 0.02]).expand(1, 33, 16, 3)
        fields = plane_stress_fields(u, xy, 200.0, 0.3, periodic=False)
        torch.testing.assert_close(fields["strain"], strain)
        # Treating the quarter as a ring wraps the y=0 edge onto the x=0 edge;
        # the Jacobian check rejects that instead of returning wrong strains.
        with self.assertRaisesRegex(ValueError, "orientation"):
            plane_stress_fields(u, xy, 200.0, 0.3)

    def test_quarter_model_forward_backward(self):
        model = PlaneStressGeoFNO(condition_channels=2, width=8, modes_theta=4,
                                  modes_radial=4, depth=2, periodic=False)
        self.assertEqual(model.lift.in_features, 2 + 2 + 2)
        for nt, nr in ((33, 48), (17, 24)):
            xy = make_plate_mesh(nt, nr, plate_size=(200.0, 200.0), hole="ellipse",
                                 hole_size=(50.0, 20.0), quarter=True)
            xy = xy.unsqueeze(0).expand(2, -1, -1, -1) / 100
            model.zero_grad(set_to_none=True)
            u = model(xy, torch.zeros(2, nt, nr, 2))
            self.assertEqual(u.shape, xy.shape)
            fields = plane_stress_fields(u, xy, 1.0, 0.3, periodic=False)
            (u.square().mean() + fields["stress"].square().mean()).backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
