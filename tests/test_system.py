"""Tests for the scene layer: geometry, null momenta, image formation."""
import numpy as np
import pytest

import grayt
from grayt import _core
from grayt.system import STATUS_SCREEN


class TestNullMomentum:
    @pytest.mark.parametrize("a", [0.0, 0.9])
    def test_constructed_momentum_is_null(self, a):
        bh = grayt.BlackHole(a)
        for point, direc in [((-500.0, 50.0, 3.0), (1.0, 0.0, 0.0)),
                             ((100.0, -80.0, 40.0), (-0.5, 0.7, -0.1))]:
            y0, pt, pphi = grayt.null_momentum(bh, point, direc)
            gu, _, _ = _core.kerr_metric.metric_contra(a, y0[1], y0[2])
            h = 0.5*(gu[0]*pt**2 + 2*gu[1]*pt*pphi + gu[2]*y0[4]**2 +
                     gu[3]*y0[5]**2 + gu[4]*pphi**2)
            assert abs(h) < 1e-12

    def test_weak_field_deflection(self):
        # alpha ~ 4M/b + 15 pi M^2 / (4 b^2) for b = 50 -> 0.0847 rad
        sys3 = grayt.System(
            physical=grayt.PhysicalSystem(spacetime=grayt.BlackHole(0.0)))
        ray = sys3.trace_ray((-1000.0, 50.0, 0.0), (1.0, 0.0, 0.0),
                             lambda_max=2500.0)
        d1 = ray.points[-1] - ray.points[-2]
        d1 = d1/np.linalg.norm(d1)
        alpha = np.arccos(np.clip(d1 @ np.array([1.0, 0, 0]), -1, 1))
        expected = 4.0/50.0 + 15*np.pi/(4*50.0**2)
        assert alpha == pytest.approx(expected, rel=0.02)


class TestPlaneTracing:
    def test_hit_lands_on_plane(self):
        bh = grayt.BlackHole(0.5)
        y0, pt, pphi = grayt.null_momentum(bh, (-400.0, 200.0, -30.0),
                                           (1.0, 0.0, 0.0))
        st, yout, ns = _core.raytracer.trace_to_plane(
            bh.mid, bh.par, pt, pphi, y0, 1, 1e-9, 1e-11, 1.0,
            [1.0, 0.0, 0.0], 400.0, 5000.0, 100000)
        assert st == STATUS_SCREEN
        hit = grayt.bl_to_cart(yout[1], yout[2], yout[3])
        assert hit[0] == pytest.approx(400.0, abs=1e-6)

    def test_central_ray_is_captured(self):
        bh = grayt.BlackHole(0.0)
        y0, pt, pphi = grayt.null_momentum(bh, (-200.0, 0.0, 0.0),
                                           (1.0, 0.0, 0.0))
        st, yout, ns = _core.raytracer.trace_to_plane(
            bh.mid, bh.par, pt, pphi, y0, 1, 1e-9, 1e-11, 1.0,
            [1.0, 0.0, 0.0], 200.0, 2000.0, 100000)
        assert st == grayt.STATUS_CAPTURED


class TestScreen:
    def test_binning_roundtrip(self):
        scr = grayt.Screen(center=(10.0, 0, 0), normal=(1, 0, 0),
                           up=(0, 0, 1), width=4.0, height=4.0,
                           resolution=(8, 8))
        pts = scr.to_world(np.array([-1.5, 0.0, 1.5]),
                           np.array([-1.5, 0.0, 1.5]))
        n = scr.add_hits(pts, np.ones((3, 3)))
        assert n == 3
        assert scr._count.sum() == 3

    def test_far_field_mapping_is_nearly_straight(self):
        # A collimated bundle far from the hole must land where it left.
        bh = grayt.BlackHole(0.0)
        src_img = np.ones((8, 8, 3))
        source = grayt.ImageSource(center=(-100.0, 600.0, 0.0),
                                   normal=(1, 0, 0), up=(0, 0, 1),
                                   width=10.0, height=10.0, image=src_img,
                                   emission="collimated")
        screen = grayt.Screen(center=(100.0, 600.0, 0.0), normal=(1, 0, 0),
                              up=(0, 0, 1), width=12.0, height=12.0,
                              resolution=(6, 6))
        sys3 = grayt.System(physical=grayt.PhysicalSystem(spacetime=bh))
        stats = sys3.form_image(source, screen, max_rays=64, r_max=5000.0)
        # deflection ~4/600 rad over 200M path -> ~1.3M shift; all rays
        # must reach the (slightly larger) screen
        assert stats["status_counts"].get(STATUS_SCREEN, 0) == stats["n_rays"]
        assert stats["n_on_screen"] == stats["n_rays"]
