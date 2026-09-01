"""Validation of the Fortran core against known analytic results."""
import numpy as np
import pytest

import grayt
from grayt import _core

KM = _core.kerr_metric
RT = _core.raytracer


class TestMetric:
    @pytest.mark.parametrize("a,r,th", [(0.0, 10.0, 1.2), (0.5, 4.0, 0.7),
                                        (0.95, 2.5, 1.5), (0.998, 1.8, 2.4)])
    def test_contravariant_is_inverse(self, a, r, th):
        gd = KM.metric_cov(a, r, th)
        gu, _, _ = KM.metric_contra(a, r, th)
        tp = np.array([[gd[0], gd[1]], [gd[1], gd[4]]])
        tp_inv = np.linalg.inv(tp)
        assert tp_inv[0, 0] == pytest.approx(gu[0], rel=1e-12)
        assert tp_inv[0, 1] == pytest.approx(gu[1], rel=1e-12)
        assert tp_inv[1, 1] == pytest.approx(gu[4], rel=1e-12)
        assert 1.0/gd[2] == pytest.approx(gu[2], rel=1e-12)
        assert 1.0/gd[3] == pytest.approx(gu[3], rel=1e-12)

    @pytest.mark.parametrize("a,r,th", [(0.0, 8.0, 1.0), (0.9, 3.0, 2.0)])
    def test_derivatives_match_finite_differences(self, a, r, th):
        _, dgr, dgt = KM.metric_contra(a, r, th)
        eps = 1e-6
        fd_r = (KM.metric_contra(a, r + eps, th)[0] -
                KM.metric_contra(a, r - eps, th)[0])/(2*eps)
        fd_t = (KM.metric_contra(a, r, th + eps)[0] -
                KM.metric_contra(a, r, th - eps)[0])/(2*eps)
        assert np.allclose(fd_r, dgr, rtol=1e-6, atol=1e-8)
        assert np.allclose(fd_t, dgt, rtol=1e-6, atol=1e-8)

    def test_horizon_and_isco(self):
        assert grayt.BlackHole(0.0).horizon == pytest.approx(2.0)
        assert grayt.BlackHole(0.0).isco == pytest.approx(6.0)
        assert grayt.BlackHole(0.5).isco == pytest.approx(4.233, abs=1e-3)
        assert grayt.BlackHole(0.95).isco == pytest.approx(1.9372, abs=1e-3)
        assert grayt.BlackHole(1.0).isco == pytest.approx(1.0)


class TestGeodesics:
    def test_hamiltonian_constraint_stays_small(self):
        # Escape orbit of Fig. 4 (a = 0.98, r0 = 100). The published
        # initial conditions are rounded to 6 decimals, so |H| starts at
        # a small constant offset; what must stay small is the drift.
        y0 = [0.0, 100.0, 1.570796, 1.570796, -1.009327, 1.87]
        out = grayt.trace(grayt.BlackHole(0.98), y0, p_t=-0.989953,
                          p_phi=2.000098, rtol=1e-11, atol=1e-13,
                          lambda_max=150.0)
        assert len(out["r"]) > 10
        drift = np.abs(out["herr"] - out["herr"][0]).max()
        assert drift < 1e-9

    def test_camera_rays_conserve_constraint(self):
        bh = grayt.BlackHole(0.5)
        cam = grayt.Camera()
        # Escaping ray: constraint must stay tiny along the whole path.
        y0, pt, pphi = grayt.camera_ray(bh, cam, 15.0, 6.0)
        st, _, hmax, _ = RT.trace_ray(bh.a, pt, pphi, y0, 1, 1e-10,
                                      1e-12, 1100.0, 0, 6.0, 20.0, 500000)
        assert st == grayt.STATUS_ESCAPED
        assert hmax < 1e-9
        # Captured ray: error grows in the final near-horizon plunge
        # (coordinate degeneracy, cf. Fig. 5 of the paper) but stays bounded.
        y0, pt, pphi = grayt.camera_ray(bh, cam, 5.0, 3.0)
        st, _, hmax, _ = RT.trace_ray(bh.a, pt, pphi, y0, 1, 1e-10,
                                      1e-12, 1100.0, 0, 6.0, 20.0, 500000)
        assert st == grayt.STATUS_CAPTURED
        assert hmax < 1e-6

    def test_initial_conditions_are_null(self):
        bh = grayt.BlackHole(0.9)
        cam = grayt.Camera()
        gd = KM.metric_cov(bh.a, cam.r, np.deg2rad(cam.theta))
        for x, y in [(0.0, 0.0), (10.0, -4.0), (-7.0, 2.0)]:
            y0, pt, pphi = grayt.camera_ray(bh, cam, x, y)
            gu, _, _ = KM.metric_contra(bh.a, y0[1], y0[2])
            h = 0.5*(gu[0]*pt**2 + 2*gu[1]*pt*pphi + gu[2]*y0[4]**2 +
                     gu[3]*y0[5]**2 + gu[4]*pphi**2)
            assert abs(h) < 1e-12


class TestShadow:
    def test_schwarzschild_shadow_radius(self):
        # Critical impact parameter sqrt(27) ~ 5.196
        bh = grayt.BlackHole(0.0)
        cam = grayt.Camera(theta=90.0, x=(-8, 8), y=(-8, 8),
                           resolution=(160, 5))
        img = grayt.shadow(bh, cam)
        row = img.status[:, 2]  # y ~ 0 scan line
        xs = np.linspace(-8, 8, 160, endpoint=False) + 8.0/160
        captured = xs[row == grayt.STATUS_CAPTURED]
        b_crit = np.sqrt(27.0)
        assert captured.min() == pytest.approx(-b_crit, abs=0.15)
        assert captured.max() == pytest.approx(b_crit, abs=0.15)


class TestDisk:
    def test_page_thorne_flux_shape(self):
        rs, fs = grayt.flux_profile(grayt.BlackHole(0.0), r_out=30.0)
        assert fs[0] == pytest.approx(0.0, abs=1e-12)
        assert np.all(fs[1:] > 0)
        assert rs[np.argmax(fs)] == pytest.approx(9.55, abs=0.1)

    def test_redshift_asymmetry(self):
        # Approaching side blueshifted (g > 1), receding side redshifted
        bh = grayt.BlackHole(0.0)
        cam = grayt.Camera()
        disk = grayt.ThinDisk(l0=2.8)
        img = grayt.render(bh, cam, disk,
                           rtol=1e-8, atol=1e-10)
        g = img.g[img.status == grayt.STATUS_DISK]
        assert g.max() > 1.0
        assert g.min() < 1.0

    def test_render_smoke(self):
        bh = grayt.BlackHole(0.95)
        cam = grayt.Camera(resolution=(96, 48))
        disk = grayt.ThinDisk(l0=1.8)
        img = grayt.render(bh, cam, disk)
        assert (img.status == grayt.STATUS_DISK).sum() > 0
        assert (img.status == grayt.STATUS_CAPTURED).sum() > 0
        assert img.intensity.max() > 0
        assert np.isfinite(img.intensity).all()
        # Doppler-bright side on the left (paper convention)
        nx = img.intensity.shape[0]
        assert img.intensity[:nx//2].max() > img.intensity[nx//2:].max()
