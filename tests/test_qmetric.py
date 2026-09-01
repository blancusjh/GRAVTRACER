"""Validation of the q-metric spacetime (Appendix A of the paper)."""
import numpy as np
import pytest

import grayt
from grayt import _core

ST = _core.spacetime


class TestMetric:
    @pytest.mark.parametrize("r,th", [(5.0, 1.2), (12.0, 0.4), (3.1, 2.6)])
    def test_q0_is_schwarzschild(self, r, th):
        par0 = [0.0, 0, 0, 0]
        assert np.allclose(ST.metric_cov(2, par0, r, th),
                           ST.metric_cov(1, par0, r, th), rtol=1e-13)
        gk = ST.metric_contra(1, par0, r, th)
        gq = ST.metric_contra(2, par0, r, th)
        for a, b in zip(gk, gq):
            assert np.allclose(a, b, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("q", [-0.4, 0.5, 1.0])
    def test_contra_is_inverse(self, q):
        par = [q, 0, 0, 0]
        r, th = 6.7, 1.05
        gd = ST.metric_cov(2, par, r, th)
        gu, _, _ = ST.metric_contra(2, par, r, th)
        for i in (0, 2, 3, 4):
            assert 1.0/gd[i] == pytest.approx(gu[i], rel=1e-12)
        assert gd[1] == 0.0 and gu[1] == 0.0

    def test_derivatives_match_finite_differences(self):
        par = [0.8, 0, 0, 0]
        r, th = 4.3, 0.9
        _, dgr, dgt = ST.metric_contra(2, par, r, th)
        eps = 1e-6
        fd_r = (ST.metric_contra(2, par, r + eps, th)[0] -
                ST.metric_contra(2, par, r - eps, th)[0])/(2*eps)
        fd_t = (ST.metric_contra(2, par, r, th + eps)[0] -
                ST.metric_contra(2, par, r, th - eps)[0])/(2*eps)
        assert np.allclose(fd_r, dgr, rtol=1e-6, atol=1e-8)
        assert np.allclose(fd_t, dgt, rtol=1e-6, atol=1e-8)


class TestGeodesics:
    def test_q0_trajectory_matches_schwarzschild(self):
        y0, pt, pphi = grayt.orbit_ic(grayt.BlackHole(0.0), 25.0, 0.973, 4.2)
        kw = dict(p_t=pt, p_phi=pphi, lambda_max=500.0, rtol=1e-11,
                  atol=1e-13)
        t_k = grayt.trace(grayt.BlackHole(0.0), y0, **kw)
        t_q = grayt.trace(grayt.QMetric(0.0), y0, **kw)
        n = min(len(t_k.r), len(t_q.r))
        assert n > 50
        # Same physics, different algebraic form of the metric: agreement
        # is limited by accumulated round-off, not by the integrator.
        assert np.allclose(t_k.r[:n], t_q.r[:n], rtol=1e-6)
        assert np.allclose(t_k.phi[:n], t_q.phi[:n], rtol=1e-6, atol=1e-6)

    def test_null_ics_and_constraint(self):
        qm = grayt.QMetric(q=1.0)
        cam = grayt.Camera(theta=90.0, x=(-12, 12), y=(-12, 12))
        y0, pt, pphi = grayt.camera_ray(qm, cam, 9.0, 2.0)
        gu, _, _ = qm.metric_contra(y0[1], y0[2])
        h = 0.5*(gu[0]*pt**2 + 2*gu[1]*pt*pphi + gu[2]*y0[4]**2 +
                 gu[3]*y0[5]**2 + gu[4]*pphi**2)
        assert abs(h) < 1e-12
        st, _, hmax, _ = _core.raytracer.trace_ray(
            qm.mid, qm.par, pt, pphi, y0, 1, 1e-10, 1e-12, 1100.0, 0,
            6.0, 20.0, 500000)
        assert st in (grayt.STATUS_ESCAPED, grayt.STATUS_CAPTURED)
        # constraint error grows near the singular surface on capture
        assert hmax < 1e-4

    def test_shadow_scales_with_adm_mass(self):
        # ZV ADM mass is m(1+q); the q = 1 shadow diameter should be
        # roughly twice the q = 0 (Schwarzschild) one.
        cam = grayt.Camera(theta=90.0, x=(-16, 16), y=(-16, 16),
                           resolution=(120, 120))
        n0 = (grayt.shadow(grayt.QMetric(0.0), cam).status == 1).sum()
        n1 = (grayt.shadow(grayt.QMetric(1.0), cam).status == 1).sum()
        ratio = np.sqrt(n1/n0)
        assert 1.6 < ratio < 2.4


class TestValidation:
    def test_disk_requires_kerr(self):
        with pytest.raises(ValueError, match="Kerr"):
            grayt.render(grayt.QMetric(q=0.5), grayt.Camera(),
                         grayt.ThinDisk())

    def test_bad_quadrupole_rejected(self):
        with pytest.raises(ValueError):
            grayt.QMetric(q=-1.5)

    def test_bad_method_message(self):
        with pytest.raises(ValueError, match="rkdp45"):
            grayt.shadow(grayt.BlackHole(0.5),
                         grayt.Camera(resolution=(8, 8)), method="rk4")

    def test_radians_theta_rejected(self):
        with pytest.raises(ValueError, match="degrees"):
            grayt.Camera(theta=np.pi/2)

    def test_zero_resolution_rejected(self):
        with pytest.raises(ValueError):
            grayt.Camera(resolution=(0, 0))
