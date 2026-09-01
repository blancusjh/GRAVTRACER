"""GPU backend (OpenCL) vs the Fortran CPU reference.

Skipped entirely when pyopencl or an OpenCL device is unavailable, so
the suite stays green on CI machines without a GPU stack. On fp32-only
devices (Apple Silicon) the comparisons use tolerances matched to
single precision; on fp64 devices they are much tighter automatically.
"""
import warnings

import numpy as np
import pytest

import grayt

gpu = pytest.importorskip("grayt.gpu")

pytestmark = pytest.mark.skipif(
    not gpu.available(), reason="no OpenCL device available")

CAM = grayt.Camera(r=1000.0, theta=80.0, x=(-12, 12), y=(-12, 12),
                   resolution=(64, 64))


def _fp64():
    return gpu.devices()[0]["fp64"]


def _render_gpu(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # fp32 tolerance clamp
        return grayt.render(*args, backend="gpu", **kwargs)


class TestAgainstCPU:
    def test_kerr_disk_matches_cpu(self):
        bh = grayt.BlackHole(a=0.95)
        disk = grayt.ThinDisk(l0=0.0, r_out=18.0)
        cpu = grayt.render(bh, CAM, disk)
        img = _render_gpu(bh, CAM, disk)

        # classification must agree except possibly at shadow/disk edges
        frac = (cpu.status != img.status).mean()
        assert frac < 0.01
        both = (cpu.status == 2) & (img.status == 2)
        assert both.any()
        tol = 1e-8 if _fp64() else 1e-2
        assert np.abs(cpu.r_hit[both] - img.r_hit[both]).max() < tol
        rel = np.abs(cpu.intensity[both] - img.intensity[both]).max()
        assert rel / cpu.intensity.max() < (1e-8 if _fp64() else 1e-2)

    def test_kerr_shadow_matches_cpu(self):
        bh = grayt.BlackHole(a=0.9)
        cpu = grayt.shadow(bh, CAM)
        img = _render_gpu(bh, CAM)
        assert (cpu.status != img.status).mean() < 0.01
        both = (cpu.status == 0) & (img.status == 0)
        tol = 1e-8 if _fp64() else 1e-2
        assert np.abs(cpu.theta_inf[both] - img.theta_inf[both]).max() < tol

    def test_qmetric_shadow_matches_cpu(self):
        qm = grayt.QMetric(q=0.5)
        cpu = grayt.shadow(qm, CAM)
        img = _render_gpu(qm, CAM)
        assert (cpu.status != img.status).mean() < 0.01

    def test_constraint_monitor(self):
        # |H| stays small along GPU rays too (fp32: roundoff-limited)
        bh = grayt.BlackHole(a=0.5)
        img = _render_gpu(bh, CAM, constraint_monitor=True)
        cap = 1e-8 if _fp64() else 1e-2
        escaped = img.status == 0
        assert escaped.any()
        assert np.median(img.herr[escaped]) < cap


class TestBackendAPI:
    def test_devices_listed(self):
        devs = gpu.devices()
        assert devs and all("name" in d and "fp64" in d for d in devs)

    def test_gpu_meta(self):
        img = _render_gpu(grayt.BlackHole(a=0.5), CAM)
        assert img.meta["backend"] == "gpu"
        assert img.meta["precision"] in ("fp32", "fp64")

    def test_rejects_non_dp45(self):
        with pytest.raises(ValueError, match="rkdp45"):
            grayt.render(grayt.BlackHole(a=0.5), CAM, backend="gpu",
                         method="rkck45")

    def test_rejects_disk_off_kerr(self):
        with pytest.raises(ValueError, match="Kerr"):
            grayt.render(grayt.QMetric(q=0.5), CAM, grayt.ThinDisk(),
                         backend="gpu")

    def test_fp64_request_on_fp32_device(self):
        if _fp64():
            pytest.skip("device has fp64")
        with pytest.raises(ValueError, match="fp64"):
            gpu.render(grayt.BlackHole(a=0.5), CAM, precision="fp64")

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="backend"):
            grayt.render(grayt.BlackHole(a=0.5), CAM, backend="tpu")

    def test_backend_kwargs_cpu_rejected(self):
        with pytest.raises(TypeError, match="GPU-only"):
            grayt.render(grayt.BlackHole(a=0.5), CAM, precision="fp32")
