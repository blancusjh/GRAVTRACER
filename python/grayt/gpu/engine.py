"""OpenCL host layer for the GPU render backend.

One kernel source (kernel.cl) runs on every OpenCL device: Apple
Silicon GPUs through Apple's OpenCL-on-Metal (fp32 only -- Apple GPUs
have no double-precision hardware) and NVIDIA/AMD/Intel GPUs (fp64
when the device exposes ``cl_khr_fp64``). The program is compiled per
(device, metric, precision) and cached for the process lifetime.

The GPU backend supports the ``render``/``shadow`` path (the hot loop)
with the Dormand-Prince integrator only; single-geodesic tracing and
the System/plane instruments stay on the Fortran CPU core.
"""
from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np

try:
    import pyopencl as cl
    _CL_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    cl = None
    _CL_ERROR = str(exc)

from .. import _core
from ..spacetime import Spacetime, MID_KERR
from ..instruments import Camera
from ..matter import ThinDisk
from ..results import Image

# fp32 cannot resolve tolerances much below its epsilon (~1.2e-7); the
# step controller would stall rejecting steps whose error estimate is
# pure roundoff. Requests tighter than these floors are clamped.
FP32_RTOL_FLOOR = 1e-5
FP32_ATOL_FLOOR = 1e-7

_KERNEL_SOURCE = None
_ENGINES: dict[int, "_Engine"] = {}


def available() -> bool:
    """True if pyopencl is importable and at least one device exists."""
    if cl is None:
        return False
    try:
        return any(p.get_devices() for p in cl.get_platforms())
    except cl.Error:
        return False


def devices() -> list[dict]:
    """All OpenCL devices, flattened over platforms. The list index is
    the ``device`` argument of :func:`render`."""
    if cl is None:
        raise RuntimeError(f"pyopencl is not installed: {_CL_ERROR}")
    out = []
    for plat in cl.get_platforms():
        for dev in plat.get_devices():
            out.append({
                "platform": plat.name.strip(),
                "name": dev.name.strip(),
                "type": cl.device_type.to_string(dev.type),
                "fp64": dev.double_fp_config != 0,
                "compute_units": dev.max_compute_units,
            })
    return out


def _flat_devices():
    devs = []
    for plat in cl.get_platforms():
        devs.extend(plat.get_devices())
    return devs


def _pick_device(index: int | None):
    devs = _flat_devices()
    if not devs:
        raise RuntimeError("no OpenCL devices found")
    if index is not None:
        try:
            return devs[index], index
        except IndexError:
            raise ValueError(
                f"device index {index} out of range; see grayt.gpu.devices()"
            ) from None
    # default: first GPU device, else first device of any type
    for i, d in enumerate(devs):
        if d.type & cl.device_type.GPU:
            return d, i
    return devs[0], 0


class _Engine:
    """Context/queue/program cache for one device."""

    def __init__(self, device):
        self.device = device
        self.fp64 = device.double_fp_config != 0
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)
        self._programs: dict[tuple[int, bool], cl.Program] = {}

    def program(self, mid: int, fp64: bool) -> "cl.Program":
        key = (mid, fp64)
        if key not in self._programs:
            global _KERNEL_SOURCE
            if _KERNEL_SOURCE is None:
                _KERNEL_SOURCE = (Path(__file__).parent / "kernel.cl").read_text()
            options = [f"-DGRAYT_MID={mid}"]
            if fp64:
                options.append("-DGRAYT_FP64")
            else:
                # keep the kernel's unsuffixed literals single precision
                # on devices without fp64 (Apple GPUs)
                options.append("-cl-single-precision-constant")
            self._programs[key] = cl.Program(
                self.context, _KERNEL_SOURCE).build(options)
        return self._programs[key]


def _engine(index: int | None) -> tuple["_Engine", int]:
    device, i = _pick_device(index)
    if i not in _ENGINES:
        _ENGINES[i] = _Engine(device)
    return _ENGINES[i], i


def render(spacetime: Spacetime, camera: Camera, disk: ThinDisk | None = None,
           method: str = "rkdp45", rtol: float = 1e-8, atol: float = 1e-10,
           max_steps: int = 500_000, constraint_monitor: bool = False,
           precision: str = "auto", device: int | None = None) -> Image:
    """GPU counterpart of :func:`grayt.render` (one work-item per pixel).

    ``precision``: "auto" (fp64 if the device supports it, else fp32),
    "fp64", or "fp32" (useful on NVIDIA, where fp32 is much faster).
    On fp32 the tolerances are clamped to ``FP32_RTOL_FLOOR``/
    ``FP32_ATOL_FLOOR`` with a warning; expect image-quality rather
    than reference-quality accuracy. ``device`` indexes
    :func:`grayt.gpu.devices()`.
    """
    if cl is None:
        raise RuntimeError(
            f"the GPU backend needs pyopencl (pip install pyopencl): {_CL_ERROR}")
    if method != "rkdp45":
        raise ValueError("the GPU backend implements rkdp45 only; use "
                         "backend='cpu' for other integrators")
    if disk is not None and spacetime.mid != MID_KERR:
        raise ValueError("the thin-disk model (Page-Thorne, ISCO, redshift) "
                         "is defined for Kerr only; render this spacetime "
                         "without a disk")

    eng, dev_index = _engine(device)

    if precision == "auto":
        fp64 = eng.fp64
    elif precision == "fp64":
        if not eng.fp64:
            raise ValueError(
                f"device {eng.device.name.strip()!r} has no fp64 support "
                "(Apple GPUs are fp32-only); use precision='fp32' or 'auto'")
        fp64 = True
    elif precision == "fp32":
        fp64 = False
    else:
        raise ValueError("precision must be 'auto', 'fp64' or 'fp32'")

    if not fp64 and (rtol < FP32_RTOL_FLOOR or atol < FP32_ATOL_FLOOR):
        warnings.warn(
            f"fp32 GPU device: clamping tolerances to rtol>={FP32_RTOL_FLOOR}, "
            f"atol>={FP32_ATOL_FLOOR} (requested rtol={rtol}, atol={atol})",
            stacklevel=2)
        rtol = max(rtol, FP32_RTOL_FLOOR)
        atol = max(atol, FP32_ATOL_FLOOR)

    real = np.float64 if fp64 else np.float32
    nx, ny = camera.resolution
    npix = nx*ny

    disk_on = 1 if disk is not None else 0
    rout = disk.r_out if disk is not None else 20.0
    l0 = disk.l0 if disk is not None else 0.0
    rin = -1.0 if (disk is None or disk.r_in is None) else disk.r_in
    if disk_on and rin <= 0.0:
        rin = float(_core.raytracer.get_isco(spacetime.par[0]))

    # Page-Thorne flux table, built in double on the CPU core exactly as
    # the Fortran RENDER_IMAGE does (INIT_FLUX_TABLE(a, rout, 4000)).
    if disk_on:
        tab_r, tab_f = _core.disk_model.flux_profile(
            spacetime.par[0], rout, 4000)
        tab_r1, tab_r2 = float(tab_r[0]), float(tab_r[-1])
        tab_dr = float(tab_r[1] - tab_r[0])
        tab_f = np.ascontiguousarray(tab_f, dtype=real)
    else:
        tab_f = np.zeros(2, dtype=real)
        tab_r1 = tab_r2 = 0.0
        tab_dr = 1.0

    rcap = float(spacetime.capture_radius) + 1e-2
    r_esc = 1.1*camera.r

    mf = cl.mem_flags
    ctx, queue = eng.context, eng.queue
    d_tabf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tab_f)
    d_real = {name: cl.Buffer(ctx, mf.WRITE_ONLY, npix*np.dtype(real).itemsize)
              for name in ("intens", "gmap", "rhit", "herrm", "thf", "phf")}
    d_status = cl.Buffer(ctx, mf.WRITE_ONLY, npix*4)

    kern = eng.program(spacetime.mid, fp64).render_slab
    R, I = real, np.int32

    # Row slabs keep each enqueue short (Apple/Windows GPU watchdogs
    # kill kernels that hog the device for seconds).
    rows_per_slab = max(1, min(ny, 262_144 // max(nx, 1)))
    # Fortran camera x-axis is mirrored w.r.t. the paper's figures
    # (same convention as api.render): pass x in reversed, negated order.
    for row0 in range(0, ny, rows_per_slab):
        nrows = min(rows_per_slab, ny - row0)
        kern(queue, ((nx*nrows + 63)//64*64,), None,
             R(spacetime.par[0]), R(camera.r), R(np.deg2rad(camera.theta)),
             R(np.deg2rad(camera.phi)),
             R(-camera.x[1]), R(-camera.x[0]), R(camera.y[0]), R(camera.y[1]),
             I(nx), I(ny), I(row0), I(nrows),
             R(rtol), R(atol),
             I(disk_on), R(rin), R(rout), R(l0),
             I(max_steps), I(1 if constraint_monitor else 0),
             R(rcap), R(r_esc),
             d_tabf, I(len(tab_f)), R(tab_r1), R(tab_r2), R(tab_dr),
             d_real["intens"], d_real["gmap"], d_real["rhit"],
             d_status, d_real["herrm"], d_real["thf"], d_real["phf"])

    host = {}
    for name, buf in d_real.items():
        arr = np.empty((nx, ny), dtype=real)
        cl.enqueue_copy(queue, arr, buf)
        host[name] = arr.astype(np.float64)
    status = np.empty((nx, ny), dtype=np.int32)
    cl.enqueue_copy(queue, status, d_status)
    queue.finish()

    # same orientation post-processing as api.render
    intens, gmap, rhit, herrm, thf, phf, status = (
        np.flip(m, axis=0) for m in (host["intens"], host["gmap"],
                                     host["rhit"], host["herrm"],
                                     host["thf"], host["phf"], status))

    meta = {"spacetime": dataclasses.asdict(spacetime),
            "camera": dataclasses.asdict(camera),
            "disk": dataclasses.asdict(disk) if disk else None,
            "method": method, "rtol": rtol, "atol": atol,
            "backend": "gpu",
            "device": eng.device.name.strip(),
            "device_index": dev_index,
            "precision": "fp64" if fp64 else "fp32"}
    return Image(intensity=intens, g=gmap, r_hit=rhit,
                 status=status.astype(np.int32), herr=herrm,
                 theta_inf=thf, phi_inf=phf, extent=camera.extent, meta=meta)
