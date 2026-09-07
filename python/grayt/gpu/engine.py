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


class Renderer:
    """Persistent OpenCL renderer for repeated views of one scene.

    The OpenCL context, compiled kernel, Page--Thorne flux table, and all
    device/host output buffers are allocated once.  Only camera values and
    the kernel execution change between :meth:`render` calls.  This is the
    intended interface for interactive applications; :func:`render` remains
    the convenient one-shot wrapper.

    Parameters other than ``camera`` are fixed for the renderer's lifetime.
    A camera passed to :meth:`render` must use the configured ``resolution``.
    Create one renderer per resolution for progressive rendering.
    """

    _REAL_OUTPUTS = ("intens", "gmap", "rhit", "herrm", "thf", "phf")

    def __init__(self, spacetime: Spacetime, disk: ThinDisk | None = None,
                 resolution: tuple[int, int] = (512, 256),
                 method: str = "rkdp45", rtol: float = 1e-8,
                 atol: float = 1e-10, max_steps: int = 500_000,
                 constraint_monitor: bool = False,
                 precision: str = "auto", device: int | None = None):
        if cl is None:
            raise RuntimeError(
                "the GPU backend needs pyopencl "
                f"(pip install pyopencl): {_CL_ERROR}")
        if method != "rkdp45":
            raise ValueError("the GPU backend implements rkdp45 only; use "
                             "backend='cpu' for other integrators")
        if disk is not None and spacetime.mid != MID_KERR:
            raise ValueError(
                "the thin-disk model (Page-Thorne, ISCO, redshift) is "
                "defined for Kerr only; render this spacetime without a disk")
        nx, ny = resolution
        if not (isinstance(nx, (int, np.integer)) and
                isinstance(ny, (int, np.integer)) and nx > 0 and ny > 0):
            raise ValueError(
                f"resolution must be positive integers, got {resolution}")

        self.spacetime = spacetime
        self.disk = disk
        self.resolution = (int(nx), int(ny))
        self.method = method
        self.max_steps = max_steps
        self.constraint_monitor = constraint_monitor
        self._eng, self.device_index = _engine(device)
        self._par = float(spacetime.par[0])
        self._rcap = float(spacetime.capture_radius) + 1e-2

        if precision == "auto":
            self.fp64 = self._eng.fp64
        elif precision == "fp64":
            if not self._eng.fp64:
                raise ValueError(
                    f"device {self._eng.device.name.strip()!r} has no fp64 "
                    "support (Apple GPUs are fp32-only); use "
                    "precision='fp32' or 'auto'")
            self.fp64 = True
        elif precision == "fp32":
            self.fp64 = False
        else:
            raise ValueError("precision must be 'auto', 'fp64' or 'fp32'")

        if not self.fp64 and (rtol < FP32_RTOL_FLOOR or
                              atol < FP32_ATOL_FLOOR):
            warnings.warn(
                f"fp32 GPU device: clamping tolerances to "
                f"rtol>={FP32_RTOL_FLOOR}, atol>={FP32_ATOL_FLOOR} "
                f"(requested rtol={rtol}, atol={atol})", stacklevel=2)
            rtol = max(rtol, FP32_RTOL_FLOOR)
            atol = max(atol, FP32_ATOL_FLOOR)
        self.rtol = rtol
        self.atol = atol
        self.real = np.float64 if self.fp64 else np.float32

        self._disk_on = 1 if disk is not None else 0
        self._rout = disk.r_out if disk is not None else 20.0
        self._l0 = disk.l0 if disk is not None else 0.0
        self._rin = -1.0 if (disk is None or disk.r_in is None) else disk.r_in
        if self._disk_on and self._rin <= 0.0:
            self._rin = float(_core.raytracer.get_isco(self._par))

        # Build and upload this immutable scene data only once.
        if self._disk_on:
            tab_r, tab_f = _core.disk_model.flux_profile(
                self._par, self._rout, 4000)
            self._tab_r1, self._tab_r2 = float(tab_r[0]), float(tab_r[-1])
            self._tab_dr = float(tab_r[1] - tab_r[0])
            tab_f = np.ascontiguousarray(tab_f, dtype=self.real)
        else:
            tab_f = np.zeros(2, dtype=self.real)
            self._tab_r1 = self._tab_r2 = 0.0
            self._tab_dr = 1.0
        self._tab_n = len(tab_f)

        mf = cl.mem_flags
        ctx = self._eng.context
        itemsize = np.dtype(self.real).itemsize
        npix = nx*ny
        self._d_tabf = cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tab_f)
        self._d_real = {
            name: cl.Buffer(ctx, mf.WRITE_ONLY, npix*itemsize)
            for name in self._REAL_OUTPUTS
        }
        self._d_status = cl.Buffer(ctx, mf.WRITE_ONLY, npix*4)
        self._host_real = {
            name: np.empty((nx, ny), dtype=self.real)
            for name in self._REAL_OUTPUTS
        }
        self._host_status = np.empty((nx, ny), dtype=np.int32)
        self._kernel = cl.Kernel(
            self._eng.program(spacetime.mid, self.fp64), "render_slab")
        self._rows_per_slab = max(1, min(ny, 262_144 // max(nx, 1)))

    @property
    def device_name(self) -> str:
        return self._eng.device.name.strip()

    @property
    def precision(self) -> str:
        return "fp64" if self.fp64 else "fp32"

    def render(self, camera: Camera) -> Image:
        """Render ``camera`` while reusing all persistent scene resources."""
        if tuple(camera.resolution) != self.resolution:
            raise ValueError(
                f"camera resolution {camera.resolution} does not match "
                f"renderer resolution {self.resolution}")

        nx, ny = self.resolution
        queue = self._eng.queue
        R, I = self.real, np.int32
        r_esc = 1.1*camera.r

        # Row slabs keep each enqueue short (Apple/Windows GPU watchdogs
        # kill kernels that hog the device for seconds).  The camera x-axis
        # is reversed/negated to preserve GRAVTRACER's paper convention.
        for row0 in range(0, ny, self._rows_per_slab):
            nrows = min(self._rows_per_slab, ny - row0)
            self._kernel(
                queue, ((nx*nrows + 63)//64*64,), None,
                R(self._par), R(camera.r),
                R(np.deg2rad(camera.theta)), R(np.deg2rad(camera.phi)),
                R(-camera.x[1]), R(-camera.x[0]),
                R(camera.y[0]), R(camera.y[1]),
                I(nx), I(ny), I(row0), I(nrows),
                R(self.rtol), R(self.atol),
                I(self._disk_on), R(self._rin), R(self._rout), R(self._l0),
                I(self.max_steps), I(1 if self.constraint_monitor else 0),
                R(self._rcap), R(r_esc),
                self._d_tabf, I(self._tab_n), R(self._tab_r1),
                R(self._tab_r2), R(self._tab_dr),
                self._d_real["intens"], self._d_real["gmap"],
                self._d_real["rhit"], self._d_status,
                self._d_real["herrm"], self._d_real["thf"],
                self._d_real["phf"])

        for name, buf in self._d_real.items():
            cl.enqueue_copy(queue, self._host_real[name], buf)
        cl.enqueue_copy(queue, self._host_status, self._d_status)
        queue.finish()

        # Copy before returning: a later frame may reuse the host staging
        # arrays, while every returned Image remains an independent result.
        host = {name: arr.astype(np.float64, copy=True)
                for name, arr in self._host_real.items()}
        status = self._host_status.copy()
        intens, gmap, rhit, herrm, thf, phf, status = (
            np.flip(m, axis=0) for m in
            (host["intens"], host["gmap"], host["rhit"], host["herrm"],
             host["thf"], host["phf"], status))

        meta = {
            "spacetime": dataclasses.asdict(self.spacetime),
            "camera": dataclasses.asdict(camera),
            "disk": dataclasses.asdict(self.disk) if self.disk else None,
            "method": self.method, "rtol": self.rtol, "atol": self.atol,
            "backend": "gpu", "device": self.device_name,
            "device_index": self.device_index, "precision": self.precision,
            "persistent_renderer": True,
        }
        return Image(intensity=intens, g=gmap, r_hit=rhit,
                     status=status.astype(np.int32), herr=herrm,
                     theta_inf=thf, phi_inf=phf, extent=camera.extent,
                     meta=meta)


def render(spacetime: Spacetime, camera: Camera, disk: ThinDisk | None = None,
           method: str = "rkdp45", rtol: float = 1e-8, atol: float = 1e-10,
           max_steps: int = 500_000, constraint_monitor: bool = False,
           precision: str = "auto", device: int | None = None) -> Image:
    """One-shot GPU counterpart of :func:`grayt.render`.

    For repeated camera views, construct :class:`Renderer` once to retain
    the flux table and OpenCL buffers between frames.
    """
    renderer = Renderer(
        spacetime, disk, camera.resolution, method=method, rtol=rtol,
        atol=atol, max_steps=max_steps,
        constraint_monitor=constraint_monitor, precision=precision,
        device=device)
    return renderer.render(camera)
