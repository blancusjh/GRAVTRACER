"""GPU render backend (OpenCL): Apple Silicon and NVIDIA/AMD/Intel.

Requires the optional dependency ``pyopencl`` (``pip install
gravtracer[gpu]`` or ``pip install pyopencl``). One kernel source runs
everywhere; precision is fp64 on devices that support it (NVIDIA) and
fp32 on Apple GPUs, which have no double-precision hardware.

Usage::

    import grayt
    img = grayt.render(bh, cam, disk, backend="gpu")   # via the facade
    grayt.gpu.devices()                                 # what's available

or directly: ``grayt.gpu.render(bh, cam, disk, precision="fp32")``.
"""
from .engine import available, devices, render, FP32_RTOL_FLOOR, FP32_ATOL_FLOOR

__all__ = ["available", "devices", "render",
           "FP32_RTOL_FLOOR", "FP32_ATOL_FLOOR"]
