"""grayt — backward ray tracing around Kerr black holes.

Replication of OSIRIS (Velásquez-Cadavid et al., arXiv:2202.00086):
Hamiltonian null geodesics integrated from the observer's image plane,
with a Page-Thorne thin accretion disk.

Quick start::

    import grayt
    img = grayt.render(grayt.BlackHole(a=0.95),
                       grayt.Camera(resolution=(1024, 512)),
                       grayt.ThinDisk(l0=1.8))
    img.plot()
"""
from .api import (BlackHole, Camera, ThinDisk, Image, render, shadow,
                  trace, camera_ray, flux_profile,
                  STATUS_ESCAPED, STATUS_CAPTURED, STATUS_DISK,
                  STATUS_FAILED)
from .scene import Scene

__all__ = ["BlackHole", "Camera", "ThinDisk", "Image", "Scene", "render",
           "shadow", "trace", "camera_ray", "flux_profile",
           "STATUS_ESCAPED", "STATUS_CAPTURED", "STATUS_DISK",
           "STATUS_FAILED"]

__version__ = "0.1.0"
