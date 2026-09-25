"""grayt — GRAVTRACER: relativistic ray tracing around compact objects.

Replication and extension of OSIRIS (Velásquez-Cadavid et al.,
arXiv:2202.00086): Hamiltonian null and time-like geodesics integrated
in stationary axisymmetric spacetimes (Kerr, q-metric), with a
Page-Thorne thin accretion disk, image formation, and 2D/3D orbit
visualization.

Quick start::

    import grayt
    img = grayt.render(grayt.BlackHole(a=0.95),
                       grayt.Camera(resolution=(1024, 512)),
                       grayt.ThinDisk(l0=1.8))
    img.plot()

Ontology: ``Spacetime`` (BlackHole, QMetric) -> ``PhysicalSystem``
(spacetime + matter) -> ``System`` (physics + instruments + experiments).
"""
from .spacetime import Spacetime, BlackHole, QMetric
from .geometry import PlanarSurface, bl_to_cart, cart_to_bl, null_momentum
from .matter import ThinDisk, ImageSource
from .instruments import Camera, Screen
from .results import Image, Photograph, Ray, Trajectory
from .api import (render, shadow, trace, camera_ray, orbit_ic,
                  flux_profile, STATUS_ESCAPED, STATUS_CAPTURED,
                  STATUS_DISK, STATUS_FAILED)
from .system import PhysicalSystem, System
from .plotting import (plot_orbits_2d, plot_image, plot_shadow,
                       plot_lensing)
from . import gpu, style
from .viewer import view

__all__ = [
    "Spacetime", "BlackHole", "QMetric",
    "PlanarSurface", "bl_to_cart", "cart_to_bl", "null_momentum",
    "ThinDisk", "ImageSource", "Camera", "Screen",
    "Image", "Photograph", "Ray", "Trajectory",
    "render", "shadow", "trace", "camera_ray", "orbit_ic", "flux_profile",
    "PhysicalSystem", "System",
    "plot_orbits_2d", "plot_image", "plot_shadow", "plot_lensing",
    "gpu", "style", "view",
    "STATUS_ESCAPED", "STATUS_CAPTURED", "STATUS_DISK", "STATUS_FAILED",
]

__version__ = "0.3.0"

from .metrics import CustomMetric, TabulatedMetric, ReissnerNordstrom, SphericalStar
from .emission import TabulatedDisk, EmittingDisk, EmittingSurface, PageThorneDisk, SlabDisk, circular_velocity, rotating_velocity
from .sky import CelestialSky
from .scene import RayBundle, SceneImage, trace_bundle, trace_crossings, render_scene

__all__ += ["TabulatedDisk", "CustomMetric", "TabulatedMetric", "ReissnerNordstrom", "SphericalStar",
            "EmittingDisk", "EmittingSurface", "PageThorneDisk", "SlabDisk", "circular_velocity",
            "rotating_velocity", "CelestialSky", "RayBundle", "SceneImage",
            "trace_bundle", "trace_crossings", "render_scene"]

from .animation import orbit_cameras, render_movie
__all__ += ["orbit_cameras", "render_movie"]

from .volume import EmittingVolume, VolumeGrid, VolumeImage, render_volume
__all__ += ["EmittingVolume", "VolumeGrid", "VolumeImage", "render_volume"]
