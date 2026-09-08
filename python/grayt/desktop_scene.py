"""Native viewer adapter for the gallery's independent radiation models.

Kerr Page–Thorne geometry stays on OpenCL. Surface brightness is evaluated
from the hit radius and ZAMO camera constants with Keplerian motion and g^4.
Other imported geometries use the existing CPU scene renderer explicitly.
"""

import numpy as np

from ._runtime import metadata
from .emission import PageThorneDisk, metric_samples
from .matter import ThinDisk


class SceneRenderer:
    def __init__(
        self,
        spacetime,
        disk,
        resolution,
        *,
        surface=None,
        escape_radius=None,
        precision="auto",
        device=None,
        **options,
    ):
        self.spacetime, self.disk, self.surface = spacetime, disk, surface
        self.resolution = tuple(resolution)
        self.escape_radius = escape_radius
        self.options = options
        self.gpu = None
        if (
            isinstance(disk, PageThorneDisk)
            and metadata(spacetime) != disk.spacetime_metadata
        ):
            raise ValueError("PageThorneDisk was built for a different Kerr spacetime")
        if isinstance(disk, PageThorneDisk) and surface is None:
            from .gpu import Renderer

            self.gpu = Renderer(
                spacetime,
                ThinDisk(r_in=disk.r_in, r_out=disk.r_out),
                resolution,
                escape_radius=escape_radius,
                precision=precision,
                device=device,
                **options,
            )
        self.device_name = self.gpu.device_name if self.gpu else "CPU fp64"

    def render(self, camera):
        if self.gpu is None:
            from .scene import render_scene

            return render_scene(
                self.spacetime,
                camera,
                self.disk,
                surface=self.surface,
                escape_radius=self.escape_radius,
                **self.options,
            )
        image = self.gpu.render(camera)
        hit = image.status == 2
        r = image.r_hit[hit]
        if r.size:
            nx, ny = camera.resolution
            xs = camera.x[0] + (np.arange(nx) + 0.5) * (camera.x[1] - camera.x[0]) / nx
            x = np.broadcast_to(xs[:, None], (nx, ny))[hit]
            gd = self.spacetime.metric_cov(camera.r, np.deg2rad(camera.theta))
            at = np.sqrt(gd[4] / (gd[1] ** 2 - gd[0] * gd[4]))
            pt = 1 / at + x * gd[1] / (camera.r * np.sqrt(gd[4]))
            pp = x * np.sqrt(gd[4]) / camera.r
            omega = 1 / (r**1.5 + self.spacetime.a)
            local = metric_samples(self.spacetime, r, np.full_like(r, np.pi / 2))
            g = np.sqrt(
                -local[:, 0] - 2 * omega * local[:, 1] - omega**2 * local[:, 4]
            ) / (pt + omega * pp)
            image.g[hit] = g
            image.intensity[hit] = g**4 * self.disk.intensity(r, np.zeros_like(r), 0)
        image.meta["disk"] = self.disk.metadata()
        image.meta["radiation"] = "Page-Thorne / circular geodesics / bolometric g^4"
        return image
