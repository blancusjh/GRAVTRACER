"""The laboratory layer.

- ``PhysicalSystem`` — the physics: a spacetime plus the matter and
  optical objects living in it (disk, image sources). Pure data.
- ``System`` — the laboratory: a PhysicalSystem plus instruments
  (cameras, screens), integrator settings, the record of traced rays,
  and the experiments themselves (render, photograph, form_image,
  trace_ray, visualize3d). Serializable from YAML via
  ``System.from_yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _core
from ._runtime import metric_context, metadata
from .api import (render as _render, _method_id, STATUS_ESCAPED,
                  STATUS_CAPTURED)
from .geometry import bl_to_cart, null_momentum
from .instruments import Camera, Screen
from .matter import ThinDisk, ImageSource
from .emission import EmittingDisk
from .results import Photograph, Ray
from .spacetime import Spacetime, BlackHole, QMetric

STATUS_SCREEN = 2  # TRACE_TO_PLANE: ray reached the target plane


@dataclass
class PhysicalSystem:
    """The physics: spacetime + objects living in it."""

    spacetime: Spacetime = field(default_factory=BlackHole)
    disk: ThinDisk | EmittingDisk | None = None
    sources: list = field(default_factory=list)
    sky: object = None
    surface: object = None

    # Backward-compatible alias (earlier versions took black_hole=...)
    @property
    def black_hole(self):
        return self.spacetime


@dataclass
class System:
    """The laboratory: physics + instruments + experiment records."""

    physical: PhysicalSystem = field(default_factory=PhysicalSystem)
    cameras: list = field(default_factory=list)
    screens: list = field(default_factory=list)
    rays: list = field(default_factory=list)
    method: str = "rkdp45"
    rtol: float = 1e-9
    atol: float = 1e-11
    exposure: float = 1.
    escape_radius: float | None = None

    # ------------------------------------------------------ construction

    @classmethod
    def from_yaml(cls, path) -> "System":
        """Build a System from a declarative YAML scene.

        Schema (all sections optional except one of black_hole/spacetime):
          black_hole: {a: 0.5}                      # Kerr shorthand
          spacetime:  {type: kerr|qmetric, a|q: ...}
          camera:     {r, theta, phi, x: [..], y: [..], resolution: [..]}
          disk:       {model, r_in (number or 'isco'), r_out, l0}
          integrator: {method, rtol, atol}

        Additional types, sky maps and bolometric disks are documented in
        docs/custom_models.md and configs/celestial_kerr.yml. Relative data
        paths are resolved against the YAML file's directory.
        """
        import yaml
        from pathlib import Path
        from .metrics import ReissnerNordstrom, SphericalStar, TabulatedMetric
        from .emission import PageThorneDisk, TabulatedDisk
        from .sky import CelestialSky
        base = Path(path).resolve().parent
        with open(path) as f:
            cfg = yaml.safe_load(f)

        if "spacetime" in cfg:
            sc = dict(cfg["spacetime"])
            kind = sc.pop("type", "kerr")
            constructors = {"kerr": BlackHole, "qmetric": QMetric,
                            "reissner-nordstrom": ReissnerNordstrom,
                            "spherical-star": SphericalStar}
            if kind == "tabulated":
                st = TabulatedMetric.load(base / sc.pop("path"))
                if sc:
                    raise ValueError(f"unknown tabulated-metric options: {list(sc)}")
            elif kind in constructors:
                st = constructors[kind](**sc)
            else:
                raise ValueError(f"unknown spacetime type: {kind}")
        else:
            st = BlackHole(**cfg.get("black_hole", {}))

        cam_cfg = dict(cfg.get("camera", {}))
        for key in ("x", "y", "resolution"):
            if key in cam_cfg:
                cam_cfg[key] = tuple(cam_cfg[key])
        cam = Camera(**cam_cfg)

        disk = None
        if cfg.get("disk") is not None:
            disk_cfg = dict(cfg["disk"])
            if disk_cfg.get("r_in") == "isco":
                disk_cfg["r_in"] = None
            model = disk_cfg.get("model", "page-thorne")
            if model == "page-thorne-bolometric":
                disk_cfg.pop("model")
                disk = PageThorneDisk(st, **disk_cfg)
            elif model == "tabulated":
                disk_cfg.pop("model")
                disk = TabulatedDisk.load(base / disk_cfg.pop("path"))
                if disk_cfg:
                    raise ValueError(f"unknown tabulated-disk options: {list(disk_cfg)}")
            else:
                disk = ThinDisk(**disk_cfg)

        sky = None
        if cfg.get("sky") is not None:
            sky_cfg = dict(cfg["sky"])
            kind = sky_cfg.pop("type", "procedural")
            if kind == "procedural":
                sky = CelestialSky.procedural(**sky_cfg)
            elif kind == "image":
                sky = CelestialSky(base / sky_cfg.pop("path"), **sky_cfg)
            else:
                raise ValueError(f"unknown sky type: {kind}")

        integ = cfg.get("integrator", {})
        return cls(physical=PhysicalSystem(spacetime=st, disk=disk, sky=sky),
                   cameras=[cam],
                   method=integ.get("method", "rkdp45"),
                   rtol=float(integ.get("rtol", 1e-8)),
                   atol=float(integ.get("atol", 1e-10)),
                   exposure=float(cfg.get("exposure", 1.)),
                   escape_radius=cfg.get("escape_radius"))

    # ------------------------------------------------------- experiments

    def render(self, camera: Camera | int = 0, **kwargs):
        """Backward-ray-trace an intensity image with the given camera
        (index into ``self.cameras`` or a Camera object)."""
        cam = self.cameras[camera] if isinstance(camera, int) else camera
        kwargs.setdefault("method", self.method)
        kwargs.setdefault("rtol", self.rtol)
        kwargs.setdefault("atol", self.atol)
        kwargs.setdefault("sky", self.physical.sky)
        kwargs.setdefault("surface", self.physical.surface)
        kwargs.setdefault("exposure", self.exposure)
        kwargs.setdefault("escape_radius", self.escape_radius)
        return _render(self.physical.spacetime, cam, self.physical.disk,
                       **kwargs)

    @metric_context
    def trace_ray(self, origin_cart, direction_cart, lambda_max=400.0,
                  keep=True):
        """Trace a single geodesic from a point along a direction and
        record it in ``self.rays`` (for the 3D viewer)."""
        st = self.physical.spacetime
        y0, pt, pphi = null_momentum(st, origin_cart, direction_cart)
        traj, herr, nout = _core.raytracer.trace_geodesic(
            st.mid, st.par, y0, pt, pphi, _method_id(self.method),
            self.rtol, self.atol, 1.0, lambda_max, 50_000)
        t = traj[:nout]
        pts = bl_to_cart(t[:, 2], t[:, 3], t[:, 4])
        pts = np.vstack([np.atleast_2d(np.asarray(origin_cart, float)), pts])
        captured = (t[-1, 2] <= st.capture_radius + 0.05) if nout else False
        ray = Ray(points=pts,
                  status=STATUS_CAPTURED if captured else STATUS_ESCAPED)
        if keep:
            self.rays.append(ray)
        return ray

    @metric_context
    def form_image(self, source: ImageSource | int = 0,
                   screen: Screen | int = 0, max_rays=40_000, r_max=None,
                   max_steps=100_000, keep_sample_rays=0):
        """Forward (collimated) image formation: one ray per sampled
        source pixel, emitted along the source normal, traced until it
        hits the screen plane, is captured, or escapes.

        Returns a dict with per-ray status counts.
        """
        src = (self.physical.sources[source]
               if isinstance(source, int) else source)
        scr = self.screens[screen] if isinstance(screen, int) else screen
        if src.emission != "collimated":
            raise ValueError(
                "forward projection needs emission='collimated'; for a "
                "lambertian source use System.photograph instead")
        st = self.physical.spacetime

        pos, colors, _ = src.sample_rays(max_rays=max_rays)
        n_ray = len(pos)
        y0s = np.empty((n_ray, 6))
        pts_ = np.empty(n_ray)
        pph_ = np.empty(n_ray)
        for i in range(n_ray):
            y0s[i], pts_[i], pph_[i] = null_momentum(st, pos[i], src._n)

        nhat, d = scr.plane
        if r_max is None:
            r_max = 4.0*max(np.linalg.norm(src._c),
                            np.linalg.norm(scr._c), 50.0)
        status, youts = _core.raytracer.trace_bundle_to_plane(
            st.mid, st.par, pts_, pph_, y0s, _method_id(self.method),
            self.rtol, self.atol, 1.0, nhat, d, r_max, max_steps)

        hit = status == STATUS_SCREEN
        hit_pts = bl_to_cart(youts[hit, 1], youts[hit, 2], youts[hit, 3])
        n_binned = scr.add_hits(hit_pts, colors[hit])

        if keep_sample_rays:
            idx = np.linspace(0, n_ray - 1, keep_sample_rays).astype(int)
            for i in idx:
                self.trace_ray(pos[i], src._n, lambda_max=3.0*r_max)

        counts = {int(k): int(v)
                  for k, v in zip(*np.unique(status, return_counts=True))}
        return {"n_rays": n_ray, "status_counts": counts,
                "n_on_screen": n_binned}

    @metric_context
    def photograph(self, camera: Camera, source: ImageSource | int = 0,
                   background=0.0, max_steps=200_000) -> Photograph:
        """Photograph a lambertian ``ImageSource`` through the spacetime.

        Backward ray tracing: one ray per camera pixel, integrated toward
        the source plane; where it lands on the card, the texture color
        is sampled (direction-independent emission). Captured or escaped
        rays show ``background``. Purely geometric — no redshift factor
        is applied to the sampled colors.

        Approximation: the ray terminates at its first crossing of the
        *infinite* source plane, even off the card.
        """
        ps = self.physical
        src = ps.sources[source] if isinstance(source, int) else source
        if src.emission != "lambertian":
            raise ValueError("photograph needs emission='lambertian'")

        st = ps.spacetime
        nx, ny = camera.resolution
        r0 = camera.r
        th0 = np.deg2rad(camera.theta)
        ph0 = np.deg2rad(camera.phi)

        # Vectorized camera initial conditions (eq. 11, corrected sign;
        # xf = -x_screen implements the paper's screen orientation).
        xs = (np.linspace(camera.x[0], camera.x[1], nx, endpoint=False) +
              0.5*(camera.x[1] - camera.x[0])/nx)
        ys = (np.linspace(camera.y[0], camera.y[1], ny, endpoint=False) +
              0.5*(camera.y[1] - camera.y[0])/ny)
        XF, YY = np.meshgrid(-xs, ys, indexing="ij")
        XF, YY = XF.ravel(), YY.ravel()
        gd = st.metric_cov(r0, th0)
        at = np.sqrt(gd[4]/(gd[1]*gd[1] - gd[0]*gd[4]))
        pts_ = 1.0/at - XF*gd[1]/(r0*np.sqrt(gd[4]))
        pph_ = -XF*np.sqrt(gd[4])/r0
        pth = YY*np.sqrt(gd[3])/r0
        pr = np.sqrt(gd[2]*np.clip(1.0 - (XF/r0)**2 - (YY/r0)**2, 0, None))
        n_ray = nx*ny
        y0s = np.zeros((n_ray, 6))
        y0s[:, 1] = r0
        y0s[:, 2] = th0
        y0s[:, 3] = ph0
        y0s[:, 4] = pr
        y0s[:, 5] = pth

        nhat, d = src.plane
        r_max = 1.2*max(r0, float(np.linalg.norm(src._c)))
        status, youts = _core.raytracer.trace_bundle_to_plane(
            st.mid, st.par, pts_, pph_, y0s, _method_id(self.method),
            self.rtol, self.atol, -1.0, nhat, d, r_max, max_steps)

        rgb = np.full((n_ray, 3), background, dtype=float)
        hit = status == STATUS_SCREEN
        if hit.any():
            pts3 = bl_to_cart(youts[hit, 1], youts[hit, 2], youts[hit, 3])
            u, v = src.to_local(pts3)
            rgb[hit], _ = src.sample_color(u, v, background=background)

        import dataclasses
        return Photograph(rgb=rgb.reshape(nx, ny, 3),
                          status=status.reshape(nx, ny),
                          extent=camera.extent,
                          meta={"spacetime": metadata(st),
                                "camera": dataclasses.asdict(camera)})

    # ------------------------------------------------------ visualization

    def visualize3d(self, ax=None, show_rays=True, max_rays=200,
                    show_surfaces=True, elev=18.0, azim=-60.0):
        """3D view of the scene: capture surface, disk annulus, planar
        surfaces, and traced rays (captured black, escaped colored)."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_subplot(projection="3d")

        st = self.physical.spacetime
        uu, vv = np.meshgrid(np.linspace(0, 2*np.pi, 40),
                             np.linspace(0, np.pi, 20))
        rh = st.capture_radius
        ax.plot_surface(rh*np.cos(uu)*np.sin(vv), rh*np.sin(uu)*np.sin(vv),
                        rh*np.cos(vv), color="black", shade=False)
        if self.physical.disk is not None:
            disk = self.physical.disk
            r_in = disk.r_in if disk.r_in is not None else st.isco
            rr, pp = np.meshgrid(np.linspace(r_in, disk.r_out, 12),
                                 np.linspace(0, 2*np.pi, 60))
            ax.plot_surface(rr*np.cos(pp), rr*np.sin(pp), 0.0*rr,
                            color="darkorange", alpha=0.25, shade=False)
        if show_surfaces:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            for surf, col in ([(s, "#4477cc") for s in self.physical.sources]
                              + [(s, "#888888") for s in self.screens]):
                ax.add_collection3d(Poly3DCollection(
                    [surf.corners()], alpha=0.25, facecolor=col,
                    edgecolor="black"))
        if show_rays:
            cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            k = 0
            for ray in self.rays[:max_rays]:
                if ray.status == STATUS_CAPTURED:
                    c, lw, al = "black", 0.7, 0.9
                else:
                    c = ray.color if ray.color is not None else cyc[k % len(cyc)]
                    lw, al = 0.8, 0.8
                    k += 1
                ax.plot(ray.points[:, 0], ray.points[:, 1],
                        ray.points[:, 2], color=c, lw=lw, alpha=al)
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$z$")
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        return ax
