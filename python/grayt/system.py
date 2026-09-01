"""Scene-level abstractions: ray sources, screens, and 3D visualization.

Layer model
-----------
- ``PhysicalSystem`` — the physics: the spacetime (``BlackHole``), the
  matter/optical objects living in it (``ThinDisk``, ``ImageSource``),
  and the rays traced through it.
- ``System`` — the laboratory: a ``PhysicalSystem`` plus the measuring
  devices (``Camera`` for backward rendering, ``Screen`` surfaces where
  formed images appear) and visualization utilities (3D viewer).

Geometry: surfaces live in the pseudo-Cartesian embedding
x = r sin(th) cos(ph), y = r sin(th) sin(ph), z = r cos(th); this is exact
only asymptotically, so sources and screens should sit far from the hole
(r >> M). The spin axis is +z.

Image formation: an ``ImageSource`` holds a loaded image; each pixel
knows only color/illumination, not direction. In general pixel k could
emit n_k rays distributed over (theta, phi); the default emission model
is ``"collimated"`` — one ray per sampled pixel, perpendicular to the
image plane — which keeps the cost linear in pixels. The architecture
leaves room for angular emission models later.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _core
from .api import (BlackHole, Camera, ThinDisk, _METHODS, STATUS_ESCAPED,
                  STATUS_CAPTURED, STATUS_FAILED)

STATUS_SCREEN = 2  # TRACE_TO_PLANE: ray reached the target plane


# ----------------------------------------------------------------- geometry

def bl_to_cart(r, theta, phi):
    """Boyer-Lindquist -> pseudo-Cartesian embedding."""
    s = np.sin(theta)
    return np.stack([r*s*np.cos(phi), r*s*np.sin(phi), r*np.cos(theta)],
                    axis=-1)


def cart_to_bl(p):
    """Pseudo-Cartesian point -> (r, theta, phi)."""
    x, y, z = p
    r = np.sqrt(x*x + y*y + z*z)
    return r, np.arccos(np.clip(z/r, -1, 1)), np.arctan2(y, x)


def null_momentum(black_hole: BlackHole, point_cart, direction_cart):
    """Null 4-momentum at ``point_cart`` moving along ``direction_cart``.

    The spatial direction is expressed in the ZAMO orthonormal frame
    (coordinate direction mapped through the embedding Jacobian), and the
    photon energy in that frame is normalized to 1. Returns
    ``(y0, p_t, p_phi)`` ready for the Fortran tracer.
    """
    r, th, ph = cart_to_bl(np.asarray(point_cart, float))
    d = np.asarray(direction_cart, float)
    d = d/np.linalg.norm(d)

    st, ct = np.sin(th), np.cos(th)
    cp, sp = np.cos(ph), np.sin(ph)
    # Jacobian d(x,y,z)/d(r,th,ph) of the embedding
    jac = np.array([[st*cp, r*ct*cp, -r*st*sp],
                    [st*sp, r*ct*sp, r*st*cp],
                    [ct, -r*st, 0.0]])
    dr, dth, dph = np.linalg.solve(jac, d)

    gd = _core.kerr_metric.metric_cov(black_hole.a, r, th)
    g_tt, g_tp, g_rr, g_thth, g_pp = gd
    # Orthonormal (ZAMO) components of the spatial direction
    n = np.array([np.sqrt(g_rr)*dr, np.sqrt(g_thth)*dth, np.sqrt(g_pp)*dph])
    n = n/np.linalg.norm(n)

    gu, _, _ = _core.kerr_metric.metric_contra(black_hole.a, r, th)
    alpha = 1.0/np.sqrt(-gu[0])
    u_t = np.sqrt(-gu[0])            # ZAMO u^t
    u_p = -alpha*gu[1]               # ZAMO u^phi
    p_up_t = u_t
    p_up_r = n[0]/np.sqrt(g_rr)
    p_up_th = n[1]/np.sqrt(g_thth)
    p_up_ph = u_p + n[2]/np.sqrt(g_pp)

    p_t = g_tt*p_up_t + g_tp*p_up_ph
    p_phi = g_tp*p_up_t + g_pp*p_up_ph
    p_r = g_rr*p_up_r
    p_th = g_thth*p_up_th
    y0 = np.array([0.0, r, th, ph, p_r, p_th])
    return y0, float(p_t), float(p_phi)


# ----------------------------------------------------------------- surfaces

@dataclass
class PlanarSurface:
    """Finite rectangular plane: ``center`` + span along ``(e1, e2)``,
    where e1 = up x normal (width direction) and e2 completes the basis."""

    center: tuple = (0.0, 0.0, 0.0)
    normal: tuple = (1.0, 0.0, 0.0)
    up: tuple = (0.0, 0.0, 1.0)
    width: float = 40.0
    height: float = 40.0

    def __post_init__(self):
        n = np.asarray(self.normal, float)
        self._n = n/np.linalg.norm(n)
        u = np.asarray(self.up, float)
        e1 = np.cross(u, self._n)
        self._e1 = e1/np.linalg.norm(e1)
        self._e2 = np.cross(self._n, self._e1)
        self._c = np.asarray(self.center, float)
        self._d = float(self._n @ self._c)

    @property
    def plane(self):
        """(normal, d) with plane equation n.x = d."""
        return self._n, self._d

    def to_local(self, points):
        """Cartesian points -> in-plane coordinates (u, v)."""
        rel = np.atleast_2d(points) - self._c
        return rel @ self._e1, rel @ self._e2

    def to_world(self, u, v):
        return (self._c + np.multiply.outer(u, self._e1) +
                np.multiply.outer(v, self._e2))

    def corners(self):
        w, h = 0.5*self.width, 0.5*self.height
        return np.array([self.to_world(su*w, sv*h)
                         for su, sv in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])


@dataclass
class ImageSource(PlanarSurface):
    """A loaded image placed on a plane; the source of rays.

    ``image`` is an (H, W, 3) float array in [0, 1] or a path to load.
    ``emission="collimated"``: every sampled pixel emits one ray along
    the surface normal (the direction-agnostic default; angular emission
    models n_k(theta, phi) can be added later without changing callers).
    """

    image: object = None
    emission: str = "collimated"

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.image, (str, bytes)):
            import matplotlib.image as mpimg
            arr = mpimg.imread(self.image)
            if arr.dtype.kind in "ui":
                arr = arr/255.0
            self.image = np.asarray(arr[..., :3], float)
        elif self.image is None:
            raise ValueError("ImageSource requires an image array or path")
        if self.emission != "collimated":
            raise NotImplementedError(
                "only the collimated emission model is implemented")

    def sample_rays(self, max_rays=40_000):
        """Subsample pixels -> (positions (N,3), colors (N,3), uv).

        Pixel row 0 is the top of the image (matplotlib convention); it
        maps to +e2 (the ``up`` side of the surface).
        """
        h, w = self.image.shape[:2]
        stride = max(1, int(np.ceil(np.sqrt(h*w/max_rays))))
        ii, jj = np.meshgrid(np.arange(0, h, stride),
                             np.arange(0, w, stride), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        u = ((jj + 0.5)/w - 0.5)*self.width
        v = (0.5 - (ii + 0.5)/h)*self.height
        pos = self.to_world(u, v)
        colors = self.image[ii, jj]
        return pos, colors, (u, v)


@dataclass
class Screen(PlanarSurface):
    """Detector surface where formed images appear.

    Accumulates ray hits into a pixel buffer; ``image`` averages the
    colors deposited in each bin (black where nothing arrived).
    """

    resolution: tuple = (256, 256)

    def __post_init__(self):
        super().__post_init__()
        self.reset()

    def reset(self):
        nu, nv = self.resolution
        self._rgb = np.zeros((nu, nv, 3))
        self._count = np.zeros((nu, nv), dtype=int)

    def add_hits(self, points_cart, colors):
        """Bin Cartesian hit points into screen pixels."""
        if len(points_cart) == 0:
            return 0
        u, v = self.to_local(points_cart)
        nu, nv = self.resolution
        iu = np.floor((u/self.width + 0.5)*nu).astype(int)
        iv = np.floor((v/self.height + 0.5)*nv).astype(int)
        ok = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
        np.add.at(self._rgb, (iu[ok], iv[ok]), colors[ok])
        np.add.at(self._count, (iu[ok], iv[ok]), 1)
        return int(ok.sum())

    @property
    def image(self):
        """(H, W, 3) formed image, rows top-down for imshow."""
        img = np.where(self._count[..., None] > 0,
                       self._rgb/np.maximum(self._count[..., None], 1), 0.0)
        return np.clip(img.transpose(1, 0, 2)[::-1], 0.0, 1.0)


# ------------------------------------------------------------------- rays

@dataclass
class Ray:
    """A traced geodesic in Cartesian coordinates, for visualization."""

    points: np.ndarray            # (N, 3)
    status: int
    color: object = None


# --------------------------------------------------------- physical system

@dataclass
class PhysicalSystem:
    """The physics: spacetime + objects living in it + traced rays."""

    black_hole: BlackHole = field(default_factory=BlackHole)
    disk: ThinDisk | None = None
    sources: list = field(default_factory=list)
    rays: list = field(default_factory=list)
    method: str = "rkdp45"
    rtol: float = 1e-9
    atol: float = 1e-11

    def trace_ray(self, origin_cart, direction_cart, lambda_max=400.0,
                  keep=True):
        """Trace a single geodesic from a point along a direction and
        store it (for the 3D viewer). Returns the ``Ray``."""
        y0, pt, pphi = null_momentum(self.black_hole, origin_cart,
                                     direction_cart)
        traj, herr, nout = _core.raytracer.trace_geodesic(
            self.black_hole.a, y0, pt, pphi, _METHODS[self.method],
            self.rtol, self.atol, 1.0, lambda_max, 50_000)
        t = traj[:nout]
        pts = bl_to_cart(t[:, 2], t[:, 3], t[:, 4])
        pts = np.vstack([np.atleast_2d(np.asarray(origin_cart, float)), pts])
        captured = t[-1, 2] <= self.black_hole.horizon + 0.05 if nout else False
        ray = Ray(points=pts,
                  status=STATUS_CAPTURED if captured else STATUS_ESCAPED)
        if keep:
            self.rays.append(ray)
        return ray

    def propagate_to_screen(self, source: ImageSource, screen: Screen,
                            max_rays=40_000, r_max=None, max_steps=100_000,
                            keep_sample_rays=0):
        """Collimated image formation: one ray per sampled source pixel,
        emitted along the source normal, traced until it hits the screen
        plane, falls into the hole, or escapes. Screen accumulates hits.

        Returns a dict with per-ray status counts.
        """
        pos, colors, _ = source.sample_rays(max_rays=max_rays)
        n_ray = len(pos)
        y0s = np.empty((n_ray, 6))
        pts_ = np.empty(n_ray)
        pph_ = np.empty(n_ray)
        for i in range(n_ray):
            y0s[i], pts_[i], pph_[i] = null_momentum(
                self.black_hole, pos[i], source._n)

        nhat, d = screen.plane
        if r_max is None:
            r_max = 4.0*max(np.linalg.norm(source._c),
                            np.linalg.norm(screen._c), 50.0)
        status, youts = _core.raytracer.trace_bundle_to_plane(
            self.black_hole.a, pts_, pph_, y0s, _METHODS[self.method],
            self.rtol, self.atol, 1.0, nhat, d, r_max, max_steps)

        hit = status == STATUS_SCREEN
        hit_pts = bl_to_cart(youts[hit, 1], youts[hit, 2], youts[hit, 3])
        n_binned = screen.add_hits(hit_pts, colors[hit])

        if keep_sample_rays:
            idx = np.linspace(0, n_ray - 1, keep_sample_rays).astype(int)
            for i in idx:
                self.trace_ray(pos[i], source._n, lambda_max=3.0*r_max)

        counts = {int(k): int(v)
                  for k, v in zip(*np.unique(status, return_counts=True))}
        return {"n_rays": n_ray, "status_counts": counts,
                "n_on_screen": n_binned}


# ----------------------------------------------------------------- system

@dataclass
class System:
    """The laboratory: physics + instruments + visualization."""

    physical: PhysicalSystem = field(default_factory=PhysicalSystem)
    cameras: list = field(default_factory=list)
    screens: list = field(default_factory=list)

    def render(self, camera: Camera | int = 0, **kwargs):
        """Backward-ray-trace an intensity image (disk scene) with the
        given camera (index into ``self.cameras`` or a Camera object)."""
        from .api import render as _render
        cam = self.cameras[camera] if isinstance(camera, int) else camera
        return _render(self.physical.black_hole, cam, self.physical.disk,
                       method=self.physical.method, **kwargs)

    def form_image(self, source: ImageSource | int = 0,
                   screen: Screen | int = 0, **kwargs):
        """Propagate a loaded image through the spacetime onto a screen."""
        src = (self.physical.sources[source]
               if isinstance(source, int) else source)
        scr = self.screens[screen] if isinstance(screen, int) else screen
        return self.physical.propagate_to_screen(src, scr, **kwargs)

    def visualize3d(self, ax=None, show_rays=True, max_rays=200,
                    show_surfaces=True, elev=18.0, azim=-60.0):
        """3D view of the scene: horizon sphere, disk annulus, surfaces,
        and traced rays (captured black, escaped colored) — Fig. 1 style."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_subplot(projection="3d")

        bh = self.physical.black_hole
        # Event horizon
        uu, vv = np.meshgrid(np.linspace(0, 2*np.pi, 40),
                             np.linspace(0, np.pi, 20))
        rh = bh.horizon
        ax.plot_surface(rh*np.cos(uu)*np.sin(vv), rh*np.sin(uu)*np.sin(vv),
                        rh*np.cos(vv), color="black", shade=False)
        # Disk annulus
        if self.physical.disk is not None:
            disk = self.physical.disk
            r_in = disk.r_in if disk.r_in else bh.isco
            rr, pp = np.meshgrid(np.linspace(r_in, disk.r_out, 12),
                                 np.linspace(0, 2*np.pi, 60))
            ax.plot_surface(rr*np.cos(pp), rr*np.sin(pp), 0.0*rr,
                            color="darkorange", alpha=0.25, shade=False)
        # Sources and screens
        if show_surfaces:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            for surf, col in ([(s, "#4477cc") for s in self.physical.sources]
                              + [(s, "#888888") for s in self.screens]):
                ax.add_collection3d(Poly3DCollection(
                    [surf.corners()], alpha=0.25, facecolor=col,
                    edgecolor="black"))
        # Rays
        if show_rays:
            cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            shown = self.physical.rays[:max_rays]
            k = 0
            for ray in shown:
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
