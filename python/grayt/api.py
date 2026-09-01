"""High-level API for the Kerr ray tracer.

Design: the user composes three ingredients and renders.

    >>> import grayt
    >>> bh = grayt.BlackHole(a=0.95)
    >>> cam = grayt.Camera(r=1000, theta=85, x=(-24, 24), y=(-12, 12),
    ...                    resolution=(1024, 512))
    >>> disk = grayt.ThinDisk(l0=1.8, r_out=20)
    >>> img = grayt.render(bh, cam, disk)
    >>> img.plot()

`Camera` is the source of rays (backward ray tracing: rays are launched
from the observer's image plane); `ThinDisk` is the matter geometry.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from . import _core

_METHODS = {"rkdp45": 1, "rkck45": 2, "rkf45": 3}

STATUS_ESCAPED = 0
STATUS_CAPTURED = 1
STATUS_DISK = 2
STATUS_FAILED = 3


@dataclass(frozen=True)
class BlackHole:
    """Kerr black hole with dimensionless spin ``a`` (geometrized M = 1)."""

    a: float = 0.0

    def __post_init__(self):
        if not -1.0 <= self.a <= 1.0:
            raise ValueError(f"spin must satisfy |a| <= 1, got {self.a}")

    @property
    def horizon(self) -> float:
        """Outer event horizon radius r_H = 1 + sqrt(1 - a^2)."""
        return float(_core.raytracer.get_horizon(self.a))

    @property
    def isco(self) -> float:
        """Prograde innermost stable circular orbit radius."""
        return float(_core.raytracer.get_isco(self.a))


@dataclass(frozen=True)
class Camera:
    """Observer image plane: the source of (backward-traced) rays.

    Parameters
    ----------
    r, theta, phi : observer position in Boyer-Lindquist coordinates
        (theta in degrees).
    x, y : image-plane extents in units of M (paper convention: the
        Doppler-approaching side of a prograde disk appears at x < 0).
    resolution : (nx, ny) pixels.
    """

    r: float = 1000.0
    theta: float = 85.0
    phi: float = 0.0
    x: tuple[float, float] = (-24.0, 24.0)
    y: tuple[float, float] = (-12.0, 12.0)
    resolution: tuple[int, int] = (512, 256)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (self.x[0], self.x[1], self.y[0], self.y[1])


@dataclass(frozen=True)
class ThinDisk:
    """Equatorial thin accretion disk (matter geometry).

    Emission follows the Page & Thorne (1974) time-averaged flux; matter
    moves on circular orbits with constant specific angular momentum
    ``l0``. ``r_in=None`` means the ISCO of the chosen black hole.
    """

    r_in: float | None = None
    r_out: float = 20.0
    l0: float = 2.8
    model: str = "page-thorne"

    def __post_init__(self):
        if self.model != "page-thorne":
            raise ValueError(f"unknown disk model: {self.model!r}")


@dataclass
class Image:
    """Result of a render: per-pixel maps, oriented as (nx, ny) arrays.

    All maps are indexed [i, j] with i along x and j along y; ``extent``
    gives (xmin, xmax, ymin, ymax) for plotting.
    """

    intensity: np.ndarray
    g: np.ndarray
    r_hit: np.ndarray
    status: np.ndarray
    herr: np.ndarray
    theta_inf: np.ndarray
    phi_inf: np.ndarray
    extent: tuple[float, float, float, float]
    meta: dict = field(default_factory=dict)

    def plot(self, ax=None, cmap="afmhot", norm_to=None, colorbar=True,
             label=None, vmax=None):
        from .plotting import plot_image
        return plot_image(self, ax=ax, cmap=cmap, norm_to=norm_to,
                          colorbar=colorbar, label=label, vmax=vmax)

    def save(self, path):
        """Save all maps + metadata to a compressed .npz archive."""
        np.savez_compressed(
            path, intensity=self.intensity, g=self.g, r_hit=self.r_hit,
            status=self.status, herr=self.herr, theta_inf=self.theta_inf,
            phi_inf=self.phi_inf, extent=np.array(self.extent),
            meta=np.array(repr(self.meta)))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        return cls(intensity=d["intensity"], g=d["g"], r_hit=d["r_hit"],
                   status=d["status"], herr=d["herr"],
                   theta_inf=d["theta_inf"], phi_inf=d["phi_inf"],
                   extent=tuple(d["extent"]), meta={"loaded": str(path)})


def render(black_hole: BlackHole, camera: Camera, disk: ThinDisk | None = None,
           method: str = "rkdp45", rtol: float = 1e-8, atol: float = 1e-10,
           max_steps: int = 500_000) -> Image:
    """Trace every pixel of ``camera`` through the spacetime of
    ``black_hole``; if ``disk`` is given, shade disk intersections with
    I_obs = g^3 * F_PageThorne (eq. 19 of arXiv:2202.00086)."""
    nx, ny = camera.resolution
    disk_on = 1 if disk is not None else 0
    rin = -1.0 if (disk is None or disk.r_in is None) else disk.r_in
    rout = disk.r_out if disk is not None else 20.0
    l0 = disk.l0 if disk is not None else 0.0

    # The Fortran camera x-axis is mirrored w.r.t. the paper's figures;
    # render with x negated so that screen-x follows the paper convention.
    out = _core.raytracer.render_image(
        black_hole.a, camera.r, np.deg2rad(camera.theta),
        np.deg2rad(camera.phi), -camera.x[1], -camera.x[0],
        camera.y[0], camera.y[1], nx, ny, _METHODS[method], rtol, atol,
        disk_on, rin, rout, l0, max_steps)
    intens, gmap, rhit, status, herrm, thf, phf = (np.flip(m, axis=0) for m in out)

    meta = {"black_hole": dataclasses.asdict(black_hole),
            "camera": dataclasses.asdict(camera),
            "disk": dataclasses.asdict(disk) if disk else None,
            "method": method, "rtol": rtol, "atol": atol}
    return Image(intensity=intens, g=gmap, r_hit=rhit,
                 status=status.astype(np.int32), herr=herrm,
                 theta_inf=thf, phi_inf=phf, extent=camera.extent, meta=meta)


def shadow(black_hole: BlackHole, camera: Camera, **kwargs) -> Image:
    """Render without a disk; ``status == STATUS_CAPTURED`` is the shadow."""
    return render(black_hole, camera, disk=None, **kwargs)


def trace(black_hole: BlackHole, y0, p_t: float, p_phi: float,
          method: str = "rkdp45", rtol: float = 1e-10, atol: float = 1e-12,
          h0: float = 1.0, lambda_max: float = 1e4, n_max: int = 100_000):
    """Integrate a single geodesic from raw initial conditions.

    ``y0 = (t, r, theta, phi, p_r, p_theta)``; returns a dict with the
    trajectory columns and the Hamiltonian constraint error per step.
    """
    traj, herr, nout = _core.raytracer.trace_geodesic(
        black_hole.a, np.asarray(y0, dtype=float), p_t, p_phi,
        _METHODS[method], rtol, atol, h0, lambda_max, n_max)
    t = traj[:nout]
    return {"lambda": t[:, 0], "t": t[:, 1], "r": t[:, 2],
            "theta": t[:, 3], "phi": t[:, 4], "p_r": t[:, 5],
            "p_theta": t[:, 6], "herr": herr[:nout]}


def camera_ray(black_hole: BlackHole, camera: Camera, x: float, y: float):
    """Initial conditions (y0, p_t, p_phi) for one image-plane point."""
    y0, pt, pphi = _core.raytracer.camera_init(
        black_hole.a, camera.r, np.deg2rad(camera.theta),
        np.deg2rad(camera.phi), -x, y)
    return y0, float(pt), float(pphi)


def orbit_ic(black_hole: BlackHole, r0: float, energy: float,
             angular_momentum: float, mass: float = 1.0,
             theta0: float = 90.0, pr_sign: int = 0):
    """Initial conditions for a geodesic with conserved E and L.

    ``mass=1`` gives a time-like orbit (H = -1/2), ``mass=0`` a photon.
    ``pr_sign``: sign of the initial radial momentum (0 = turning point).
    Returns ``(y0, p_t, p_phi)`` for ``grayt.trace``.
    """
    th = np.deg2rad(theta0)
    p_t, p_phi = -energy, angular_momentum
    gu, _, _ = _core.kerr_metric.metric_contra(black_hole.a, r0, th)
    pr2 = (-mass**2 - (gu[0]*p_t**2 + 2*gu[1]*p_t*p_phi +
                       gu[4]*p_phi**2))/gu[2]
    if pr2 < -1e-12:
        raise ValueError("no orbit with these (E, L) at r0: p_r^2 < 0")
    p_r = pr_sign*np.sqrt(max(pr2, 0.0))
    return np.array([0.0, r0, th, 0.0, p_r, 0.0]), p_t, p_phi


def flux_profile(black_hole: BlackHole, r_out: float = 20.0, n: int = 2000):
    """Radial Page-Thorne flux profile on [r_isco, r_out]."""
    rs, fs = _core.disk_model.flux_profile(black_hole.a, r_out, n)
    return rs, fs
