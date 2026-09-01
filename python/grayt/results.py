"""Results of experiments: images, photographs, trajectories.

All result types share the minimal protocol ``.plot(ax=None, **kw)``
and carry their provenance in ``.meta`` where applicable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Image:
    """Backward-render result: per-pixel maps, oriented as (nx, ny)
    arrays indexed [i, j] with i along x and j along y; ``extent`` gives
    (xmin, xmax, ymin, ymax) for plotting."""

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
        import ast
        d = np.load(path, allow_pickle=False)
        try:
            meta = ast.literal_eval(str(d["meta"]))
        except (ValueError, SyntaxError, KeyError):
            meta = {"loaded": str(path)}
        return cls(intensity=d["intensity"], g=d["g"], r_hit=d["r_hit"],
                   status=d["status"], herr=d["herr"],
                   theta_inf=d["theta_inf"], phi_inf=d["phi_inf"],
                   extent=tuple(d["extent"]), meta=meta)


@dataclass
class Photograph:
    """Lambertian backward-imaging result: an RGB picture plus the
    per-pixel ray status, oriented like ``Image`` ((nx, ny) first).

    Purely geometric: colors are sampled with no redshift factor.
    """

    rgb: np.ndarray               # (nx, ny, 3)
    status: np.ndarray            # (nx, ny)
    extent: tuple
    meta: dict = field(default_factory=dict)

    def plot(self, ax=None, label=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 6))
        ax.imshow(np.clip(self.rgb, 0, 1).transpose(1, 0, 2)[::-1],
                  origin="upper", extent=self.extent, aspect="equal",
                  interpolation="bilinear")
        if label:
            ax.text(0.04, 0.08, label, transform=ax.transAxes,
                    fontsize=12, color="black",
                    bbox=dict(facecolor="white", alpha=0.9,
                              boxstyle="round,pad=0.3"))
        return ax


@dataclass
class Ray:
    """A traced geodesic in Cartesian coordinates, for visualization."""

    points: np.ndarray            # (N, 3)
    status: int
    color: object = None


@dataclass
class Trajectory:
    """A traced geodesic in coordinate form, with the constraint monitor.

    Supports dict-style access (``traj["r"]``) for backward
    compatibility with the earlier plain-dict return of ``trace``.
    """

    lam: np.ndarray
    t: np.ndarray
    r: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    p_r: np.ndarray
    p_theta: np.ndarray
    herr: np.ndarray
    meta: dict = field(default_factory=dict)

    _ALIASES = {"lambda": "lam"}

    def __getitem__(self, key):
        return getattr(self, self._ALIASES.get(key, key))

    @property
    def points(self):
        """(N, 3) pseudo-Cartesian points (for 3D/2D orbit plotting)."""
        from .geometry import bl_to_cart
        return bl_to_cart(self.r, self.theta, self.phi)

    def plot(self, ax=None, spacetime=None, plane="xy", **kw):
        from .plotting import plot_orbits_2d
        return plot_orbits_2d([self], black_hole=spacetime, plane=plane,
                              ax=ax, **kw)
