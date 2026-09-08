"""Stationary axisymmetric metric imports, in (-+++) signature and G=c=M=1.

Only (tt,tphi,rr,thetatheta,phiphi) may be nonzero. Coordinates are
(t,r,theta,phi), with angles in radians. Tables use x=log(r-origin),
mu=cos(theta); the inverse angular components are regularized by r^2 and
r^2 sin^2(theta). This is a geometric interface, not a field-equation solver.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np

from . import _core
from .spacetime import Spacetime
from ._runtime import metric_context

_active_table = None


def _gradient(values, step, axis):
    """Fourth-order centered slopes, one-sided second order at grid edges."""
    v = np.moveaxis(values, axis, 0)
    d = np.gradient(v, step, axis=0, edge_order=2)
    d[2:-2] = (v[:-4] - 8 * v[1:-3] + 8 * v[3:-1] - v[4:]) / (12 * step)
    return np.moveaxis(d, 0, axis)


def inverse_components(g):
    """Invert the five-component circular metric, preserving leading axes."""
    g = np.asarray(g, dtype=float)
    det = g[..., 0] * g[..., 4] - g[..., 1] ** 2
    return np.stack(
        (
            g[..., 4] / det,
            -g[..., 1] / det,
            1 / g[..., 2],
            1 / g[..., 3],
            g[..., 0] / det,
        ),
        axis=-1,
    )


class TabulatedMetric(Spacetime):
    """Import a stationary metric from a uniform log-radius/cosine grid.

    ``covariant`` has shape (nr,nmu,5), evaluated at r and theta=acos(mu).
    Radial samples must be uniform in log(r-radial_origin), nr >= 8.
    mu runs uniformly from -1 to 1; at endpoints supply limiting values
    evaluated at acos(clip(mu,-1+1e-10,1-1e-10)).

    ``inner_radius`` is the absorbing boundary (horizon buffer or surface).
    The table must extend below it to support RK stages. Outside the table
    evaluation returns NaN to Fortran (step rejection), never extrapolation.
    Refining integration tolerance does NOT remove table interpolation error.
    """

    mid = 3

    def __init__(
        self,
        r,
        mu,
        covariant,
        *,
        inner_radius,
        radial_origin=0.0,
        name="tabulated",
        provenance=None,
        boundary_kind="absorbing",
    ):
        if boundary_kind not in ("absorbing", "surface"):
            raise ValueError("boundary_kind must be absorbing or surface")
        self.boundary_kind = boundary_kind
        r, mu, gd = (np.array(a, dtype=float, copy=True) for a in (r, mu, covariant))
        if r.ndim != 1 or mu.ndim != 1 or min(r.size, mu.size) < 8:
            raise ValueError("metric axes must be 1D with at least 8 samples")
        if (
            not np.isfinite(r).all()
            or not np.isfinite(mu).all()
            or not np.isfinite(radial_origin)
            or np.any(r <= radial_origin)
        ):
            raise ValueError("invalid radial domain")
        x = np.log(r - radial_origin)
        if (
            np.any(np.diff(x) <= 0)
            or not np.allclose(np.diff(x), np.diff(x)[0], rtol=1e-7)
            or not np.allclose(mu, np.linspace(-1, 1, len(mu)), atol=1e-12)
        ):
            raise ValueError("use a uniform log(r-origin) grid and mu=linspace(-1,1,n)")
        if not r[0] < inner_radius < r[-1] or inner_radius <= 0:
            raise ValueError(
                "inner_radius must lie strictly inside the positive radial domain"
            )
        if gd.shape != (len(r), len(mu), 5) or not np.isfinite(gd).all():
            raise ValueError("covariant must be finite with shape (nr,nmu,5)")
        det = gd[..., 0] * gd[..., 4] - gd[..., 1] ** 2
        if np.any(det >= 0) or np.any(gd[..., 2:] <= 0):
            raise ValueError(
                "metric must have Lorentzian signature and positive spatial diagonal"
            )
        gu = inverse_components(gd)
        gu[..., 3] *= r[:, None] ** 2
        gu[..., 4] *= r[:, None] ** 2 * (1 - np.clip(mu, -1 + 1e-10, 1 - 1e-10) ** 2)
        dx, dm = x[1] - x[0], mu[1] - mu[0]
        fx = _gradient(gu, dx, axis=0)
        fy = _gradient(gu, dm, axis=1)
        fxy = _gradient(fx, dm, axis=1)
        self._data = np.asfortranarray(
            np.stack((gu, fx * dx, fy * dm, fxy * dx * dm), axis=0).transpose(
                3, 0, 1, 2
            )
        )
        self._r, self._mu, self._covariant = r, mu, gd
        for a in (self._data, r, mu, gd):
            a.setflags(write=False)
        self._x0, self._dx = float(x[0]), float(dx)
        self.radial_origin = float(radial_origin)
        self.inner_radius = float(inner_radius)
        self.name = str(name)
        self.provenance = dict(provenance or {})
        self.digest = hashlib.sha256(self._data.tobytes()).hexdigest()

    @property
    def par(self):
        # The legacy tracer adds 0.01 to its inner boundary.
        return np.array([self.inner_radius - 0.01, 0.0, 0.0, 0.0])

    @property
    def capture_radius(self):
        return self.inner_radius

    @property
    def radial_domain(self):
        return float(self._r[0]), float(self._r[-1])

    def _activate(self):
        global _active_table
        if _active_table is not self:
            _core.tabulated_metric.set_metric_table(
                self._data, self._x0, self._dx, self.radial_origin
            )
            _active_table = self

    @metric_context
    def metric_cov(self, r, theta):
        self._check_radius(r)
        return _core.spacetime.metric_cov(self.mid, self.par, r, theta)

    @metric_context
    def metric_contra(self, r, theta):
        self._check_radius(r)
        return _core.spacetime.metric_contra(self.mid, self.par, r, theta)

    def _check_radius(self, r):
        if not self._r[0] <= r <= self._r[-1]:
            raise ValueError(
                f"radius {r} is outside metric domain {self.radial_domain}"
            )

    def metadata(self):
        return {
            "type": type(self).__name__,
            "name": self.name,
            "sha256": self.digest,
            "shape": list(self._covariant.shape),
            "radial_domain": list(self.radial_domain),
            "radial_origin": self.radial_origin,
            "inner_radius": self.inner_radius,
            "boundary_kind": self.boundary_kind,
            "interpolation": "bicubic-Hermite inverse metric",
            "provenance": self.provenance,
        }

    def save(self, path):
        np.savez_compressed(
            path,
            r=self._r,
            mu=self._mu,
            covariant=self._covariant,
            metadata=json.dumps(self.metadata()),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["metadata"]))
            return TabulatedMetric(
                d["r"],
                d["mu"],
                d["covariant"],
                inner_radius=meta["inner_radius"],
                radial_origin=meta["radial_origin"],
                name=meta["name"],
                provenance=meta.get("provenance"),
                boundary_kind=meta.get("boundary_kind", "absorbing"),
            )


class CustomMetric(TabulatedMetric):
    """Sample an analytic callback ``covariant(r,theta) -> (...,5)`` once.

    The callback must broadcast over NumPy arrays. The compiled CPU tracer
    uses its interpolated geometry; callbacks are never invoked per RK step.
    No recompilation is required. Double the grid to assess metric error.
    """

    def __init__(
        self,
        covariant,
        *,
        r_min,
        r_max,
        inner_radius,
        radial_origin=0.0,
        resolution=(768, 129),
        name="custom",
        provenance=None,
        boundary_kind="absorbing",
    ):
        nr, nm = resolution
        if r_min <= radial_origin or r_max <= r_min:
            raise ValueError("require radial_origin < r_min < r_max")
        r = radial_origin + np.geomspace(
            r_min - radial_origin, r_max - radial_origin, nr
        )
        mu = np.linspace(-1, 1, nm)
        th = np.arccos(np.clip(mu, -1 + 1e-10, 1 - 1e-10))
        rr, tt = np.broadcast_arrays(r[:, None], th[None, :])
        gd = np.asarray(covariant(rr, tt), dtype=float)
        super().__init__(
            r,
            mu,
            gd,
            inner_radius=inner_radius,
            radial_origin=radial_origin,
            name=name,
            provenance=provenance,
            boundary_kind=boundary_kind,
        )


def spherical_components(r, theta, lapse_squared):
    """Static spherical exterior with g_rr=1/f; useful analytic building block."""
    return np.stack(
        (
            -lapse_squared,
            np.zeros_like(r),
            1 / lapse_squared,
            r * r,
            r * r * np.sin(theta) ** 2,
        ),
        axis=-1,
    )


class ReissnerNordstrom(CustomMetric):
    """Charged spherical black hole; mass=1, |charge|<1 (nonextremal).

    This is an Einstein-Maxwell comparison geometry, not a claim of large
    astrophysical black-hole charge. Photons are neutral vacuum test rays.
    """

    def __init__(self, charge=0.0, *, r_max=3000.0, resolution=(768, 65)):
        if not np.isfinite(charge) or abs(charge) >= 1:
            raise ValueError("require finite |charge| < 1")
        self.charge = float(charge)
        self.horizon = 1 + np.sqrt(1 - charge**2)
        super().__init__(
            lambda r, t: spherical_components(r, t, 1 - 2 / r + charge**2 / r**2),
            r_min=self.horizon + 1e-4,
            r_max=r_max,
            inner_radius=self.horizon + 0.01,
            radial_origin=self.horizon,
            resolution=resolution,
            name="Reissner-Nordstrom",
            provenance={"mass": 1.0, "charge": self.charge},
        )


class SphericalStar(CustomMetric):
    """Schwarzschild vacuum exterior terminated at a spherical stellar surface.

    No equation of state or interior is assumed. radius=5 means R=5 GM/c^2.
    Surface emission is configured separately with EmittingSurface.
    """

    def __init__(self, radius=5.0, *, r_max=3000.0, resolution=(512, 33)):
        if not np.isfinite(radius) or radius <= 2.02:
            raise ValueError("stellar radius must exceed 2.02 M")
        self.radius = float(radius)
        super().__init__(
            lambda r, t: spherical_components(r, t, 1 - 2 / r),
            r_min=2.0001,
            r_max=r_max,
            inner_radius=radius,
            radial_origin=2.0,
            resolution=resolution,
            name="spherical stellar exterior",
            provenance={"mass": 1.0, "radius": self.radius},
            boundary_kind="surface",
        )
