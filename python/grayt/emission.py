"""Prescribed radiation and emitter kinematics, independent of geodesics.

Emissivities are local *bolometric specific intensities*, in user-chosen
consistent units. The observer receives g^4 I_em. No gas dynamics are solved.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from ._runtime import metadata


def metric_samples(spacetime, r, theta):
    """Evaluate metric components on broadcast positions."""
    r, theta = np.broadcast_arrays(r, theta)
    if spacetime.mid == 1:
        a = spacetime.a
        s2 = np.sin(theta) ** 2
        sigma = r * r + a * a * np.cos(theta) ** 2
        delta = r * r - 2 * r + a * a
        return np.stack(
            (
                -(1 - 2 * r / sigma),
                -2 * a * r * s2 / sigma,
                sigma / delta,
                sigma,
                (r * r + a * a + 2 * a * a * r * s2 / sigma) * s2,
            ),
            axis=-1,
        )
    return np.array(
        [
            spacetime.metric_cov(float(rr), float(tt))
            for rr, tt in zip(r.ravel(), theta.ravel())
        ]
    ).reshape(r.shape + (5,))


def circular_velocity(spacetime, r, theta, phi, *, sense=1):
    """Equatorial circular-geodesic u^mu, choosing positive/negative Omega.

    A circular orbit must exist and be timelike; stability and the chosen
    disk inner edge remain the caller's responsibility for custom metrics.
    """
    if sense not in (-1, 1):
        raise ValueError("sense must be -1 or 1")
    eps = np.maximum(np.abs(r) * 1e-5, 1e-6)
    gp = metric_samples(spacetime, r + eps, theta)
    gm = metric_samples(spacetime, r - eps, theta)
    dg = (gp - gm) / (2 * eps[..., None])
    discriminant = dg[..., 1] ** 2 - dg[..., 0] * dg[..., 4]
    if np.any(discriminant < 0) or np.any(np.abs(dg[..., 4]) < 1e-20):
        raise ValueError("no circular geodesic exists at some emitting positions")
    omega = (-dg[..., 1] + sense * np.sqrt(discriminant)) / dg[..., 4]
    return rotating_velocity(spacetime, r, theta, omega)


def rotating_velocity(spacetime, r, theta, omega=0.0):
    """Normalized circular u^mu for a prescribed angular velocity."""
    g = metric_samples(spacetime, r, theta)
    norm = -(g[..., 0] + 2 * omega * g[..., 1] + omega**2 * g[..., 4])
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0):
        raise ValueError("emitter velocity must be timelike (subluminal)")
    ut = 1 / np.sqrt(norm)
    return np.stack((ut, np.zeros_like(ut), np.zeros_like(ut), ut * omega), axis=-1)


def frequency_shift(spacetime, r, theta, momentum, velocity):
    """g=E_obs/E_em using E_obs=1 and the legacy past-oriented covector.

    ``momentum`` is ordered (p_t,p_r,p_theta,p_phi). For this camera convention
    E_em = p_mu u^mu > 0. Validate normalization rather than clamp bad speeds.
    """
    u = np.broadcast_to(np.asarray(velocity, float), momentum.shape)
    gd = metric_samples(spacetime, r, theta)
    norm = (
        gd[..., 0] * u[..., 0] ** 2
        + 2 * gd[..., 1] * u[..., 0] * u[..., 3]
        + gd[..., 2] * u[..., 1] ** 2
        + gd[..., 3] * u[..., 2] ** 2
        + gd[..., 4] * u[..., 3] ** 2
    )
    if (
        not np.isfinite(u).all()
        or np.any(u[..., 0] <= 0)
        or not np.allclose(norm, -1, atol=1e-6)
    ):
        raise ValueError(
            "four_velocity must be future-directed and normalized: g(u,u)=-1"
        )
    energy = np.sum(momentum * u, axis=-1)
    if np.any(energy <= 0) or not np.isfinite(energy).all():
        raise ValueError("nonpositive emitted photon energy")
    return 1 / energy


@dataclass
class EmittingDisk:
    """Opaque equatorial emitting annulus, independent of the metric model.

    intensity(r,phi,t) -> nonnegative bolometric I_em; t is the emission
    coordinate time (supports a prescribed moving pattern on a fixed metric).
    four_velocity(metric,r,theta,phi) -> (...,4); defaults to circular geodesics.
    Callbacks must broadcast over arrays. Custom spectra/volumes are separate.
    """

    r_in: float
    r_out: float
    intensity: object
    four_velocity: object = circular_velocity
    name: str = "custom emitting disk"
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if (
            not np.isfinite([self.r_in, self.r_out]).all()
            or not 0 < self.r_in < self.r_out
        ):
            raise ValueError("require finite 0 < r_in < r_out")
        if not callable(self.intensity) or not callable(self.four_velocity):
            raise TypeError("intensity and four_velocity must be callables")

    def metadata(self):
        return {
            "type": type(self).__name__,
            "name": self.name,
            "r_in": self.r_in,
            "r_out": self.r_out,
            "radiation": "bolometric intensity; g^4 transfer",
            "provenance": self.provenance,
        }


class PageThorneDisk(EmittingDisk):
    """Kerr Page-Thorne emission with consistent circular-geodesic motion.

    F/pi converts the local one-face flux to isotropic bolometric intensity.
    Accretion normalization is Mdot/(4*pi)=1, matching the legacy flux table.
    This is separate from ThinDisk's historical constant-l0, g^3 convention.
    """

    def __init__(self, spacetime, r_out=20.0, n=4000):
        from .api import flux_profile

        if spacetime.mid != 1 or abs(spacetime.a) >= 1:
            raise ValueError("PageThorneDisk requires nonextremal Kerr")
        if not np.isfinite(r_out) or r_out <= spacetime.isco or n < 8:
            raise ValueError("require r_out > ISCO and n >= 8")
        r, f = flux_profile(spacetime, r_out=r_out, n=n)
        self.spacetime_metadata = metadata(spacetime)
        super().__init__(
            r_in=spacetime.isco,
            r_out=r_out,
            intensity=lambda rr, phi, t: np.interp(rr, r, f) / np.pi,
            name="Page-Thorne / circular geodesics",
            provenance={"metric": self.spacetime_metadata, "flux_samples": n},
        )


@dataclass
class EmittingSurface:
    """Bolometric emission at a metric's spherical absorbing inner boundary.

    intensity(theta,phi,t) -> I_em. Intended for SphericalStar, not horizons.
    A rotating brightness pattern does not automatically rotate the material;
    supply a consistent four_velocity if the surface rotates.
    """

    intensity: object
    four_velocity: object = field(
        default=lambda st, r, th, ph: rotating_velocity(st, r, th)
    )
    name: str = "stellar surface"

    def metadata(self):
        return {
            "type": type(self).__name__,
            "name": self.name,
            "radiation": "bolometric; g^4",
        }


class TabulatedDisk(EmittingDisk):
    """Import a radial surface model: r, bolometric intensity, angular velocity.

    Linear interpolation; this imports an already chosen radiation model,
    not raw density or temperature from a GRMHD calculation.
    """

    def __init__(self, radius, intensity, omega, *, name="tabulated disk"):
        self.radius, self.emission, self.omega = [
            np.array(a, float, copy=True) for a in (radius, intensity, omega)
        ]
        if (
            self.radius.ndim != 1
            or len(self.radius) < 2
            or self.emission.shape != self.radius.shape
            or self.omega.shape != self.radius.shape
            or not np.isfinite([self.radius, self.emission, self.omega]).all()
            or np.any(np.diff(self.radius) <= 0)
            or np.any(self.emission < 0)
        ):
            raise ValueError(
                "require increasing radii and finite matching intensity/omega arrays"
            )
        for a in (self.radius, self.emission, self.omega):
            a.setflags(write=False)
        import hashlib

        digest = hashlib.sha256(
            np.stack((self.radius, self.emission, self.omega)).tobytes()
        ).hexdigest()
        super().__init__(
            float(self.radius[0]),
            float(self.radius[-1]),
            lambda r, ph, t: np.interp(r, self.radius, self.emission),
            lambda st, r, th, ph: rotating_velocity(
                st, r, th, np.interp(r, self.radius, self.omega)
            ),
            name=name,
            provenance={"sha256": digest, "interpolation": "linear"},
        )

    def save(self, path):
        np.savez_compressed(
            path,
            radius=self.radius,
            intensity=self.emission,
            omega=self.omega,
            name=self.name,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            return cls(d["radius"], d["intensity"], d["omega"], name=str(d["name"]))
