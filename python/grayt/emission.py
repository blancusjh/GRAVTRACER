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


_E3_TAU = None
_E3_ESCAPE = None


def slab_escape_fraction(tau):
    """1 - 2 E_3(tau): emergent flux of a gray isothermal slab in units of pi S.

    Computed as 2 int_0^1 mu (1 - exp(-tau/mu)) dmu, tabulated once on a
    log grid (exact to ~1e-7 relative) and interpolated in log-log. It tends
    to 2 tau (thin) and to 1 (thick).
    """
    global _E3_TAU, _E3_ESCAPE
    if _E3_TAU is None:
        grid = np.logspace(-9, 3, 1201)
        x, w = np.polynomial.legendre.leggauss(400)
        # map [-1, 1] -> [0, 1] with mu = u^2 to resolve the mu ~ tau corner
        u = 0.5 * (x + 1)
        mu, weight = u**2, w * u        # dmu = 2u du, du = dx/2
        vals = 2 * np.sum(weight * mu * -np.expm1(-grid[:, None] / mu), axis=1)
        _E3_TAU, _E3_ESCAPE = np.log(grid), np.log(vals)
    tau = np.asarray(tau, float)
    small = tau < 1e-9
    big = tau > 1e3
    t = np.clip(tau, 1e-9, 1e3)
    out = np.exp(np.interp(np.log(t), _E3_TAU, _E3_ESCAPE))
    out = np.where(small, 2 * tau, out)
    return np.where(big, 1.0, out)


@dataclass
class SlabDisk:
    """A geometrically thin, gray emitting slab of finite optical depth.

    ``disk`` gives the emission I_disk(r, phi, t) of the opaque surface
    and the material four-velocity. ``optical_depth(r, phi)`` is the
    vertical optical depth tau_perp of the slab (a number or a callable).
    A ray crossing the slab at angle eta to its normal, measured in the
    comoving frame, sees tau = tau_perp / cos(eta) and receives
    g^4 S (1 - exp(-tau)); light from behind is attenuated by exp(-tau).
    Every crossing along the ray (all image orders, up to
    ``max_crossings``) and the celestial background are combined.

    With ``conserve_flux=True`` (default) the slab's isotropic source
    function is S = I_disk / (1 - 2 E_3(tau_perp)), so each face emits
    exactly the flux pi I_disk of the opaque disk (for Page-Thorne, the
    locally dissipated flux) at any optical depth: thin regions are
    fainter face-on and limb-brightened edge-on, but radiate the same
    energy. With ``conserve_flux=False``, S = I_disk.

    tau_perp -> infinity reproduces the opaque ``disk`` exactly. The slab
    has zero geometric thickness: no self-occultation by its height.
    """

    disk: EmittingDisk
    optical_depth: object = 1.0
    max_crossings: int = 6
    conserve_flux: bool = True
    name: str = "gray thin slab"

    def __post_init__(self):
        if not isinstance(self.disk, EmittingDisk):
            raise TypeError("SlabDisk.disk must be an EmittingDisk")
        if not callable(self.optical_depth):
            tau = float(self.optical_depth)
            if not tau > 0:
                raise ValueError("optical depth must be positive")
        if not 1 <= int(self.max_crossings) <= 32:
            raise ValueError("max_crossings must be between 1 and 32")

    @property
    def r_in(self):
        return self.disk.r_in

    @property
    def r_out(self):
        return self.disk.r_out

    def tau_perp(self, r, phi):
        if callable(self.optical_depth):
            tau = np.asarray(self.optical_depth(r, phi), float)
        else:
            tau = np.full(np.shape(r), float(self.optical_depth))
        tau = np.broadcast_to(tau, np.shape(r))
        if not np.isfinite(tau).all() or np.any(tau < 0):
            raise ValueError("optical depth must be finite and nonnegative")
        return tau

    def source_function(self, r, phi, t):
        """Isotropic comoving source function S of the slab."""
        emission = np.asarray(self.disk.intensity(r, phi, t), float)
        if not self.conserve_flux:
            return emission
        return emission / slab_escape_fraction(self.tau_perp(r, phi))

    def metadata(self):
        return {
            "type": type(self).__name__,
            "name": self.name,
            "conserve_flux": bool(self.conserve_flux),
            "source": self.disk.metadata(),
            "optical_depth": getattr(self.optical_depth, "__doc__", None)
            if callable(self.optical_depth) else float(self.optical_depth),
            "max_crossings": int(self.max_crossings),
            "radiation": "gray slab: g^4 S (1 - exp(-tau_perp / cos eta)), "
                         "all crossings, attenuated background",
        }


def spreading_disk(spacetime, r_c=8.0, tau_c=1e4, r_out=None, base=None,
                   r_trunc=None, trunc_power=8.0):
    """Page-Thorne disk with the outer edge of a viscously spreading disk.

    A steady thin disk has no natural outer edge; a real one ends where it
    has spread to. The Lynden-Bell & Pringle (1974) similarity solution
    with viscosity nu ~ r (gamma = 1) has nu Sigma ~ exp(-r / r_c) and
    Sigma ~ exp(-r / r_c) / r. So the dissipated flux is the Page-Thorne
    flux times exp(-r / r_c) (exact as r_c -> infinity), and the vertical
    optical depth is tau_perp = tau_c (r_c / r) exp(1 - r / r_c), with
    tau_c = tau_perp(r_c) (electron scattering gives 10^2-10^5 for
    luminous thin disks).

    Each element emits S (1 - exp(-tau)) with S the blackbody intensity of
    its effective temperature (``conserve_flux=False``): exact where the
    disk is optically thick, and the gas simply fades where it becomes thin.
    The energy that tail would radiate is ~1e-5 of the disk luminosity for
    the defaults; forcing it out of near-transparent gas (flux
    conservation) would require a hot optically thin phase this gray model
    does not describe. The visible disk then ends where T_eff falls below
    ~1000 K (the Wien tail collapses); with these defaults the gas turns
    transparent while still glowing, so starlight shows through the fading
    rim. ``r_out`` (default 15 r_c) only bounds the computation and lies
    beyond the visible disk.

    ``base`` replaces the Page-Thorne emission with another EmittingDisk
    (any metric); its intensity and four-velocity are tapered the same way.

    ``r_trunc`` truncates the gas itself (a finite disk, e.g. set by the
    circularization radius of the inflow or by tidal truncation): both the
    dissipation and the surface density are multiplied by
    exp(-(r / r_trunc)^trunc_power). Without it the exponential taper
    leaves a cold outer skirt that is ~1e7 times fainter than the inner
    disk yet still opaque at grazing incidence (tau / flux grows as r^2).
    """
    r_out = 15.0 * r_c if r_out is None else float(r_out)
    if base is None:
        base = PageThorneDisk(spacetime, r_out=r_out)
    elif base.r_out < r_out:
        raise ValueError("base disk must extend to r_out")
    if not (r_c > base.r_in and tau_c > 0):
        raise ValueError("require r_c > the disk's inner edge and tau_c > 0")
    if r_trunc is not None and not (r_trunc > base.r_in and trunc_power > 0):
        raise ValueError("require r_trunc > the inner edge and trunc_power > 0")

    def cut(r):
        if r_trunc is None:
            return 1.0
        return np.exp(-(np.asarray(r, float) / r_trunc) ** trunc_power)
    disk = EmittingDisk(
        base.r_in, r_out,
        lambda r, phi, t: base.intensity(r, phi, t) * np.exp(-r / r_c) * cut(r),
        base.four_velocity,
        name=f"{base.name} x exp(-r/{r_c:g}) (spreading disk)",
        provenance={"model": "Lynden-Bell & Pringle 1974, gamma=1",
                    "r_c": r_c, "tau_c": tau_c, "r_trunc": r_trunc,
                    "trunc_power": trunc_power, "base": base.metadata()})

    def tau_perp(r, phi):
        """tau_c (r_c / r) exp(1 - r / r_c)"""
        return tau_c * (r_c / r) * np.exp(1.0 - r / r_c) * cut(r)

    return SlabDisk(disk, tau_perp, conserve_flux=False,
                    name="viscously spreading thin disk")
