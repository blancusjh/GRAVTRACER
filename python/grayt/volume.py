"""CPU reference transfer through prescribed stationary, gray emitting volumes.

This imports radiation coefficients, not raw GRMHD primitive variables. A
microphysical model must first convert density, electron temperature and B
into emission/absorption. Gray means frequency-independent absorption and
bolometric emission; no scattering, polarization, plasma dispersion or
time-dependent snapshot interpolation is included.
"""

from dataclasses import dataclass, field
import hashlib
import numpy as np

from ._runtime import metric_context, metadata
from .emission import metric_samples, frequency_shift
from .scene import trace_bundle, SceneImage
from .sky import compose_rgb


def integrate_gray(emissivity, absorption, g, distance):
    """Observer-directed formal solution, segments ordered camera to source.

    j is comoving bolometric intensity per proper length; alpha is inverse
    proper length; distance is positive comoving photon path length. Each
    segment has constant coefficients (exponential absorption is exact).
    Returns observed intensity and total optical depth.
    """
    j, a, g, ds = np.broadcast_arrays(
        *(np.asarray(v, float) for v in (emissivity, absorption, g, distance))
    )
    if (
        not all(np.isfinite(v).all() for v in (j, a, g, ds))
        or any(np.any(v < 0) for v in (j, a, ds))
        or np.any(g <= 0)
    ):
        raise ValueError("require finite nonnegative j, alpha, distance and positive g")
    dtau = a * ds
    foreground = np.cumsum(dtau) - dtau
    factor = np.ones_like(dtau)
    np.divide(-np.expm1(-dtau), dtau, out=factor, where=dtau > 0)
    return float(np.sum(np.exp(-foreground) * g**4 * j * ds * factor)), float(
        dtau.sum()
    )


@dataclass
class EmittingVolume:
    """sample(metric,r,theta,phi,t) -> (j,alpha,u^mu), vectorized.

    Coefficients outside [r_in,r_out] are zero. The radiation domain must be
    exterior to the absorbing boundary and inside the celestial sphere.
    """

    r_in: float
    r_out: float
    sample: object
    name: str = "gray emitting volume"
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if (
            not np.isfinite([self.r_in, self.r_out]).all()
            or not 0 < self.r_in < self.r_out
        ):
            raise ValueError("require finite 0 < r_in < r_out")
        if not callable(self.sample):
            raise TypeError("sample must be callable")

    def metadata(self):
        return {
            "name": self.name,
            "r_in": self.r_in,
            "r_out": self.r_out,
            "radiation": "gray bolometric emission and absorption",
            "provenance": self.provenance,
        }


class VolumeGrid(EmittingVolume):
    """Static 3D snapshot of gray radiation coefficients and coordinate u^mu.

    Arrays j,alpha have shape (nr,ntheta,nphi); velocity adds a final axis of
    length 4. Axes are increasing, theta includes 0 and pi, phi is uniform
    [0,2pi) without a duplicate endpoint. Interpolated u is renormalized.
    This does not require matter to be axisymmetric, only the metric.
    """

    def __init__(
        self,
        radius,
        theta,
        phi,
        emissivity,
        absorption,
        four_velocity,
        *,
        name="volume snapshot",
    ):
        self.axes = [np.array(a, float, copy=True) for a in (radius, theta, phi)]
        shape = tuple(len(a) for a in self.axes)
        if any(
            a.ndim != 1
            or len(a) < 2
            or not np.isfinite(a).all()
            or np.any(np.diff(a) <= 0)
            for a in self.axes
        ):
            raise ValueError("snapshot axes must be finite and increasing")
        if not np.allclose(self.axes[1][[0, -1]], [0, np.pi]) or not np.allclose(
            self.axes[2], np.linspace(0, 2 * np.pi, shape[2], endpoint=False)
        ):
            raise ValueError(
                "require full theta [0,pi] and periodic uniform phi [0,2pi)"
            )
        j, a, u = [
            np.array(v, float, copy=True)
            for v in (emissivity, absorption, four_velocity)
        ]
        if j.shape != shape or a.shape != shape or u.shape != shape + (4,):
            raise ValueError(
                "coefficient shapes must match axes; velocity has four components"
            )
        if (
            not all(np.isfinite(v).all() for v in (j, a, u))
            or np.any(j < 0)
            or np.any(a < 0)
            or np.any(u[..., 0] <= 0)
        ):
            raise ValueError(
                "invalid radiation coefficients or future-directed velocity"
            )
        self.values = np.concatenate((j[..., None], a[..., None], u), axis=-1)
        for v in self.axes + [self.values]:
            v.setflags(write=False)
        digest = hashlib.sha256(
            b"".join(v.tobytes() for v in self.axes + [self.values])
        ).hexdigest()
        super().__init__(
            float(self.axes[0][0]),
            float(self.axes[0][-1]),
            self._sample,
            name,
            {"sha256": digest, "shape": list(shape), "interpolation": "trilinear"},
        )

    def _sample(self, st, r, th, ph, t):
        th = np.arccos(np.clip(np.cos(th), -1, 1))
        ph = np.mod(ph, 2 * np.pi)
        coords = [r, th, ph]
        indices = []
        fractions = []
        for k, (axis, x) in enumerate(zip(self.axes, coords)):
            if k == 2:
                s = x / (2 * np.pi) * len(axis)
                i = np.floor(s).astype(int)
                f = s - i
            else:
                i = np.clip(np.searchsorted(axis, x) - 1, 0, len(axis) - 2)
                f = (x - axis[i]) / (axis[i + 1] - axis[i])
            indices.append(i)
            fractions.append(f)
        out = np.zeros(np.shape(r) + (6,))
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    weight = np.ones_like(r)
                    for bit, f in zip((a, b, c), fractions):
                        weight *= f if bit else 1 - f
                    out += (
                        weight[..., None]
                        * self.values[
                            indices[0] + a,
                            indices[1] + b,
                            (indices[2] + c) % len(self.axes[2]),
                        ]
                    )
        u = out[..., 2:]
        gd = metric_samples(st, r, th)
        norm = (
            gd[..., 0] * u[..., 0] ** 2
            + 2 * gd[..., 1] * u[..., 0] * u[..., 3]
            + np.sum(gd[..., 2:] * u[..., 1:] ** 2, axis=-1)
        )
        if np.any(norm >= 0):
            raise ValueError("interpolated snapshot velocity is not timelike")
        return out[..., 0], out[..., 1], u / np.sqrt(-norm)[..., None]

    def save(self, path):
        np.savez_compressed(
            path,
            radius=self.axes[0],
            theta=self.axes[1],
            phi=self.axes[2],
            emissivity=self.values[..., 0],
            absorption=self.values[..., 1],
            four_velocity=self.values[..., 2:],
            name=self.name,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            return cls(
                *(
                    d[k]
                    for k in (
                        "radius",
                        "theta",
                        "phi",
                        "emissivity",
                        "absorption",
                        "four_velocity",
                    )
                ),
                name=str(d["name"]),
            )


@metric_context
def render_volume(
    spacetime,
    camera,
    volume,
    *,
    sky=None,
    exposure=1.0,
    max_step=0.25,
    escape_radius=None,
    rtol=1e-8,
    atol=1e-10,
    max_steps=100_000,
):
    """Reference gray transfer; one sampled Fortran trajectory per pixel.

    Refine max_step independently of the geodesic tolerance and snapshot
    resolution. This path is deliberately CPU-only and slower than surfaces.
    max_step is an affine-parameter step, with camera photon energy=1.
    """
    from .api import camera_ray, trace

    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be positive and finite")
    escape = escape_radius or 1.1 * camera.r
    if not spacetime.capture_radius < volume.r_in < volume.r_out < escape:
        raise ValueError("volume must be between absorbing boundary and escape sphere")
    rays = trace_bundle(
        spacetime,
        camera,
        escape_radius=escape,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    intensity = np.zeros(camera.resolution)
    tau = np.zeros(camera.resolution)
    nx, ny = camera.resolution
    for i in range(nx):
        x = camera.x[0] + (i + 0.5) * (camera.x[1] - camera.x[0]) / nx
        for k in range(ny):
            yy = camera.y[0] + (k + 0.5) * (camera.y[1] - camera.y[0]) / ny
            y0, pt, pp = camera_ray(spacetime, camera, x, yy)
            tr = trace(
                spacetime,
                y0,
                pt,
                pp,
                h0=-min(1.0, max_step),
                lambda_max=10 * escape,
                n_max=max_steps,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
                step_radius=2 * volume.r_out,
                escape_radius=escape,
            )
            end_radius = spacetime.capture_radius + (
                0.01 if spacetime.mid in (1, 2) else 0.0
            )
            if not len(tr.r) or not (tr.r[-1] >= escape or tr.r[-1] <= end_radius):
                rays.status[i, k] = 3
                continue
            sampled_status = 0 if tr.r[-1] >= escape else 1
            if sampled_status != rays.status[i, k]:
                # Do not combine a background from one path with radiation
                # from a numerically different capture/escape outcome.
                rays.status[i, k] = 3
                continue
            rays.herr[i, k] = max(rays.herr[i, k], float(np.max(tr.herr)))
            states = np.vstack(
                (
                    y0,
                    np.stack(
                        (tr.t, tr.r, tr.theta, tr.phi, tr.p_r, tr.p_theta), axis=-1
                    ),
                )
            )
            mid = (states[:-1] + states[1:]) / 2
            dl = np.abs(np.diff(np.r_[0.0, tr.lam]))
            inside = (mid[:, 1] >= volume.r_in) & (mid[:, 1] <= volume.r_out)
            y = mid[inside]
            if not len(y):
                continue
            r, th, ph = y[:, 1], y[:, 2], y[:, 3]
            j, a, u = volume.sample(spacetime, r, th, ph, -np.abs(y[:, 0]))
            momentum = np.stack(
                (np.full(len(y), pt), y[:, 4], y[:, 5], np.full(len(y), pp)), axis=-1
            )
            g = frequency_shift(spacetime, r, th, momentum, u)
            intensity[i, k], tau[i, k] = integrate_gray(j, a, g, dl[inside] / g)
    rgb = compose_rgb(
        intensity,
        rays.status,
        rays.endpoint[..., 2],
        rays.endpoint[..., 3],
        None,
        exposure=exposure,
    )
    # The map is uncalibrated RGB: its attenuation is illustrative only.
    if sky is not None:
        mask = rays.status == 0
        background = sky.sample(
            rays.endpoint[..., 2][mask], rays.endpoint[..., 3][mask]
        )
        rgb[mask] = np.clip(rgb[mask] + background * np.exp(-tau[mask, None]), 0, 1)
    meta = {
        **rays.meta,
        "volume": metadata(volume),
        "max_affine_step": max_step,
        "exposure": exposure,
        "sky": sky.metadata() if sky else None,
        "radiation": "gray bolometric; midpoint quadrature; no scattering",
    }
    # Optical depth is retained as a distinct map, never overloaded into g.
    return VolumeImage(rays, intensity, np.zeros(camera.resolution), rgb, meta, tau)


@dataclass
class VolumeImage(SceneImage):
    optical_depth: np.ndarray = field(default_factory=lambda: np.array([]))

    def save(self, path):
        super().save(path)
        with np.load(path, allow_pickle=False) as d:
            arrays = {k: d[k] for k in d.files}
        np.savez_compressed(path, **arrays, optical_depth=self.optical_depth)

    @classmethod
    def load(cls, path):
        img = SceneImage.load(path)
        with np.load(path, allow_pickle=False) as d:
            return cls(
                img.rays, img.intensity, img.g, img.rgb, img.meta, d["optical_depth"]
            )
