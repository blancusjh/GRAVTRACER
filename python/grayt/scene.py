"""Geometry-only ray bundles followed by optional surface radiation transfer."""

from dataclasses import dataclass, field
import json
import numpy as np

from . import _core
from ._runtime import metric_context, metadata
from .emission import (EmittingDisk, PageThorneDisk, SlabDisk, metric_samples,
                       frequency_shift)
from .sky import compose_rgb


@dataclass
class RayBundle:
    """Raw endpoints (nx,ny,6), conserved momenta (nx,ny,2), and diagnostics.

    Coordinates use the legacy camera convention; elapsed coordinate light
    travel time is abs(endpoint[...,0]). All other axes follow Image.
    """

    endpoint: np.ndarray
    momenta: np.ndarray
    status: np.ndarray
    herr: np.ndarray
    extent: tuple
    meta: dict = field(default_factory=dict)

    @property
    def travel_time(self):
        return np.abs(self.endpoint[..., 0])


@dataclass
class SceneImage:
    """Bolometric maps and an explicitly display-only RGB composite."""

    rays: RayBundle
    intensity: np.ndarray
    g: np.ndarray
    rgb: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def status(self):
        return self.rays.status

    @property
    def r_hit(self):
        return self.rays.endpoint[..., 1]

    @property
    def herr(self):
        return self.rays.herr

    @property
    def theta_inf(self):
        """Asymptotic direction (polar angle) of escaped rays."""
        return self._directions()[0]

    @property
    def phi_inf(self):
        """Asymptotic direction (azimuth) of escaped rays."""
        return self._directions()[1]

    def _directions(self):
        # Set by the renderers (and by load); archives written before
        # directions were stored fall back to the escape-sphere position.
        dirs = getattr(self, "_dirs", None)
        if dirs is None:
            return self.rays.endpoint[..., 2], self.rays.endpoint[..., 3]
        return dirs

    @property
    def extent(self):
        return self.rays.extent

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(self.rgb.transpose(1, 0, 2), origin="lower", extent=self.extent)
        ax.set(xlabel="image-plane x / M", ylabel="image-plane y / M")
        return ax

    def save(self, path):
        """Portable NPZ, JSON provenance, no pickle or serialized callbacks."""
        np.savez_compressed(
            path,
            endpoint=self.rays.endpoint,
            momenta=self.rays.momenta,
            status=self.status,
            herr=self.rays.herr,
            extent=self.extent,
            intensity=self.intensity,
            g=self.g,
            rgb=self.rgb,
            theta_dir=self.theta_inf,
            phi_dir=self.phi_inf,
            metadata=json.dumps(self.meta),
            ray_metadata=json.dumps(self.rays.meta),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            rays = RayBundle(
                d["endpoint"],
                d["momenta"],
                d["status"],
                d["herr"],
                tuple(d["extent"]),
                json.loads(str(d["ray_metadata"])),
            )
            image = cls(
                rays, d["intensity"], d["g"], d["rgb"], json.loads(str(d["metadata"]))
            )
            if "theta_dir" in d.files:
                image._dirs = (d["theta_dir"], d["phi_dir"])
            return image


def _metric_on_sphere(spacetime, r, theta):
    """Covariant metric at points that share (almost) one radius.

    Kerr is evaluated in closed form. Other metrics are sampled on a fine
    theta grid at that radius and interpolated, which avoids a Python loop
    over every pixel; all escape points lie on the same sphere.
    """
    if spacetime.mid == 1 or r.size == 0:
        return metric_samples(spacetime, r, theta)
    radius = float(np.median(r))
    if np.max(np.abs(r - radius)) > 1e-6 * radius:
        return metric_samples(spacetime, r, theta)
    grid = np.linspace(0.0, np.pi, 4097)
    table = np.array([spacetime.metric_cov(radius, float(t)) for t in grid])
    th = np.arccos(np.clip(np.cos(theta), -1, 1))
    return np.stack([np.interp(th, grid, table[:, k]) for k in range(5)], -1)


def escape_directions(rays, spacetime):
    """Asymptotic propagation direction (theta, phi) of escaped rays.

    The tracer stops on a finite escape sphere. Sampling a sky map at the
    stopping *position* adds a parallax error of order camera.r / r_escape
    (several degrees for r_escape = 2 camera.r). Instead, take the photon's
    spatial momentum there, map it to the pseudo-Cartesian embedding, and
    use its direction: that is where the ray points at infinity, up to the
    small residual deflection O(M / r_escape) beyond the sphere.
    Non-escaped pixels keep their endpoint angles.
    """
    ep, mom = rays.endpoint, rays.momenta
    theta = ep[..., 2].copy()
    phi = ep[..., 3].copy()
    esc = rays.status == 0
    if not esc.any():
        return theta, phi
    r, th, ph, pr, pth = (ep[..., k][esc] for k in (1, 2, 3, 4, 5))
    pt, pph = mom[..., 0][esc], mom[..., 1][esc]
    gd = _metric_on_sphere(spacetime, r, th)
    gtt, gtp, grr, gthth, gpp = (gd[..., k] for k in range(5))
    det = gtt * gpp - gtp**2
    ur, uth = pr / grr, pth / gthth
    uph = (gtt * pph - gtp * pt) / det
    s, c = np.sin(th), np.cos(th)
    cp, sp = np.cos(ph), np.sin(ph)
    v = np.stack((s * cp * ur + r * c * cp * uth - r * s * sp * uph,
                  s * sp * ur + r * c * sp * uth + r * s * cp * uph,
                  c * ur - r * s * uth), axis=-1)
    outward = np.sum(v * np.stack((s * cp, s * sp, c), axis=-1), axis=-1)
    v *= np.where(outward < 0, -1.0, 1.0)[:, None]   # covector sign convention
    v /= np.linalg.norm(v, axis=-1)[:, None]
    theta[esc] = np.arccos(np.clip(v[:, 2], -1, 1))
    phi[esc] = np.arctan2(v[:, 1], v[:, 0])
    return theta, phi


@metric_context
def trace_bundle(
    spacetime,
    camera,
    disk=None,
    *,
    escape_radius=None,
    method="rkdp45",
    rtol=1e-8,
    atol=1e-10,
    max_steps=500_000,
    constraint_monitor=True,
):
    """Trace vacuum null rays to an annulus, inner boundary or finite sphere.

    The camera is a ZAMO orthonormal observer with photon energy=1. As in
    Camera, screen coordinates are direction cosines multiplied by camera.r.
    """
    from .api import _method_id

    escape = 1.1 * camera.r if escape_radius is None else float(escape_radius)
    if (
        not np.isfinite([rtol, atol, escape]).all()
        or min(rtol, atol) <= 0
        or max_steps < 1
    ):
        raise ValueError(
            "positive finite tolerances, escape radius and step budget required"
        )
    if not spacetime.capture_radius < camera.r < escape:
        raise ValueError("require inner boundary < camera radius < escape radius")
    if not 0 < camera.theta < 180:
        raise ValueError("camera must be off the spherical coordinate poles")
    if (
        hasattr(spacetime, "radial_domain")
        and escape >= spacetime.radial_domain[1] * 0.95
    ):
        raise ValueError("escape sphere needs >=5% radial table margin for RK stages")
    if disk is not None and not isinstance(disk, EmittingDisk):
        raise TypeError("trace_bundle disk must be an EmittingDisk")
    if disk is not None and disk.r_in <= spacetime.capture_radius:
        raise ValueError("disk inner edge must lie outside the absorbing boundary")
    if disk is not None and disk.r_out >= escape:
        raise ValueError("disk must lie inside the escape sphere")
    nx, ny = camera.resolution
    xs = camera.x[0] + (np.arange(nx) + 0.5) * (camera.x[1] - camera.x[0]) / nx
    ys = camera.y[0] + (np.arange(ny) + 0.5) * (camera.y[1] - camera.y[0]) / ny
    if max(abs(xs)) ** 2 + max(abs(ys)) ** 2 >= camera.r**2:
        raise ValueError("image-plane direction cosines exceed the camera hemisphere")
    ep, mom, status, herr = _core.raytracer.render_geometry(
        spacetime.mid,
        spacetime.par,
        camera.r,
        np.deg2rad(camera.theta),
        np.deg2rad(camera.phi),
        xs,
        ys,
        _method_id(method),
        rtol,
        atol,
        escape,
        int(disk is not None),
        disk.r_in if disk else 0.0,
        disk.r_out if disk else 0.0,
        max_steps,
        errmon=int(constraint_monitor),
    )
    return RayBundle(
        ep.transpose(1, 2, 0).copy(),
        mom.transpose(1, 2, 0).copy(),
        status,
        herr,
        camera.extent,
        {
            "spacetime": metadata(spacetime),
            "camera": metadata(camera),
            "escape_radius": escape,
            "method": method,
            "rtol": rtol,
            "atol": atol,
            "max_steps": max_steps,
            "constraint_monitor": bool(constraint_monitor),
            "backend": "Fortran CPU fp64",
            "coordinates": "t,r,theta,phi; G=c=M=1; angles radians",
            "momentum_convention": "legacy past-oriented covector; E_obs=1",
            "sky_directions": "asymptotic momentum direction at escape",
        },
    )


@metric_context
def trace_crossings(
    spacetime,
    camera,
    slab,
    *,
    escape_radius=None,
    method="rkdp45",
    rtol=1e-8,
    atol=1e-10,
    max_steps=500_000,
):
    """Trace rays through a see-through equatorial annulus.

    Returns (RayBundle of final escape/capture states, crossing count
    (nx,ny), crossing states (nx,ny,max_crossings,6)), crossings ordered
    from the camera outward.
    """
    from .api import _method_id

    escape = 1.1 * camera.r if escape_radius is None else float(escape_radius)
    if not spacetime.capture_radius < slab.r_in < slab.r_out < escape:
        raise ValueError("slab must lie between the inner boundary and escape sphere")
    if not spacetime.capture_radius < camera.r < escape:
        raise ValueError("require inner boundary < camera radius < escape radius")
    nx, ny = camera.resolution
    xs = camera.x[0] + (np.arange(nx) + 0.5) * (camera.x[1] - camera.x[0]) / nx
    ys = camera.y[0] + (np.arange(ny) + 0.5) * (camera.y[1] - camera.y[0]) / ny
    if max(abs(xs)) ** 2 + max(abs(ys)) ** 2 >= camera.r**2:
        raise ValueError("image-plane direction cosines exceed the camera hemisphere")
    ncmax = int(slab.max_crossings)
    ep, mom, status, herr, ncross, cross = _core.raytracer.render_crossings(
        spacetime.mid, spacetime.par, camera.r, np.deg2rad(camera.theta),
        np.deg2rad(camera.phi), xs, ys, _method_id(method), rtol, atol,
        escape, slab.r_in, slab.r_out, max_steps, ncmax)
    rays = RayBundle(
        ep.transpose(1, 2, 0).copy(),
        mom.transpose(1, 2, 0).copy(),
        status,
        herr,
        camera.extent,
        {
            "spacetime": metadata(spacetime),
            "camera": metadata(camera),
            "escape_radius": escape,
            "method": method,
            "rtol": rtol,
            "atol": atol,
            "max_steps": max_steps,
            "backend": "Fortran CPU fp64",
            "coordinates": "t,r,theta,phi; G=c=M=1; angles radians",
            "momentum_convention": "legacy past-oriented covector; E_obs=1",
            "slab_crossings": ncmax,
            "sky_directions": "asymptotic momentum direction at escape",
        },
    )
    return rays, ncross, cross.transpose(2, 3, 1, 0).copy()


def _slab_transfer(spacetime, slab, rays, ncross, cross, observer_time,
                   layers=None):
    """Observed intensity, first-crossing g, and optical depth per pixel.

    If ``layers`` is a list, (mask, g, S, weight) is appended per crossing,
    where weight = exp(-tau in front) (1 - exp(-tau)); photometry uses it.
    """
    shape = rays.status.shape
    intensity = np.zeros(shape)
    tau_total = np.zeros(shape)
    gmap = np.zeros(shape)
    for k in range(cross.shape[2]):
        mask = ncross > k
        if not mask.any():
            break
        y, constants = cross[mask, k], rays.momenta[mask]
        r, th, ph = y[:, 1], y[:, 2], y[:, 3]
        p = np.stack((constants[:, 0], y[:, 4], y[:, 5], constants[:, 1]), axis=-1)
        u = slab.disk.four_velocity(spacetime, r, th, ph)
        g = frequency_shift(spacetime, r, th, p, u)
        # |cos eta| between photon and slab normal in the comoving frame:
        # u has no theta component, so p_(theta-hat) = p_theta/sqrt(g_thth).
        g_thth = metric_samples(spacetime, r, th)[:, 3]
        cos_eta = np.clip(np.abs(y[:, 5]) * g / np.sqrt(g_thth), 1e-6, 1.0)
        tau = slab.tau_perp(r, ph) / cos_eta
        t = observer_time - np.abs(y[:, 0])
        source = np.broadcast_to(slab.source_function(r, ph, t), r.shape)
        if np.any(~np.isfinite(source)) or np.any(source < 0):
            raise ValueError("emitted intensity must be finite and nonnegative")
        foreground = tau_total[mask]
        weight = np.exp(-foreground) * -np.expm1(-tau)
        intensity[mask] += weight * g**4 * source
        if layers is not None:
            layers.append((mask, g, np.array(source), weight))
        tau_total[mask] = foreground + tau
        if k == 0:
            gmap[mask] = g
    return intensity, gmap, tau_total


@metric_context
def render_scene(
    spacetime,
    camera,
    disk=None,
    *,
    surface=None,
    sky=None,
    exposure=1.0,
    observer_time=0.0,
    tone="exp",
    decades=2.5,
    photometry=None,
    levels=None,
    **trace_options,
):
    """Vacuum propagation + bolometric surfaces + celestial RGB map.

    Opaque disks stop rays; a ``SlabDisk`` is see-through with gray
    transfer at every crossing. ``tone``/``decades`` choose the display
    curve (see ``compose_rgb``); raw intensities are unaffected.

    With ``photometry`` (a ``grayt.photometry.Photometry``) the RGB is a
    calibrated rendering instead: blackbody emission at g T in physical
    units plus the calibrated sky, one global tone curve (``levels``, see
    ``photometric_render``; the levels used are stored in ``meta``).
    Background RGB is an uncalibrated map on a finite coordinate sphere,
    not an infinite-distance direction map or a redshifted stellar spectrum.
    """
    from .matter import ThinDisk

    if isinstance(disk, SlabDisk):
        image = _render_slab(spacetime, camera, disk, surface, sky, exposure,
                             observer_time, tone, decades, trace_options)
        return _photometric(image, photometry, sky, levels)
    legacy = isinstance(disk, ThinDisk)
    source_metadata = metadata(disk) if disk else None
    if legacy:
        from .api import flux_profile
        from .emission import rotating_velocity

        if spacetime.mid != 1:
            raise ValueError("legacy ThinDisk requires Kerr")
        old = disk
        rs, fs = flux_profile(spacetime, old.r_out)

        def velocity(st, r, th, ph):
            gd = metric_samples(st, r, th)
            omega = -(gd[..., 1] + old.l0 * gd[..., 0]) / (
                gd[..., 4] + old.l0 * gd[..., 1]
            )
            return rotating_velocity(st, r, th, omega)

        disk = EmittingDisk(
            spacetime.isco if old.r_in is None else old.r_in,
            old.r_out,
            lambda r, ph, t: np.interp(r, rs, fs),
            velocity,
        )
    if (
        isinstance(disk, PageThorneDisk)
        and metadata(spacetime) != disk.spacetime_metadata
    ):
        raise ValueError("PageThorneDisk was built for a different Kerr spacetime")
    if surface is not None:
        if getattr(spacetime, "boundary_kind", None) != "surface":
            raise ValueError("emission requires a metric with boundary_kind=surface")
    rays = trace_bundle(spacetime, camera, disk, **trace_options)
    intensity = np.zeros(camera.resolution)
    gmap = np.zeros(camera.resolution)
    layers = []
    for source, code in ((disk, 2), (surface, 1)):
        mask = rays.status == code
        if source is None or not mask.any():
            continue
        y, constants = rays.endpoint[mask], rays.momenta[mask]
        r, th, ph = y[:, 1], y[:, 2], y[:, 3]
        p = np.stack((constants[:, 0], y[:, 4], y[:, 5], constants[:, 1]), axis=-1)
        u = source.four_velocity(spacetime, r, th, ph)
        g = frequency_shift(spacetime, r, th, p, u)
        t = observer_time - np.abs(y[:, 0])
        emitted = np.broadcast_to(
            np.asarray(source.intensity(r if code == 2 else th, ph, t), float), r.shape
        )
        if np.any(~np.isfinite(emitted)) or np.any(emitted < 0):
            raise ValueError("emitted intensity must be finite and nonnegative")
        gmap[mask] = g
        intensity[mask] = g ** (3 if legacy and code == 2 else 4) * emitted
        if not (legacy and code == 2):
            layers.append((mask, g, np.array(emitted), np.ones_like(g)))
    directions = escape_directions(rays, spacetime)
    rgb = compose_rgb(
        intensity,
        rays.status,
        *directions,
        sky,
        exposure=exposure,
        tone=tone,
        decades=decades,
    )
    meta = {
        **rays.meta,
        "disk": source_metadata,
        "surface": metadata(surface) if surface else None,
        "sky": sky.metadata() if sky else None,
        "observer_time": observer_time,
        "display": {
            "exposure": exposure,
            "cmap": "afmhot",
            "transfer": _transfer(tone, decades),
        },
        "radiation": "legacy disk g^3"
        if legacy
        else "bolometric g^4; no volume transfer",
    }
    image = SceneImage(rays, intensity, gmap, rgb, meta)
    image._dirs = directions
    image.layers = layers
    return _photometric(image, photometry, sky, levels)


def _photometric(image, photometry, sky, levels):
    if photometry is None:
        return image
    from .photometry import photometric_render

    keys = ("floor", "knee", "white", "disk_decades")
    fixed = {k: v for k, v in (levels or {}).items() if k in keys}
    _, image.rgb, used = photometric_render(image, photometry, sky, **fixed)
    image.meta["display"] = {"photometry": photometry.metadata(),
                             "tone": "log10(Y) piecewise linear", **used}
    return image


def _transfer(tone, decades):
    if tone == "exp":
        return "1-exp(-exposure*I)"
    return f"log10(exposure*I) over {decades:g} decades"


def _render_slab(spacetime, camera, slab, surface, sky, exposure, observer_time,
                 tone, decades, trace_options):
    from .volume import VolumeImage

    if surface is not None:
        raise ValueError("SlabDisk scenes do not combine with emitting surfaces")
    if (
        isinstance(slab.disk, PageThorneDisk)
        and metadata(spacetime) != slab.disk.spacetime_metadata
    ):
        raise ValueError("PageThorneDisk was built for a different Kerr spacetime")
    trace_options = {k: v for k, v in trace_options.items()
                     if k != "constraint_monitor"}
    rays, ncross, cross = trace_crossings(spacetime, camera, slab, **trace_options)
    layers = []
    intensity, gmap, tau = _slab_transfer(spacetime, slab, rays, ncross, cross,
                                          observer_time, layers)
    directions = escape_directions(rays, spacetime)
    rgb = compose_rgb(
        intensity,
        rays.status,
        *directions,
        sky,
        exposure=exposure,
        transmission=np.exp(-tau),
        tone=tone,
        decades=decades,
    )
    meta = {
        **rays.meta,
        "disk": slab.metadata(),
        "surface": None,
        "sky": sky.metadata() if sky else None,
        "observer_time": observer_time,
        "display": {
            "exposure": exposure,
            "cmap": "afmhot",
            "transfer": _transfer(tone, decades) + ", added to exp(-tau) * sky",
        },
        "radiation": "bolometric g^4; gray thin slab at every crossing",
        "crossing_counts": np.bincount(ncross.ravel(),
                                       minlength=slab.max_crossings + 1).tolist(),
    }
    image = VolumeImage(rays, intensity, gmap, rgb, meta, tau)
    image._dirs = directions
    image.layers = layers
    image.transmission = np.exp(-tau)
    return image
