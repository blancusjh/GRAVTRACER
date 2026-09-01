"""Functional facade over the Fortran core.

The one-shot workflow composes a spacetime, a camera, and optionally
matter, then renders:

    >>> import grayt
    >>> img = grayt.render(grayt.BlackHole(a=0.95),
    ...                    grayt.Camera(resolution=(1024, 512)),
    ...                    grayt.ThinDisk(l0=1.8))
    >>> img.plot()

For multi-instrument setups (screens, photographs, 3D scenes) use
``grayt.System`` (see grayt.system).
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np

from . import _core
from .spacetime import Spacetime, BlackHole, QMetric, MID_KERR
from .instruments import Camera
from .matter import ThinDisk
from .results import Image, Trajectory

_METHODS = {"rkdp45": 1, "rkck45": 2, "rkf45": 3}

STATUS_ESCAPED = 0
STATUS_CAPTURED = 1
STATUS_DISK = 2
STATUS_FAILED = 3


def _method_id(name: str) -> int:
    try:
        return _METHODS[name]
    except KeyError:
        raise ValueError(f"unknown integrator {name!r}; valid methods: "
                         f"{', '.join(sorted(_METHODS))}") from None


def render(spacetime: Spacetime, camera: Camera, disk: ThinDisk | None = None,
           method: str = "rkdp45", rtol: float = 1e-8, atol: float = 1e-10,
           max_steps: int = 500_000) -> Image:
    """Trace every pixel of ``camera`` through ``spacetime``; if ``disk``
    is given, shade disk intersections with I_obs = g^3 * F_PageThorne
    (eq. 19 of arXiv:2202.00086). The disk model is Kerr-only."""
    if disk is not None and spacetime.mid != MID_KERR:
        raise ValueError("the thin-disk model (Page-Thorne, ISCO, redshift) "
                         "is defined for Kerr only; render this spacetime "
                         "without a disk")
    if disk is not None and isinstance(spacetime, BlackHole):
        if disk.r_out <= spacetime.isco:
            warnings.warn(
                f"disk r_out = {disk.r_out} lies inside the ISCO "
                f"({spacetime.isco:.3f}); the disk is empty", stacklevel=2)

    nx, ny = camera.resolution
    disk_on = 1 if disk is not None else 0
    rin = -1.0 if (disk is None or disk.r_in is None) else disk.r_in
    rout = disk.r_out if disk is not None else 20.0
    l0 = disk.l0 if disk is not None else 0.0

    # The Fortran camera x-axis is mirrored w.r.t. the paper's figures;
    # render with x negated so that screen-x follows the paper convention.
    out = _core.raytracer.render_image(
        spacetime.mid, spacetime.par, camera.r, np.deg2rad(camera.theta),
        np.deg2rad(camera.phi), -camera.x[1], -camera.x[0],
        camera.y[0], camera.y[1], nx, ny, _method_id(method), rtol, atol,
        disk_on, rin, rout, l0, max_steps)
    intens, gmap, rhit, status, herrm, thf, phf = (np.flip(m, axis=0)
                                                  for m in out)

    meta = {"spacetime": dataclasses.asdict(spacetime),
            "camera": dataclasses.asdict(camera),
            "disk": dataclasses.asdict(disk) if disk else None,
            "method": method, "rtol": rtol, "atol": atol}
    return Image(intensity=intens, g=gmap, r_hit=rhit,
                 status=status.astype(np.int32), herr=herrm,
                 theta_inf=thf, phi_inf=phf, extent=camera.extent, meta=meta)


def shadow(spacetime: Spacetime, camera: Camera, **kwargs) -> Image:
    """Render without a disk; ``status == STATUS_CAPTURED`` is the shadow."""
    return render(spacetime, camera, disk=None, **kwargs)


def trace(spacetime: Spacetime, y0, p_t: float, p_phi: float,
          method: str = "rkdp45", rtol: float = 1e-10, atol: float = 1e-12,
          h0: float = 1.0, lambda_max: float = 1e4,
          n_max: int = 100_000) -> Trajectory:
    """Integrate a single geodesic (photon or massive particle) from raw
    initial conditions ``y0 = (t, r, theta, phi, p_r, p_theta)``
    (angles in RADIANS here — raw phase-space coordinates)."""
    traj, herr, nout = _core.raytracer.trace_geodesic(
        spacetime.mid, spacetime.par, np.asarray(y0, dtype=float), p_t,
        p_phi, _method_id(method), rtol, atol, h0, lambda_max, n_max)
    t = traj[:nout]
    return Trajectory(lam=t[:, 0], t=t[:, 1], r=t[:, 2], theta=t[:, 3],
                      phi=t[:, 4], p_r=t[:, 5], p_theta=t[:, 6],
                      herr=herr[:nout],
                      meta={"spacetime": dataclasses.asdict(spacetime),
                            "p_t": p_t, "p_phi": p_phi, "method": method})


def camera_ray(spacetime: Spacetime, camera: Camera, x: float, y: float):
    """Initial conditions (y0, p_t, p_phi) for one image-plane point."""
    y0, pt, pphi = _core.raytracer.camera_init(
        spacetime.mid, spacetime.par, camera.r, np.deg2rad(camera.theta),
        np.deg2rad(camera.phi), -x, y)
    return y0, float(pt), float(pphi)


def orbit_ic(spacetime: Spacetime, r0: float, energy: float,
             angular_momentum: float, mass: float = 1.0,
             theta0: float = 90.0, pr_sign: int = 0):
    """Initial conditions for a geodesic with conserved E and L.

    ``mass=1`` gives a time-like orbit (H = -1/2), ``mass=0`` a photon.
    ``theta0`` in degrees. ``pr_sign``: sign of the initial radial
    momentum (0 = turning point). Returns ``(y0, p_t, p_phi)``.
    """
    th = np.deg2rad(theta0)
    p_t, p_phi = -energy, angular_momentum
    gu, _, _ = spacetime.metric_contra(r0, th)
    pr2 = (-mass**2 - (gu[0]*p_t**2 + 2*gu[1]*p_t*p_phi +
                       gu[4]*p_phi**2))/gu[2]
    if pr2 < -1e-12:
        raise ValueError("no orbit with these (E, L) at r0: p_r^2 < 0")
    p_r = pr_sign*np.sqrt(max(pr2, 0.0))
    return np.array([0.0, r0, th, 0.0, p_r, 0.0]), p_t, p_phi


def flux_profile(black_hole: BlackHole, r_out: float = 20.0, n: int = 2000):
    """Radial Page-Thorne flux profile on [r_isco, r_out] (Kerr only)."""
    if black_hole.mid != MID_KERR:
        raise ValueError("the Page-Thorne flux is defined for Kerr only")
    rs, fs = _core.disk_model.flux_profile(black_hole.a, r_out, n)
    return rs, fs
