"""Prescribed turbulent structure for thin-disk figures and movies.

Magnetized accretion disks are clumpy: knots of enhanced dissipation form
and are sheared into trailing arcs by differential rotation. Here N knots
with log-uniform radii are advected rigidly with the local Keplerian
angular velocity, Omega(r) = 1 / (r^1.5 + a), evaluated at every point:
a knot is therefore stretched into an arc as it ages (``age`` sets how
long the pattern has already been sheared at t = 0). The dissipated flux
is the smooth disk's flux times exp(A (p - <p>)), so the pattern is a
log-normal modulation with a fixed mean level.

This is a prescribed brightness pattern with consistent kinematics, not
an MHD simulation.
"""
from __future__ import annotations

import numpy as np

import grayt


def turbulent_disk(bh, r_c=5.0, n_knots=160, amplitude=0.9, age=240.0,
                   seed=11, tau_c=1e4, r_trunc=None):
    """Spreading thin disk (grayt.spreading_disk) with sheared knots."""
    base = grayt.PageThorneDisk(bh, r_out=15 * r_c)
    rng = np.random.default_rng(seed)
    radii = np.exp(rng.uniform(np.log(bh.isco + 0.4), np.log(3.5 * r_c),
                               n_knots))
    phases = rng.uniform(0, 2 * np.pi, n_knots)
    widths = rng.uniform(0.04, 0.12, n_knots) * radii
    kappa = rng.uniform(6.0, 40.0, n_knots)        # azimuthal concentration
    weights = rng.lognormal(0.0, 0.5, n_knots)
    # mean of the von Mises bump exp(kappa (cos - 1)) over phi
    mean_bump = np.i0(kappa) * np.exp(-kappa)

    def intensity(r, phi, t):
        r = np.asarray(r, float)
        omega = 1.0 / (r**1.5 + bh.a)
        drift = omega * (t + age)
        p = np.zeros(r.shape)
        mean = np.zeros(r.shape)
        for rk, pk, wk, kk, ak, mb in zip(radii, phases, widths, kappa,
                                          weights, mean_bump):
            radial = ak * np.exp(-0.5 * ((r - rk) / wk) ** 2)
            p += radial * np.exp(kk * (np.cos(phi - pk - drift) - 1.0))
            mean += radial * mb
        return base.intensity(r, phi, t) * np.exp(amplitude * (p - mean))

    knots = grayt.EmittingDisk(
        base.r_in, base.r_out, intensity,
        name="Page-Thorne x sheared knots (prescribed, not MHD)",
        provenance={"knots": n_knots, "seed": seed, "age_M": age,
                    "amplitude": amplitude})
    return grayt.spreading_disk(bh, r_c=r_c, tau_c=tau_c, base=knots,
                                r_trunc=r_trunc)
