"""The q-metric (Zipoy-Voorhees) spacetime: shadows and time-like
orbits as a function of the quadrupole q (Appendix A / Fig. 14 of
arXiv:2202.00086; shadows as in Arrieta-Villamizar et al. 2020).

q = 0 is Schwarzschild; q != 0 sources are naked singularities for a
range of q, with ADM mass m(1+q).

Usage: python examples/qmetric.py
"""
import argparse

import numpy as np

from _common import out

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grayt


def turning_energy(spacetime, r0, L):
    """E such that (r0, L) is a radial turning point of a unit-mass
    orbit: g^tt E^2 + g^pp L^2 = -1."""
    gu, _, _ = spacetime.metric_contra(r0, np.pi/2)
    return np.sqrt(-(1.0 + gu[4]*L*L)/gu[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=300)
    ap.add_argument("-o", "--output", default=out("qmetric.png"))
    args = ap.parse_args()

    fig = plt.figure(figsize=(16, 4.8), constrained_layout=True)

    # --- shadows for different quadrupoles (equatorial observer)
    ax = fig.add_subplot(1, 3, 1)
    cam = grayt.Camera(r=1000.0, theta=90.0, x=(-13, 13), y=(-13, 13),
                       resolution=(args.res, args.res))
    for q, color in [(-0.4, "tab:blue"), (0.0, "black"), (1.0, "tab:red")]:
        img = grayt.shadow(grayt.QMetric(q=q), cam, rtol=1e-9, atol=1e-11)
        mask = (img.status == grayt.STATUS_CAPTURED).T.astype(float)
        xs = np.linspace(*img.extent[:2], args.res)
        ys = np.linspace(*img.extent[2:], args.res)
        ax.contour(xs, ys, mask, levels=[0.5], colors=[color],
                   linewidths=1.6)
        ax.plot([], [], color=color, label=f"$q = {q:g}$")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("shadow rim vs quadrupole (ADM mass $= 1+q$)")

    # --- bounded time-like orbits (Fig. 14, second row)
    ax = fig.add_subplot(1, 3, 2)
    orbits = []
    for q, color in [(0.0, "tab:blue"), (-0.022, "tab:green"),
                     (0.022, "tab:orange")]:
        st = grayt.QMetric(q=q)
        r0, L = 25.0, 4.2
        E = turning_energy(st, r0, L)
        y0, pt, pphi = grayt.orbit_ic(st, r0, E, L)
        orb = grayt.trace(st, y0, p_t=pt, p_phi=pphi, lambda_max=4000.0,
                          n_max=100_000)
        orbits.append((orb, f"$q = {q:g}$"))
    grayt.plot_orbits_2d(orbits, black_hole=grayt.QMetric(0.0), ax=ax,
                         colors=["tab:blue", "tab:green", "tab:orange"])
    ax.set_title("bounded orbits, $r_0 = 25$, $L = 4.2$")

    # --- unbounded time-like orbits started at a turning point
    #     (Fig. 14, first row): the quadrupole reshapes the loops.
    ax = fig.add_subplot(1, 3, 3)
    orbits = []
    for q, color in [(0.0, "tab:blue"), (1.0, "tab:red"),
                     (-0.5, "tab:green")]:
        st = grayt.QMetric(q=q)
        r0, L = 8.0, 6.0
        E = turning_energy(st, r0, L)
        y0, pt, pphi = grayt.orbit_ic(st, r0, E, L)
        orb = grayt.trace(st, y0, p_t=pt, p_phi=pphi, lambda_max=1200.0,
                          n_max=100_000)
        orbits.append((orb, f"$q = {q:g}$, $E = {E:.3f}$"))
    grayt.plot_orbits_2d(orbits, black_hole=grayt.QMetric(0.0), ax=ax,
                         colors=["tab:blue", "tab:red", "tab:green"])
    ax.set_xlim(-45, 45); ax.set_ylim(-45, 45)
    ax.set_title("orbits with $\\dot r = 0$ at $r_0 = 8$, $L = 6$")

    fig.savefig(args.output, dpi=170, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
