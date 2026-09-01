"""2D projections of ray and particle orbits (paper Fig. 3 / Fig. 14
style): trajectories in the equatorial plane with the horizon drawn at
the origin.

Panels: (1) photon escape/fall orbits around a fast-spinning hole,
(2) bounded time-like orbits (precessing rosettes), (3) unbounded
time-like orbits started at a turning point.

Usage: python examples/orbits2d.py
"""
import argparse

from _common import out

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grayt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=out("orbits2d.png"))
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2),
                             constrained_layout=True)

    # --- photons around a = 0.98 (initial conditions of Figs. 3-5)
    bh = grayt.BlackHole(a=0.98)
    esc = grayt.trace(bh, [0, 100, 1.570796, 1.570796, -1.009327, 1.87],
                      p_t=-0.989953, p_phi=2.000098, lambda_max=220.0)
    fall = grayt.trace(bh, [0, 100, 1.570796, 1.570796, -1.009314, 3.75],
                       p_t=-0.989952, p_phi=1.250061, lambda_max=140.0)
    grayt.plot_orbits_2d([(esc, "escape"), (fall, "fall")],
                         black_hole=bh, ax=axes[0],
                         colors=["tab:blue", "tab:red"])
    axes[0].set_xlim(-25, 25); axes[0].set_ylim(-25, 25)
    axes[0].set_title("photons, $a = 0.98$")

    # --- bounded time-like orbits: precessing rosettes (Schwarzschild)
    bh0 = grayt.BlackHole(a=0.0)
    orbits = []
    for r0, L, color in [(25.0, 4.2, None), (20.0, 3.9, None)]:
        gu = None
        import numpy as np
        E = np.sqrt((1 - 2/r0)*(1 + L*L/(r0*r0)))  # turning point at r0
        y0, pt, pphi = grayt.orbit_ic(bh0, r0, E, L)
        orb = grayt.trace(bh0, y0, p_t=pt, p_phi=pphi, lambda_max=4000.0,
                          n_max=100_000)
        orbits.append((orb, f"$L = {L}$, $E = {E:.3f}$"))
    grayt.plot_orbits_2d(orbits, black_hole=bh0, ax=axes[1])
    axes[1].set_title("bounded time-like orbits, $a = 0$")

    # --- unbounded time-like orbits started with dr/dlambda = 0 just
    #     outside the unstable circular radius: the particle lingers,
    #     winds around the hole, and spirals out to infinity.
    import numpy as np
    orbits = []
    for L, eps in [(4.4, 0.005), (4.4, 0.08)]:
        r_uns = (L*L - np.sqrt(L**4 - 12*L*L))/2.0
        r0 = r_uns*(1.0 + eps)
        E = np.sqrt((1 - 2/r0)*(1 + L*L/(r0*r0)))
        y0, pt, pphi = grayt.orbit_ic(bh0, r0, E, L)
        orb = grayt.trace(bh0, y0, p_t=pt, p_phi=pphi, lambda_max=1500.0,
                          n_max=100_000)
        orbits.append((orb, f"$r_0 = {r0:.2f}$, $E = {E:.4f}$"))
    grayt.plot_orbits_2d(orbits, black_hole=bh0, ax=axes[2])
    axes[2].set_xlim(-60, 60); axes[2].set_ylim(-60, 60)
    axes[2].set_title("unbounded time-like orbits ($\\dot r = 0$), $a = 0$")

    fig.savefig(args.output, dpi=170, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
