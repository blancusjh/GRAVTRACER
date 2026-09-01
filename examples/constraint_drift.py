"""Hamiltonian constraint drift along single photon orbits, for the
available embedded RK integrators.

With the default initial conditions this reproduces the analysis of
Figs. 4-5 of arXiv:2202.00086 (escaping and falling photons around a
Kerr black hole with a = 0.98).

Usage: python examples/constraint_drift.py [-a SPIN]
"""
import argparse

import numpy as np

from _common import out

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grayt

ORBITS = {
    "escape": dict(y0=[0.0, 100.0, 1.570796, 1.570796, -1.009327, 1.87],
                   p_t=-0.989953, p_phi=2.000098, lam=150.0),
    "fall": dict(y0=[0.0, 100.0, 1.570796, 1.570796, -1.009314, 3.75],
                 p_t=-0.989952, p_phi=1.250061, lam=110.0),
}
METHODS = ["rkdp45", "rkck45", "rkf45"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--spin", type=float, default=0.98)
    ap.add_argument("-o", "--output", default=out("constraint_drift.png"))
    args = ap.parse_args()

    bh = grayt.BlackHole(a=args.spin)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, (name, orb) in zip(axes, ORBITS.items()):
        for method in METHODS:
            res = grayt.trace(bh, orb["y0"], p_t=orb["p_t"],
                              p_phi=orb["p_phi"], method=method,
                              rtol=1e-11, atol=1e-13,
                              lambda_max=orb["lam"])
            drift = np.abs(res["herr"] - res["herr"][0]) + 1e-18
            ax.semilogy(np.abs(res["lambda"]), drift, lw=0.9, label=method)
            print(f"{name} / {method}: steps={len(drift)}, "
                  f"max drift |H - H0| = {drift.max():.2e}, "
                  f"final r = {res['r'][-1]:.2f}")
        ax.set_title(f"{name} orbit, $a = {args.spin}$")
        ax.set_xlabel("$\\lambda$")
        ax.set_ylabel("$|H - H_0|$")
        ax.legend(fontsize=8)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
