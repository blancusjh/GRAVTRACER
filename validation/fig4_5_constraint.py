"""Reproduce Figs. 4-5 of arXiv:2202.00086: Hamiltonian constraint error
along one escaping and one falling photon orbit around a Kerr black hole
with a = 0.98, for the three embedded RK integrators.

Usage: python validation/fig4_5_constraint.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grayt

# Initial conditions from the captions of Figs. 4 and 5 (a = 0.98,
# r0 = 100, theta0 = phi0 = pi/2).
ORBITS = {
    "escape (Fig. 4)": dict(y0=[0.0, 100.0, 1.570796, 1.570796,
                                -1.009327, 1.87],
                            p_t=-0.989953, p_phi=2.000098, lam=150.0),
    "fall (Fig. 5)": dict(y0=[0.0, 100.0, 1.570796, 1.570796,
                              -1.009314, 3.75],
                          p_t=-0.989952, p_phi=1.250061, lam=110.0),
}
METHODS = ["rkdp45", "rkck45", "rkf45"]


def main():
    bh = grayt.BlackHole(a=0.98)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, (name, orb) in zip(axes, ORBITS.items()):
        for method in METHODS:
            out = grayt.trace(bh, orb["y0"], p_t=orb["p_t"],
                              p_phi=orb["p_phi"], method=method,
                              rtol=1e-11, atol=1e-13,
                              lambda_max=orb["lam"])
            drift = np.abs(out["herr"] - out["herr"][0]) + 1e-18
            ax.semilogy(np.abs(out["lambda"]), drift, lw=0.9, label=method)
            print(f"{name} / {method}: steps={len(drift)}, "
                  f"max drift |H - H0| = {drift.max():.2e}, "
                  f"final r = {out['r'][-1]:.2f}")
        ax.set_title(name)
        ax.set_xlabel("$\\lambda$")
        ax.set_ylabel("$|H - H_0|$")
        ax.legend(fontsize=8)
    fig.savefig("validation/fig4_5_constraint.png", dpi=200,
                bbox_inches="tight")
    print("wrote validation/fig4_5_constraint.png")


if __name__ == "__main__":
    main()
