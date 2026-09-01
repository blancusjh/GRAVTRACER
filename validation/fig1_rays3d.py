"""Reproduce Fig. 1 of arXiv:2202.00086: 3D view of null geodesics
launched from an observer toward a black hole — trapped photons in black,
escaping photons in colors.

Usage: python validation/fig1_rays3d.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grayt


def main():
    ps = grayt.PhysicalSystem(black_hole=grayt.BlackHole(a=0.9))
    observer = np.array([18.0, -14.0, -9.0])

    # Fan of rays aimed at (and around) the black hole
    rng = np.random.default_rng(7)
    target_dir = -observer/np.linalg.norm(observer)
    for _ in range(46):
        jitter = rng.normal(scale=0.16, size=3)
        d = target_dir + jitter
        ps.trace_ray(observer, d, lambda_max=260.0)

    n_cap = sum(r.status == grayt.STATUS_CAPTURED for r in ps.rays)
    print(f"traced {len(ps.rays)} rays: {n_cap} captured, "
          f"{len(ps.rays)-n_cap} escaped")

    sys3 = grayt.System(physical=ps)
    ax = sys3.visualize3d(show_surfaces=False, elev=22, azim=-55)
    ax.scatter(*observer, s=90, color="tab:blue", edgecolor="black",
               zorder=5, label="observer")
    lim = 26
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_title("Backward ray tracing (Fig. 1): trapped = black, "
                 "escaping = colored")
    ax.legend(loc="upper left")
    ax.figure.savefig("validation/fig1_rays3d.png", dpi=180,
                      bbox_inches="tight")
    print("wrote validation/fig1_rays3d.png")


if __name__ == "__main__":
    main()
