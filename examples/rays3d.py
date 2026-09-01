"""3D view of null geodesics around a black hole: a fan of rays launched
from an observer, trapped photons in black, escaping photons in colors
(the concept sketch of Fig. 1 of arXiv:2202.00086).

Usage: python examples/rays3d.py [-a SPIN] [--n-rays N]
"""
import argparse

import numpy as np

from _common import out

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grayt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--spin", type=float, default=0.9)
    ap.add_argument("--n-rays", type=int, default=46)
    ap.add_argument("--observer", type=float, nargs=3,
                    default=(18.0, -14.0, -9.0))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("-o", "--output", default=out("rays3d.png"))
    args = ap.parse_args()

    sys3 = grayt.System(physical=grayt.PhysicalSystem(
        spacetime=grayt.BlackHole(a=args.spin)))
    observer = np.asarray(args.observer)

    rng = np.random.default_rng(args.seed)
    target_dir = -observer/np.linalg.norm(observer)
    for _ in range(args.n_rays):
        d = target_dir + rng.normal(scale=0.16, size=3)
        sys3.trace_ray(observer, d, lambda_max=260.0)

    n_cap = sum(r.status == grayt.STATUS_CAPTURED for r in sys3.rays)
    print(f"traced {len(sys3.rays)} rays: {n_cap} captured, "
          f"{len(sys3.rays)-n_cap} escaped")
    ax = sys3.visualize3d(show_surfaces=False, elev=22, azim=-55)
    ax.scatter(*observer, s=90, color="tab:blue", edgecolor="black",
               zorder=5, label="observer")
    lim = 26
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_title("Null geodesics: trapped = black, escaping = colored")
    ax.legend(loc="upper left")
    ax.figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
