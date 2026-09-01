"""Gravitational lensing of a four-color celestial sphere.

Escaped backward rays are colored by the quadrant of the celestial
sphere they strike; the black mesh conveys the distortion. With the
default parameters this reproduces Fig. 12 of arXiv:2202.00086.

Usage: python examples/lensing_sphere.py [--spins A ...] [--res N]
"""
import argparse
import time

from _common import out

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grayt
from grayt.plotting import plot_lensing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spins", type=float, nargs="+", default=[0.0, 0.5, 0.98])
    ap.add_argument("--res", type=int, default=500)
    ap.add_argument("--r0", type=float, default=100.0)
    ap.add_argument("--fov", type=float, default=20.0)
    ap.add_argument("-o", "--output", default=out("lensing_sphere.png"))
    args = ap.parse_args()

    cam = grayt.Camera(r=args.r0, theta=90.0, x=(-args.fov, args.fov),
                       y=(-args.fov, args.fov),
                       resolution=(args.res, args.res))
    n = len(args.spins)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), constrained_layout=True,
                             squeeze=False)
    for ax, a in zip(axes[0], args.spins):
        t0 = time.time()
        img = grayt.shadow(grayt.BlackHole(a=a), cam, rtol=1e-9, atol=1e-11)
        print(f"a={a}: {time.time()-t0:.1f}s")
        plot_lensing(img, ax=ax)
        ax.set_title(f"$a = {a:g}$")
        ax.set_xlabel("$x$")
    axes[0, 0].set_ylabel("$y$")
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
