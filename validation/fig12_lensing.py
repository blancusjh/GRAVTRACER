"""Reproduce Fig. 12 of arXiv:2202.00086: gravitational lensing of a
four-color celestial sphere by Kerr black holes (a = 0, 0.5, 0.98),
observer in the equatorial plane at r0 = 100.

Usage: python validation/fig12_lensing.py [--res N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grayt
from grayt.plotting import plot_lensing

SPINS = [0.0, 0.5, 0.98]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=400)
    ap.add_argument("-o", "--output", default="validation/fig12_lensing.png")
    args = ap.parse_args()

    cam = grayt.Camera(r=100.0, theta=90.0, x=(-20, 20), y=(-20, 20),
                       resolution=(args.res, args.res))
    fig, axes = plt.subplots(1, len(SPINS), figsize=(5*len(SPINS), 5),
                             constrained_layout=True)
    for ax, a in zip(axes, SPINS):
        t0 = time.time()
        img = grayt.shadow(grayt.BlackHole(a=a), cam, rtol=1e-9, atol=1e-11)
        print(f"a={a}: {time.time()-t0:.1f}s")
        plot_lensing(img, ax=ax)
        ax.set_title(f"$a = {a:g}$")
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
