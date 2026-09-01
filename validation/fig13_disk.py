"""Reproduce Fig. 13 of arXiv:2202.00086: thin accretion disk around Kerr
black holes with a = 0, 0.5, 0.95.

All three panels share one normalization (the a = 0 maximum maps to 1),
so the colorbars show the relative brightening with spin, as in the paper.

Usage: python validation/fig13_disk.py [--res NX NY]
       (paper resolution: --res 2048 1024)
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

PANELS = [(0.0, 2.8), (0.5, 2.8), (0.95, 1.8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs=2, default=(1024, 512),
                    metavar=("NX", "NY"))
    ap.add_argument("-o", "--output", default="validation/fig13_disk.png")
    ap.add_argument("--npz-dir", default=None,
                    help="also save raw maps per spin")
    args = ap.parse_args()

    cam = grayt.Camera(r=1000.0, theta=85.0, x=(-24, 24), y=(-12, 12),
                       resolution=tuple(args.res))
    images = []
    for a, l0 in PANELS:
        bh = grayt.BlackHole(a=a)
        disk = grayt.ThinDisk(r_in=None, r_out=20.0, l0=l0)
        t0 = time.time()
        img = grayt.render(bh, cam, disk, rtol=1e-8, atol=1e-10)
        print(f"a={a}: rendered {args.res[0]}x{args.res[1]} in "
              f"{time.time()-t0:.1f}s, max I={img.intensity.max():.3e}, "
              f"max g={img.g.max():.3f}, max|H|={img.herr.max():.1e}")
        if args.npz_dir:
            Path(args.npz_dir).mkdir(parents=True, exist_ok=True)
            img.save(Path(args.npz_dir)/f"fig13_a{str(a).replace('.','')}.npz")
        images.append(img)

    norm = images[0].intensity.max()
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)
    for ax, (a, _), img in zip(axes, PANELS, images):
        img.plot(ax=ax, norm_to=norm, label=f"$a = {a:g}$")
    axes[-1].set_xlabel("$x\\;[M]$")
    for ax in axes:
        ax.set_ylabel("$y\\;[M]$")
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"wrote {args.output}")
    for (a, _), img in zip(PANELS, images):
        print(f"a={a}: colorbar max (relative to a=0) = "
              f"{img.intensity.max()/norm:.2f}")


if __name__ == "__main__":
    main()
